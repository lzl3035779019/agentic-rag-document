import argparse
import hashlib
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import CACHE_NAMESPACE, LLM_MODEL, PROJECT_ROOT
from src.llm import get_llm


CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "llm_cache.sqlite"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _table_columns(conn: sqlite3.Connection) -> set[str]:
    rows = conn.execute("PRAGMA table_info(llm_cache)").fetchall()
    return {row[1] for row in rows}


def _ensure_column(conn: sqlite3.Connection, columns: set[str], name: str, ddl: str) -> None:
    if name not in columns:
        conn.execute(f"ALTER TABLE llm_cache ADD COLUMN {ddl}")


def init_cache() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                cache_namespace TEXT NOT NULL,
                task_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                prompt_hash TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                created_at TEXT NOT NULL,
                last_accessed_at TEXT NOT NULL,
                hit_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        columns = _table_columns(conn)
        now = utc_now()
        _ensure_column(conn, columns, "cache_namespace", f"cache_namespace TEXT NOT NULL DEFAULT '{CACHE_NAMESPACE}'")
        _ensure_column(conn, columns, "last_accessed_at", f"last_accessed_at TEXT NOT NULL DEFAULT '{now}'")
        _ensure_column(conn, columns, "hit_count", "hit_count INTEGER NOT NULL DEFAULT 0")
        conn.execute(
            """
            UPDATE llm_cache
            SET cache_key = ? || ':' || cache_key
            WHERE cache_namespace = ?
              AND cache_key NOT LIKE ? || ':%'
            """,
            (CACHE_NAMESPACE, CACHE_NAMESPACE, CACHE_NAMESPACE),
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_cache_task_created
            ON llm_cache(task_name, created_at)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_llm_cache_namespace_access
            ON llm_cache(cache_namespace, last_accessed_at)
            """
        )


def prompt_to_text(prompt: Any) -> str:
    if hasattr(prompt, "to_string"):
        return prompt.to_string()
    return str(prompt)


def build_cache_key(task_name: str, prompt_text: str) -> tuple[str, str]:
    prompt_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
    cache_key = f"{CACHE_NAMESPACE}:{LLM_MODEL}:{task_name}:{prompt_hash}"
    return cache_key, prompt_hash


def get_cached_response(task_name: str, prompt_text: str) -> str | None:
    init_cache()
    cache_key, _ = build_cache_key(task_name, prompt_text)
    with sqlite3.connect(CACHE_PATH) as conn:
        row = conn.execute(
            "SELECT response FROM llm_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row:
            conn.execute(
                """
                UPDATE llm_cache
                SET last_accessed_at = ?, hit_count = hit_count + 1
                WHERE cache_key = ?
                """,
                (utc_now(), cache_key),
            )
    return row[0] if row else None


def save_cached_response(task_name: str, prompt_text: str, response: str) -> None:
    init_cache()
    cache_key, prompt_hash = build_cache_key(task_name, prompt_text)
    now = utc_now()
    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO llm_cache (
                cache_key,
                cache_namespace,
                task_name,
                model_name,
                prompt_hash,
                prompt,
                response,
                created_at,
                last_accessed_at,
                hit_count
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cache_key,
                CACHE_NAMESPACE,
                task_name,
                LLM_MODEL,
                prompt_hash,
                prompt_text,
                response,
                now,
                now,
                0,
            ),
        )


def invoke_llm_cached(task_name: str, prompt: Any) -> str:
    prompt_text = prompt_to_text(prompt)
    cached = get_cached_response(task_name, prompt_text)
    if cached is not None:
        return cached

    response = get_llm().invoke(prompt)
    content = response.content
    save_cached_response(task_name, prompt_text, content)
    return content


def _stream_chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", chunk)
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _cached_response_chunks(text: str, chunk_size: int = 24):
    for index in range(0, len(text), chunk_size):
        yield text[index:index + chunk_size]


def invoke_llm_cached_stream(task_name: str, prompt: Any):
    prompt_text = prompt_to_text(prompt)
    cached = get_cached_response(task_name, prompt_text)
    if cached is not None:
        yield from _cached_response_chunks(cached)
        return

    parts = []
    for chunk in get_llm().stream(prompt):
        text = _stream_chunk_text(chunk)
        if not text:
            continue
        parts.append(text)
        yield text

    save_cached_response(task_name, prompt_text, "".join(parts))


def cache_stats() -> list[tuple[str, int, int]]:
    init_cache()
    with sqlite3.connect(CACHE_PATH) as conn:
        return conn.execute(
            """
            SELECT task_name, COUNT(*), COALESCE(SUM(hit_count), 0)
            FROM llm_cache
            WHERE cache_namespace = ?
            GROUP BY task_name
            ORDER BY task_name
            """,
            (CACHE_NAMESPACE,),
        ).fetchall()


def prune_older_than(days: int) -> int:
    init_cache()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    with sqlite3.connect(CACHE_PATH) as conn:
        cursor = conn.execute(
            """
            DELETE FROM llm_cache
            WHERE cache_namespace = ? AND last_accessed_at < ?
            """,
            (CACHE_NAMESPACE, cutoff.isoformat()),
        )
        return cursor.rowcount


def prune_max_entries(max_entries: int) -> int:
    init_cache()
    with sqlite3.connect(CACHE_PATH) as conn:
        total = conn.execute(
            "SELECT COUNT(*) FROM llm_cache WHERE cache_namespace = ?",
            (CACHE_NAMESPACE,),
        ).fetchone()[0]
        overflow = max(0, total - max_entries)
        if overflow == 0:
            return 0

        cursor = conn.execute(
            """
            DELETE FROM llm_cache
            WHERE cache_key IN (
                SELECT cache_key
                FROM llm_cache
                WHERE cache_namespace = ?
                ORDER BY last_accessed_at ASC, hit_count ASC
                LIMIT ?
            )
            """,
            (CACHE_NAMESPACE, overflow),
        )
        return cursor.rowcount


def clear_cache() -> int:
    init_cache()
    with sqlite3.connect(CACHE_PATH) as conn:
        cursor = conn.execute(
            "DELETE FROM llm_cache WHERE cache_namespace = ?",
            (CACHE_NAMESPACE,),
        )
        return cursor.rowcount


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or clear the local SQLite LLM cache.")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--prune-days", type=int, default=None)
    parser.add_argument("--prune-max-entries", type=int, default=None)
    args = parser.parse_args()

    if args.clear:
        deleted = clear_cache()
        print(f"cleared {deleted} entries from namespace={CACHE_NAMESPACE}: {CACHE_PATH}")
        return

    if args.stats:
        print(f"cache_path: {CACHE_PATH}")
        print(f"namespace: {CACHE_NAMESPACE}")
        for task_name, count, hits in cache_stats():
            print(f"{task_name}: entries={count}, hits={hits}")
        return

    if args.prune_days is not None:
        deleted = prune_older_than(args.prune_days)
        print(f"deleted {deleted} entries older than {args.prune_days} days")
        return

    if args.prune_max_entries is not None:
        deleted = prune_max_entries(args.prune_max_entries)
        print(f"deleted {deleted} entries to keep max_entries={args.prune_max_entries}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
