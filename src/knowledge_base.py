from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    EMBEDDING_MODEL,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
    PARSED_DOCS_DIR,
    PROJECT_ROOT,
)


DEFAULT_KB_ID = "basecamp"
DEFAULT_COLLECTION_NAME = "basecamp_handbook_visualized"
KB_ROOT_DIR = PROJECT_ROOT / "data" / "knowledge_bases"
KB_REGISTRY_PATH = KB_ROOT_DIR / "registry.json"

EMBEDDING_MODEL_OPTIONS = {
    "English lightweight": "sentence-transformers/all-MiniLM-L6-v2",
    "English retrieval": "BAAI/bge-small-en-v1.5",
    "Chinese retrieval": "BAAI/bge-small-zh-v1.5",
    "Multilingual retrieval": "BAAI/bge-m3",
}


@dataclass(frozen=True)
class KnowledgeBaseConfig:
    kb_id: str
    name: str
    collection_name: str
    embedding_model: str
    language_strategy: str
    parent_chunk_size: int
    parent_chunk_overlap: int
    child_chunk_size: int
    child_chunk_overlap: int
    raw_dir: Path
    parsed_dir: Path
    file_names: list[str]
    created_at: str
    is_default: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["raw_dir"] = str(self.raw_dir)
        data["parsed_dir"] = str(self.parsed_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeBaseConfig":
        payload = dict(data)
        payload["raw_dir"] = Path(payload["raw_dir"])
        payload["parsed_dir"] = Path(payload["parsed_dir"])
        return cls(**payload)

    def cache_key(self) -> tuple[Any, ...]:
        return (
            self.kb_id,
            self.collection_name,
            self.embedding_model,
            self.language_strategy,
            self.parent_chunk_size,
            self.parent_chunk_overlap,
            self.child_chunk_size,
            self.child_chunk_overlap,
            str(self.parsed_dir),
        )


def get_default_knowledge_base() -> KnowledgeBaseConfig:
    return KnowledgeBaseConfig(
        kb_id=DEFAULT_KB_ID,
        name="Basecamp handbook",
        collection_name=DEFAULT_COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        language_strategy="en",
        parent_chunk_size=PARENT_CHUNK_SIZE,
        parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
        child_chunk_size=CHILD_CHUNK_SIZE,
        child_chunk_overlap=CHILD_CHUNK_OVERLAP,
        raw_dir=PARSED_DOCS_DIR,
        parsed_dir=PARSED_DOCS_DIR,
        file_names=[],
        created_at="built-in",
        is_default=True,
    )


def _slugify(value: str, separator: str = "-") -> str:
    cjk_map = {
        "中": "zhong",
        "文": "wen",
        "手": "shou",
        "册": "ce",
    }
    parts: list[str] = []
    for char in value.lower():
        if char in cjk_map:
            parts.append(separator + cjk_map[char] + separator)
        elif char.isascii() and char.isalnum():
            parts.append(char)
        else:
            parts.append(separator)
    slug = re.sub(f"{re.escape(separator)}+", separator, "".join(parts))
    return slug.strip(separator) or "knowledge-base"


def create_knowledge_base_config(
    *,
    name: str,
    root_dir: Path = KB_ROOT_DIR,
    embedding_model: str,
    language_strategy: str,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
    file_names: list[str],
) -> KnowledgeBaseConfig:
    base_slug = _slugify(name, "-")
    now = datetime.now(UTC)
    suffix = now.strftime("%Y%m%d%H%M%S")
    kb_id = f"{base_slug}-{suffix}"
    collection_slug = _slugify(name, "_")
    collection_name = f"kb_{collection_slug}_{suffix}"

    return KnowledgeBaseConfig(
        kb_id=kb_id,
        name=name.strip() or "Knowledge base",
        collection_name=collection_name,
        embedding_model=embedding_model,
        language_strategy=language_strategy,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        raw_dir=root_dir / kb_id / "raw",
        parsed_dir=root_dir / kb_id / "parsed",
        file_names=file_names,
        created_at=now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    )


def load_custom_knowledge_bases(registry_path: Path = KB_REGISTRY_PATH) -> list[KnowledgeBaseConfig]:
    if not registry_path.exists():
        return []
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    return [KnowledgeBaseConfig.from_dict(item) for item in data.get("knowledge_bases", [])]


def save_custom_knowledge_bases(
    knowledge_bases: list[KnowledgeBaseConfig],
    registry_path: Path = KB_REGISTRY_PATH,
) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    data = {"knowledge_bases": [kb.to_dict() for kb in knowledge_bases if not kb.is_default]}
    registry_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def list_knowledge_bases() -> list[KnowledgeBaseConfig]:
    return [get_default_knowledge_base(), *load_custom_knowledge_bases()]


def get_knowledge_base(kb_id: str | None) -> KnowledgeBaseConfig:
    if not kb_id or kb_id == DEFAULT_KB_ID:
        return get_default_knowledge_base()
    for kb in load_custom_knowledge_bases():
        if kb.kb_id == kb_id:
            return kb
    return get_default_knowledge_base()


def register_knowledge_base(kb: KnowledgeBaseConfig) -> None:
    custom = [item for item in load_custom_knowledge_bases() if item.kb_id != kb.kb_id]
    custom.append(kb)
    save_custom_knowledge_bases(custom)
