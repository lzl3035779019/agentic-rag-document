import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.document_parsing.models import ParsedDocument

PARSER_VERSION = "v1"


# 追加一条解析日志。jsonl 一行一个文件，适合批处理和失败排查。
def append_manifest_record(
    manifest_path: Path,
    doc: ParsedDocument,
    output_path: Path | None,
    duration_ms: int,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    record: dict[str, Any] = {
        "doc_id": doc.doc_id,
        "source_path": str(doc.source_path),
        "output_path": str(output_path) if output_path else None,
        "file_name": doc.file_name,
        "file_type": doc.file_type,
        "parser_name": doc.parser_name,
        "parser_version": PARSER_VERSION,
        "status": doc.status,
        "error": doc.error,
        "file_hash": doc.file_hash,
        "element_count": len(doc.elements),
        "text_length": len(doc.text),
        "duration_ms": duration_ms,
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }

    with manifest_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 读取每个 source_path 最近一次成功记录，用于判断是否跳过未变化文件。
def load_latest_success_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}

    records = {}

    with manifest_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "success" and record.get("file_hash"):
                records[record["source_path"]] = record

    return records