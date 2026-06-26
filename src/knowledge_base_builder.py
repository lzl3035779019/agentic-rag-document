from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.document_parsing.cleaner import clean_parsed_document
from src.document_parsing.parser_service import parse_file
from src.document_parsing.utils import safe_filename
from src.knowledge_base import KnowledgeBaseConfig, register_knowledge_base
from src.qdrant_store import build_qdrant_store


SUPPORTED_SUFFIXES = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}


def is_supported_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in SUPPORTED_SUFFIXES


def _safe_upload_filename(filename: str) -> str:
    name = Path(filename).name.strip()
    if not name:
        return "uploaded.txt"
    return name


def _parsed_markdown_content(doc) -> str:
    return (
        "---\n"
        f"doc_id: {doc.doc_id}\n"
        f"source_path: {doc.source_path}\n"
        f"file_name: {doc.file_name}\n"
        f"file_type: {doc.file_type}\n"
        f"parser_name: {doc.parser_name}\n"
        "---\n\n"
        f"{doc.text}"
    )


def _write_parsed_markdown(kb_config: KnowledgeBaseConfig, raw_path: Path, doc) -> Path:
    parsed_name = f"{safe_filename(raw_path.stem)}.{doc.doc_id[:8]}.md"
    parsed_path = kb_config.parsed_dir / parsed_name
    parsed_path.write_text(_parsed_markdown_content(doc), encoding="utf-8")
    return parsed_path


def write_uploaded_files(
    kb_config: KnowledgeBaseConfig,
    files: list[tuple[str, bytes]],
) -> list[str]:
    kb_config.raw_dir.mkdir(parents=True, exist_ok=True)
    kb_config.parsed_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for filename, content in files:
        safe_name = _safe_upload_filename(filename)
        if not is_supported_file(safe_name):
            continue

        raw_path = kb_config.raw_dir / safe_name
        raw_path.write_bytes(content)

        doc = parse_file(raw_path)
        if doc.status != "success":
            raise ValueError(f"Failed to parse {safe_name}: {doc.error}")
        doc = clean_parsed_document(doc)
        if not doc.text.strip():
            raise ValueError(f"Parsed document is empty: {safe_name}")
        _write_parsed_markdown(kb_config, raw_path, doc)
        saved_files.append(safe_name)

    return saved_files


def write_uploaded_text_files(
    kb_config: KnowledgeBaseConfig,
    files: list[tuple[str, bytes]],
) -> list[str]:
    return write_uploaded_files(kb_config, files)


def build_and_register_knowledge_base(
    kb_config: KnowledgeBaseConfig,
    files: list[tuple[str, bytes]],
) -> KnowledgeBaseConfig:
    saved_files = write_uploaded_files(kb_config, files)
    if not saved_files:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise ValueError(f"No supported files were uploaded. Supported types: {supported}.")

    kb_config = replace(kb_config, file_names=saved_files)
    build_qdrant_store(kb_config)
    register_knowledge_base(kb_config)
    return kb_config
