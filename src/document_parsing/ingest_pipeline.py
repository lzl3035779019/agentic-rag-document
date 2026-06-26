import time
from pathlib import Path

from src.config import PROJECT_ROOT
from src.document_parsing.cleaner import clean_parsed_document
from src.document_parsing.file_discovery import discover_files
from src.document_parsing.manifest import append_manifest_record, load_latest_success_records
from src.document_parsing.parser_service import parse_file
from src.document_parsing.utils import compute_file_hash, safe_filename


RAW_DOCS_DIR = PROJECT_ROOT / "data" / "raw_docs"
PARSED_DOCS_DIR = PROJECT_ROOT / "data" / "parsed_docs"
FAILED_DOCS_DIR = PROJECT_ROOT / "data" / "failed_docs"
MANIFEST_PATH = PARSED_DOCS_DIR / "manifest.jsonl"


# 文件重新解析前删除旧 parsed markdown，避免 loader 同时读到旧版本和新版本。
def remove_old_output(previous_record: dict | None) -> None:
    if not previous_record:
        return

    output_path = previous_record.get("output_path")
    if not output_path:
        return

    path = Path(output_path)
    if path.exists():
        path.unlink()


# 把统一解析结果保存为 Markdown；frontmatter 存元数据，正文存可 embedding 内容。
def save_parsed_markdown(doc) -> Path:
    PARSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = safe_filename(doc.source_path.stem)
    output_path = PARSED_DOCS_DIR / f"{safe_name}.{doc.doc_id[:8]}.md"

    content = [
        "---",
        f"doc_id: {doc.doc_id}",
        f"source_path: {doc.source_path}",
        f"file_name: {doc.file_name}",
        f"file_type: {doc.file_type}",
        f"parser_name: {doc.parser_name}",
        "---",
        "",
        doc.text,
    ]

    output_path.write_text("\n".join(content), encoding="utf-8")
    return output_path


# 一键 ingest 主流程：发现文件、跳过未变文件、解析、清洗、保存、写 manifest。
def run_ingest(force: bool = False) -> None:
    PARSED_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DOCS_DIR.mkdir(parents=True, exist_ok=True)

    latest_records = load_latest_success_records(MANIFEST_PATH)
    files = discover_files(RAW_DOCS_DIR)

    print(f"discovered files: {len(files)}")

    for path in files:
        started = time.perf_counter()
        current_hash = compute_file_hash(path)
        previous_record = latest_records.get(str(path))
        previous_hash = previous_record.get("file_hash") if previous_record else None

        if not force and previous_hash == current_hash:
            print(f"skip unchanged: {path}")
            continue

        print(f"parse: {path}")
        remove_old_output(previous_record)

        doc = parse_file(path)
        doc.file_hash = current_hash

        output_path = None
        if doc.status == "success":
            doc = clean_parsed_document(doc)
            output_path = save_parsed_markdown(doc)
        else:
            print(f"failed: {path} - {doc.error}")

        duration_ms = int((time.perf_counter() - started) * 1000)
        append_manifest_record(MANIFEST_PATH, doc, output_path, duration_ms)


if __name__ == "__main__":
    run_ingest(force=False)