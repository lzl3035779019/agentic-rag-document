from pathlib import Path

from src.config import PROJECT_ROOT
from src.document_parsing.models import DocumentElement, ParsedDocument
from src.document_parsing.utils import compute_file_hash


# Markdown/TXT 是最轻量的解析器：直接读取文本并包装成统一 ParsedDocument。
def parse_markdown_or_text(path: Path) -> ParsedDocument:
    try:
        text = path.read_text(encoding="utf-8")
        return ParsedDocument(
            doc_id=compute_file_hash(path),
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="markdown",
            status="success",
            file_hash=compute_file_hash(path),
            elements=[
                DocumentElement(
                    text=text,
                    element_type="text",
                    metadata={"source": str(path)},
                )
            ],
        )
    except Exception as exc:
        return ParsedDocument(
            doc_id=path.name,
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="markdown",
            status="failed",
            error=str(exc),
        )

if __name__ == "__main__":
    path = PROJECT_ROOT / "data/handbook/moonlighting.md"
    doc = parse_markdown_or_text(Path(path))
    print(doc.status)
    print(doc.text[:500])