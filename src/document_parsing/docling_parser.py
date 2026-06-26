from pathlib import Path

from docling.document_converter import DocumentConverter

from src.document_parsing.models import DocumentElement, ParsedDocument
from src.document_parsing.utils import compute_file_hash


# Docling 用来处理 Office 文档和复杂 PDF，依赖较重，所以放在阶段 2。
def parse_with_docling(path: Path) -> ParsedDocument:
    try:
        converter = DocumentConverter()
        result = converter.convert(str(path))
        markdown_text = result.document.export_to_markdown()

        return ParsedDocument(
            doc_id=compute_file_hash(path),
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="docling",
            status="success",
            file_hash=compute_file_hash(path),
            elements=[
                DocumentElement(
                    text=markdown_text,
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
            parser_name="docling",
            status="failed",
            error=str(exc),
        )