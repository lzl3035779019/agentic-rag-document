from pathlib import Path

import pymupdf4llm

from src.document_parsing.models import DocumentElement, ParsedDocument
from src.document_parsing.utils import compute_file_hash


# 普通文本 PDF 走 PyMuPDF4LLM，优点是轻、快、适合先验证 PDF 解析链路。
def parse_pdf_with_pymupdf(path: Path) -> ParsedDocument:
    try:
        markdown_text = pymupdf4llm.to_markdown(str(path))

        return ParsedDocument(
            doc_id=compute_file_hash(path),
            source_path=path,
            file_name=path.name,
            file_type=".pdf",
            parser_name="pymupdf4llm",
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
            file_type=".pdf",
            parser_name="pymupdf4llm",
            status="failed",
            error=str(exc),
        )