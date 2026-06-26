from pathlib import Path

from src.document_parsing.markdown_parser import parse_markdown_or_text
from src.document_parsing.models import ParsedDocument
from src.document_parsing.parser_registry import select_parser_name
from src.document_parsing.pymupdf_parser import parse_pdf_with_pymupdf


# lazy import：只有真的需要 Docling 时才导入，避免阶段 1 没装 Docling 就整体崩溃。
def _parse_with_docling_lazy(path: Path) -> ParsedDocument:
    try:
        from src.document_parsing.docling_parser import parse_with_docling
    except ImportError as exc:
        return ParsedDocument(
            doc_id=path.name,
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="docling",
            status="failed",
            error=f"Docling is not installed: {exc}",
        )

    return parse_with_docling(path)


# lazy import：只有解析图片时才导入 PaddleOCR，避免 OCR 依赖影响普通文档解析。
def _parse_with_ocr_lazy(path: Path) -> ParsedDocument:
    try:
        from src.document_parsing.ocr_parser import parse_image_with_ocr
    except ImportError as exc:
        return ParsedDocument(
            doc_id=path.name,
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="paddleocr",
            status="failed",
            error=f"OCR dependencies are not installed: {exc}",
        )

    return parse_image_with_ocr(path)


# 统一解析入口：根据文件类型选择 parser，并在 PDF 解析失败时执行 fallback。
def parse_file(path: Path) -> ParsedDocument:
    parser_name = select_parser_name(path)
    suffix = path.suffix.lower()

    if parser_name == "markdown":
        return parse_markdown_or_text(path)

    if suffix == ".pdf":
        result = parse_pdf_with_pymupdf(path)
        if result.status == "success" and result.text.strip():
            return result
        return _parse_with_docling_lazy(path)

    if parser_name == "docling":
        return _parse_with_docling_lazy(path)

    if parser_name == "ocr":
        return _parse_with_ocr_lazy(path)

    return ParsedDocument(
        doc_id=path.name,
        source_path=path,
        file_name=path.name,
        file_type=suffix,
        parser_name="unsupported",
        status="skipped",
        error=f"Unsupported file type: {suffix}",
    )