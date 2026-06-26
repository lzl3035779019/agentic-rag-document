from pathlib import Path


# 根据文件后缀选择解析器名称。第一版先按后缀路由，后续可以升级为 MIME 检测。
def select_parser_name(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".md", ".txt"}:
        return "markdown"

    if suffix == ".pdf":
        return "pymupdf"

    if suffix in {".docx", ".pptx", ".xlsx", ".html", ".htm"}:
        return "docling"

    if suffix in {".png", ".jpg", ".jpeg"}:
        return "ocr"

    return "unsupported"