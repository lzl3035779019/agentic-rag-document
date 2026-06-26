from functools import lru_cache
from pathlib import Path
from typing import Any

from paddleocr import PaddleOCR

from src.document_parsing.models import DocumentElement, ParsedDocument
from src.document_parsing.utils import compute_file_hash


# OCR 引擎初始化较重，用 lru_cache 保证整个进程只创建一次。
@lru_cache(maxsize=1)
def get_ocr_engine() -> PaddleOCR:
    try:
        return PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            engine="paddle",
        )
    except TypeError:
        return PaddleOCR(
            use_angle_cls=True,
            lang="ch",
            show_log=False,
        )


# PaddleOCR 2.x/3.x 返回结构不同，这里用递归方式尽量兼容提取文本。
def _extract_text_from_any(value: Any) -> list[str]:
    texts = []

    if value is None:
        return texts

    if isinstance(value, str):
        return [value]

    if isinstance(value, dict):
        if "rec_texts" in value and isinstance(value["rec_texts"], list):
            texts.extend(str(item) for item in value["rec_texts"] if item)
        if "rec_text" in value and value["rec_text"]:
            texts.append(str(value["rec_text"]))
        if "res" in value:
            texts.extend(_extract_text_from_any(value["res"]))
        return texts

    if hasattr(value, "json"):
        data = value.json() if callable(value.json) else value.json
        texts.extend(_extract_text_from_any(data))
        return texts

    if hasattr(value, "res"):
        texts.extend(_extract_text_from_any(value.res))
        return texts

    if isinstance(value, (list, tuple)):
        if len(value) >= 2 and isinstance(value[1], (list, tuple)) and value[1]:
            candidate = value[1][0]
            if isinstance(candidate, str):
                texts.append(candidate)
                return texts
        for item in value:
            texts.extend(_extract_text_from_any(item))

    return texts


# PaddleOCR 3.x 优先使用 predict；旧版没有 predict 时回退到 ocr。
def _run_ocr(ocr: PaddleOCR, path: Path):
    if hasattr(ocr, "predict"):
        return ocr.predict(str(path))
    return ocr.ocr(str(path), cls=True)


# 图片 OCR 解析器：把图片中的文字抽取出来，再包装成 ParsedDocument。
def parse_image_with_ocr(path: Path) -> ParsedDocument:
    try:
        ocr = get_ocr_engine()
        result = _run_ocr(ocr, path)
        text = "\n".join(_extract_text_from_any(result)).strip()

        return ParsedDocument(
            doc_id=compute_file_hash(path),
            source_path=path,
            file_name=path.name,
            file_type=path.suffix.lower(),
            parser_name="paddleocr",
            status="success",
            file_hash=compute_file_hash(path),
            elements=[
                DocumentElement(
                    text=text,
                    element_type="image",
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
            parser_name="paddleocr",
            status="failed",
            error=str(exc),
        )