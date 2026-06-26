import re

from src.document_parsing.models import DocumentElement, ParsedDocument


# 轻量清洗文本：统一换行、压缩空白，避免过度清洗误删有效内容。
def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    return text.strip()


# 对一个 ParsedDocument 里的所有 element 做清洗，并丢弃空文本 element。
def clean_parsed_document(doc: ParsedDocument) -> ParsedDocument:
    cleaned_elements = []

    for element in doc.elements:
        cleaned = clean_text(element.text)
        if not cleaned:
            continue

        cleaned_elements.append(
            DocumentElement(
                text=cleaned,
                element_type=element.element_type,
                page_number=element.page_number,
                metadata=element.metadata,
            )
        )

    doc.elements = cleaned_elements
    return doc