from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

ElementType = Literal["text", "title", "table", "image", "unknown"]
ParseStatus = Literal["success", "failed", "skipped"]


# 表示文档里的一个最小内容单元，比如一段正文、一张图片 OCR 结果、一个表格文本。
@dataclass
class DocumentElement:
    text: str
    element_type: ElementType = "text"
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# 表示一个原始文件解析后的统一结果，后续所有 parser 都返回这个结构。
@dataclass
class ParsedDocument:
    doc_id: str
    source_path: Path
    file_name: str
    file_type: str
    parser_name: str  # 解析器名称，记录用哪个解析引擎处理的该文件。
    status: ParseStatus  # 文档整体解析状态，只能是 success/failed/skipped。
    elements: list[DocumentElement] = field(default_factory=list)  # 文档所有内容块列表，每个元素都是 DocumentElement 对象，默认空列表。
    error: str | None = None  # 解析失败时的错误信息 / 异常描述；解析正常则为 None。
    file_hash: str | None = None  # 文件哈希值（MD5/SHA 等），用于文件去重、校验文件完整性，默认空。
    metadata: dict[str, Any] = field(default_factory=dict)  # 整篇文档的扩展元数据（创建时间、作者、页数、加密信息等），默认空字典。

    @property
    def text(self) -> str:
        return "\n\n".join(
            element.text.strip()
            for element in self.elements
            if element.text and element.text.strip()
        )
