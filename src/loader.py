# from pathlib import Path
#
# from langchain_core.documents import Document
#
# from src.config import DATA_PATH
#
#
#
# def load_documents() -> list[Document]:
#     data_dir = Path(DATA_PATH)
#     # print(f"数据目录: {data_dir}")
#     docs = []
#
#     for path in data_dir.rglob("*.md"):
#         text = path.read_text(encoding="utf-8")
#         docs.append(
#             Document(
#                 page_content=text,
#                 metadata={"source": str(path)},
#             )
#         )
#
#     return docs

from pathlib import Path

from langchain_core.documents import Document

from src.config import PARSED_DOCS_DIR


# 把 parsed markdown 的 frontmatter 拆成 metadata，避免元数据参与 embedding。
def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text

    end = text.find("\n---\n", 4)
    if end == -1:
        return {}, text

    raw_meta = text[4:end]
    body = text[end + len("\n---\n"):]

    metadata = {}
    for line in raw_meta.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()

    return metadata, body.strip()


# RAG 主链路只读取 parsed_docs，并把 frontmatter 放入 Document.metadata。
def load_documents(parsed_docs_dir: Path | str = PARSED_DOCS_DIR) -> list[Document]:
    docs = []
    docs_dir = Path(parsed_docs_dir)

    for path in docs_dir.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        frontmatter, body = split_frontmatter(text)

        metadata = {
            "source": frontmatter.get("source_path", str(path)),
            "parsed_path": str(path),
            "source_type": "parsed_markdown",
            **frontmatter,
        }

        docs.append(
            Document(
                page_content=body,
                metadata=metadata,
            )
        )

    return docs


if __name__ == "__main__":
    docs = load_documents()
    print(f"documents: {len(docs)}")

    for doc in docs[:3]:
        print("=" * 60)
        print(doc.metadata)
        print(doc.page_content[:500])
