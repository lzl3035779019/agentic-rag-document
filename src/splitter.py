from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP
from src.loader import load_documents


def split_documents():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=PARENT_CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", "。", "，", " "],
    )

    chunks = splitter.split_documents(docs)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


if __name__ == "__main__":
    chunks = split_documents()
    print(f"切块数量: {len(chunks)}")

    for chunk in chunks[:5]:
        print("=" * 40)
        print(chunk.metadata)
        print(chunk.page_content[:300])