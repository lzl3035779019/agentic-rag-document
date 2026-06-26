from langchain_core.documents import Document


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[source: {doc.metadata.get('source')}, chunk: {doc.metadata.get('chunk_id')}]\n{doc.page_content}"
        for doc in docs
    )
