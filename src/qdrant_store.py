from threading import Lock

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config import EMBEDDING_MODEL, TOP_K_DENSE
from src.knowledge_base import KnowledgeBaseConfig, get_default_knowledge_base
from src.parent_child_splitter import build_parent_child_chunks


COLLECTION_NAME = "basecamp_handbook_visualized"
QDRANT_URL = "http://localhost:6333"

_embedding_models = {}
_embedding_model_lock = Lock()


def get_embedding_model(model_name: str | None = None):
    selected_model = model_name or EMBEDDING_MODEL
    if selected_model not in _embedding_models:
        with _embedding_model_lock:
            if selected_model not in _embedding_models:
                # Parallel graph execution can initialize retrieval in multiple
                # threads. Serialize first load to avoid PyTorch meta-tensor errors.
                _embedding_models[selected_model] = HuggingFaceEmbeddings(
                    model_name=selected_model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
    return _embedding_models[selected_model]


def build_qdrant_store(kb_config: KnowledgeBaseConfig | None = None):
    kb = kb_config or get_default_knowledge_base()
    child_docs, parent_map = build_parent_child_chunks(kb)
    embeddings = get_embedding_model(kb.embedding_model)

    vector_store = QdrantVectorStore.from_documents(
        documents=child_docs,
        embedding=embeddings,
        url=QDRANT_URL,
        collection_name=kb.collection_name,
        force_recreate=True,
    )

    return vector_store, parent_map


def get_qdrant_retriever(kb_config: KnowledgeBaseConfig | None = None):
    kb = kb_config or get_default_knowledge_base()
    client = QdrantClient(url=QDRANT_URL)
    embeddings = get_embedding_model(kb.embedding_model)

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=kb.collection_name,
        embedding=embeddings,
    )

    return vector_store.as_retriever(search_kwargs={"k": TOP_K_DENSE})


if __name__ == "__main__":
    build_qdrant_store()
    retriever = get_qdrant_retriever()

    docs = retriever.invoke("What are the main steps in this workflow?")
    for doc in docs:
        print("=" * 60)
        print(doc.metadata)
        print(doc.page_content[:500])
