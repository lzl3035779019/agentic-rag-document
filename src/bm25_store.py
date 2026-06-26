import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from src.config import TOP_K_BM25
from src.knowledge_base import KnowledgeBaseConfig, get_default_knowledge_base
from src.parent_child_splitter import build_parent_child_chunks


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())


class BM25Store:
    def __init__(self, docs: list[Document]):
        self.docs = docs
        self.tokens = [tokenize(doc.page_content) for doc in docs]
        self.index = BM25Okapi(self.tokens) # 建立关键词索引库

    def search(self, query: str, k: int = TOP_K_BM25) -> list[Document]:
        query_tokens = tokenize(query)
        scores = self.index.get_scores(query_tokens)
        ranked = sorted(
            enumerate(scores),        # [(0, 3.25), (1, 2.5), ...]
            key=lambda item: item[1],  # 用「第 2 个元素」来排序！
            reverse=True,
        )

        results = []
        for doc_index, score in ranked[:k]:
            doc = self.docs[doc_index]
            doc.metadata["bm25_score"] = float(score)
            results.append(doc)

        return results


_bm25_cache: dict[tuple, BM25Store] = {}


def build_bm25_store(kb_config: KnowledgeBaseConfig | None = None):
    kb = kb_config or get_default_knowledge_base()
    cache_key = kb.cache_key()
    if cache_key not in _bm25_cache:
        child_docs, _ = build_parent_child_chunks(kb)
        _bm25_cache[cache_key] = BM25Store(child_docs)
    return _bm25_cache[cache_key]


if __name__ == "__main__":
    store = build_bm25_store()
    results = store.search("What does Basecamp say about work devices?")

    for doc in results:
        print("=" * 40)
        print(doc.metadata)
        print(doc.page_content[:500])
