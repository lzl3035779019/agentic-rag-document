from langchain_core.documents import Document

from src.bm25_store import build_bm25_store
from src.config import FINAL_TOP_K
from src.knowledge_base import KnowledgeBaseConfig, get_default_knowledge_base
from src.parent_child_splitter import build_parent_child_chunks
from src.qdrant_store import get_qdrant_retriever


HEADING_ONLY_MAX_CHARS = 120
HEADING_EXPANSION_TARGET_CHARS = 6000
HEADING_EXPANSION_MAX_PARENTS = 5
PRIMARY_SOURCE_NEIGHBOR_PARENTS = 5


def _parent_index(parent_id: str) -> int | None:
    if not parent_id.startswith("parent-"):
        return None
    try:
        return int(parent_id.split("-", 1)[1])
    except ValueError:
        return None


def _is_readme_toc(doc: Document) -> bool:
    source = str(doc.metadata.get("source", "")).replace("\\", "/").lower()
    content = doc.page_content.lower()
    return source.endswith("/readme.md") and (
        "## sections" in content
        or "table of contents" in content
        or "\ntoc\n" in content
    )


def _is_heading_only(doc: Document) -> bool:
    content = doc.page_content.strip()
    if len(content) > HEADING_ONLY_MAX_CHARS or not content.startswith("#"):
        return False
    text_without_marks = content.lstrip("#").strip()
    return len(text_without_marks.split()) <= 8


class HybridRetriever:
    def __init__(
        self,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        candidate_top_k: int | None = None,
        kb_config: KnowledgeBaseConfig | None = None,
    ):
        self.kb_config = kb_config or get_default_knowledge_base()
        # self.dense_retriever = get_dense_retriever()
        self.dense_retriever = get_qdrant_retriever(self.kb_config)
        self.bm25_store = build_bm25_store(self.kb_config)
        _, self.parent_map = build_parent_child_chunks(self.kb_config)
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.candidate_top_k = candidate_top_k or FINAL_TOP_K

    def retrieve_children(self, query: str) -> list[Document]:
        dense_docs = self.dense_retriever.invoke(query)
        bm25_docs = self.bm25_store.search(query)

        # RRF lets dense semantic matches and BM25 keyword matches vote together.
        # This avoids the old dense_docs + bm25_docs truncation problem where
        # good BM25 hits could be pushed out by mediocre dense results.
        rrf_k = 60
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            child_id = doc.metadata.get("child_id")
            if not child_id:
                continue
            scores[child_id] = scores.get(child_id, 0.0) + self.dense_weight / (rrf_k + rank)
            doc_map.setdefault(child_id, doc)

        for rank, doc in enumerate(bm25_docs, start=1):
            child_id = doc.metadata.get("child_id")
            if not child_id:
                continue
            scores[child_id] = scores.get(child_id, 0.0) + self.bm25_weight / (rrf_k + rank)
            doc_map.setdefault(child_id, doc)

        ranked_child_ids = sorted(scores, key=scores.get, reverse=True)
        return [doc_map[child_id] for child_id in ranked_child_ids[: self.candidate_top_k]]

    def retrieve_parents(self, query: str) -> list[Document]:
        child_docs = self.retrieve_children(query)

        parent_docs = []
        seen = set()

        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if not parent_id or parent_id in seen:
                continue

            parent_doc = self.parent_map.get(parent_id)
            if parent_doc:
                expanded_doc = self._expand_heading_parent(parent_doc)
                parent_docs.append(expanded_doc)
                for expanded_parent_id in expanded_doc.metadata.get("expanded_parent_ids", [parent_id]):
                    if expanded_parent_id:
                        seen.add(expanded_parent_id)

        parent_docs = self._demote_readme_toc(parent_docs)
        parent_docs = self._promote_primary_source_neighbors(parent_docs)
        return self._promote_primary_expanded_source(parent_docs)

    def _expand_heading_parent(self, parent_doc: Document) -> Document:
        """Merge a heading-only parent with following siblings from the same source.

        Parent/child splitting can create tiny section-title chunks. Those
        chunks are useful retrieval anchors but poor answer context on their
        own, so expand them with adjacent content.
        """
        if not _is_heading_only(parent_doc):
            return parent_doc

        parent_id = str(parent_doc.metadata.get("parent_id", ""))
        current_index = _parent_index(parent_id)
        if current_index is None:
            return parent_doc

        source = parent_doc.metadata.get("source")
        parts = [parent_doc.page_content.strip()]
        expanded_parent_ids = [parent_id]

        for offset in range(1, HEADING_EXPANSION_MAX_PARENTS + 1):
            next_parent = self.parent_map.get(f"parent-{current_index + offset}")
            if not next_parent or next_parent.metadata.get("source") != source:
                break
            parts.append(next_parent.page_content.strip())
            expanded_parent_ids.append(str(next_parent.metadata.get("parent_id", "")))
            if sum(len(part) for part in parts) >= HEADING_EXPANSION_TARGET_CHARS:
                break

        if len(parts) == 1:
            return parent_doc

        metadata = dict(parent_doc.metadata)
        metadata["expanded_from_heading"] = True
        metadata["expanded_parent_ids"] = expanded_parent_ids
        return Document(page_content="\n\n".join(parts), metadata=metadata)

    def _demote_readme_toc(self, docs: list[Document]) -> list[Document]:
        if not any(not _is_readme_toc(doc) for doc in docs):
            return docs
        factual_docs = [doc for doc in docs if not _is_readme_toc(doc)]
        readme_docs = [doc for doc in docs if _is_readme_toc(doc)]
        return factual_docs + readme_docs

    def _promote_primary_expanded_source(self, docs: list[Document]) -> list[Document]:
        if not docs or not docs[0].metadata.get("expanded_from_heading"):
            return docs

        primary_source = docs[0].metadata.get("source")
        same_source = [doc for doc in docs if doc.metadata.get("source") == primary_source]
        other_sources = [doc for doc in docs if doc.metadata.get("source") != primary_source]
        return same_source + other_sources

    def _promote_primary_source_neighbors(self, docs: list[Document]) -> list[Document]:
        if not docs:
            return docs

        first_doc = docs[0]
        first_parent_id = str(first_doc.metadata.get("parent_id", ""))
        first_index = _parent_index(first_parent_id)
        primary_source = first_doc.metadata.get("source")
        if first_index is None or not primary_source:
            return docs

        promoted: list[Document] = []
        seen_parent_ids = set()

        def add(doc: Document) -> None:
            parent_id = str(doc.metadata.get("parent_id", ""))
            if parent_id and parent_id in seen_parent_ids:
                return
            promoted.append(doc)
            if parent_id:
                seen_parent_ids.add(parent_id)

        add(first_doc)

        for offset in range(1, PRIMARY_SOURCE_NEIGHBOR_PARENTS + 1):
            neighbor = self.parent_map.get(f"parent-{first_index + offset}")
            if not neighbor or neighbor.metadata.get("source") != primary_source:
                break
            add(neighbor)

        for doc in docs[1:]:
            add(doc)

        return promoted


if __name__ == "__main__":
    retriever = HybridRetriever()
    docs = retriever.retrieve_parents("What are the main steps in this workflow?")

    for doc in docs:
        print("=" * 40)
        print(doc.metadata)
        print(doc.page_content[:800])
