from src.config import (
    ANSWER_CONTEXT_TOP_K,
    ANSWER_HIGH_RISK_CONTEXT_TOP_K,
    ANSWER_MEDIUM_RISK_CONTEXT_TOP_K,
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from src.parent_child_splitter import build_parent_child_chunks
from src.qdrant_store import COLLECTION_NAME
from src.hybrid_retriever import HybridRetriever


def test_parent_chunks_do_not_leave_heading_only_chunks():
    _, parent_map = build_parent_child_chunks()

    heading_only_chunks = [
        (parent_id, doc.metadata.get("source"), doc.page_content.strip())
        for parent_id, doc in parent_map.items()
        if len(doc.page_content.strip()) <= 120
        and doc.page_content.strip().startswith("#")
        and len(doc.page_content.strip().lstrip("#").strip().split()) <= 8
    ]

    assert heading_only_chunks == []


def test_chunk_and_answer_top_k_policy_matches_visualized_rag_defaults():
    assert PARENT_CHUNK_SIZE == 3000
    assert PARENT_CHUNK_OVERLAP == 300
    assert CHILD_CHUNK_SIZE == 500
    assert CHILD_CHUNK_OVERLAP == 100
    assert ANSWER_CONTEXT_TOP_K == 5
    assert ANSWER_MEDIUM_RISK_CONTEXT_TOP_K == 6
    assert ANSWER_HIGH_RISK_CONTEXT_TOP_K == 8


def test_visualized_project_uses_isolated_qdrant_collection():
    assert COLLECTION_NAME == "basecamp_handbook_visualized"


def test_benefits_query_keeps_benefits_parents_in_answer_window():
    docs = HybridRetriever().retrieve_parents(
        "What benefits and perks does Basecamp provide to employees?"
    )

    top_six_sources = [
        str(doc.metadata.get("source", "")).replace("\\", "/").lower()
        for doc in docs[:6]
    ]

    assert all("benefits-and-perks.md" in source for source in top_six_sources)
