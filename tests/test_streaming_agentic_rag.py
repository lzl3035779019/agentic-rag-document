from langchain_core.documents import Document

from src.config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARSED_DOCS_DIR
from src.knowledge_base import KnowledgeBaseConfig
from src.streaming_agentic_rag import (
    PreparedComplexAnswerStream,
    PreparedEvidenceAnswerStream,
    PreparedFastAnswerStream,
)


def test_prepared_fast_answer_stream_finish_builds_result():
    prepared = PreparedFastAnswerStream(
        question="What is this?",
        kb_config=KnowledgeBaseConfig(
            kb_id="kb",
            name="KB",
            collection_name="collection",
            embedding_model="model",
            language_strategy="en",
            parent_chunk_size=PARENT_CHUNK_SIZE,
            parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
            child_chunk_size=CHILD_CHUNK_SIZE,
            child_chunk_overlap=CHILD_CHUNK_OVERLAP,
            raw_dir=PARSED_DOCS_DIR,
            parsed_dir=PARSED_DOCS_DIR,
            file_names=[],
            created_at="test",
        ),
        route_result={"route": "simple", "router": "rule", "reason": "ok"},
        state={
            "question": "What is this?",
            "documents": [Document(page_content="content", metadata={"source": "doc.md", "parent_id": "parent-1"})],
            "risk_level": "low",
            "node_timings_ms": {"retrieve": 1.0},
        },
        answer_docs=[Document(page_content="content", metadata={"source": "doc.md", "parent_id": "parent-1"})],
        started=0.0,
        route_ms=1.0,
    )

    result = prepared.finish("Answer body")

    assert result["route"] == "simple"
    assert result["answer"].startswith("Answer body")
    assert "Sources:" in result["answer"]
    assert result["raw_result"]["final_status"] == "answered_fast_streamed"
    assert result["raw_result"]["source_snippets"] == [{"source": "doc.md#parent-1", "snippet": "content"}]


def test_prepared_evidence_answer_stream_finish_builds_grounded_result():
    prepared = PreparedEvidenceAnswerStream(
        question="How should credentials be handled?",
        kb_config=KnowledgeBaseConfig(
            kb_id="kb",
            name="KB",
            collection_name="collection",
            embedding_model="model",
            language_strategy="en",
            parent_chunk_size=PARENT_CHUNK_SIZE,
            parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
            child_chunk_size=CHILD_CHUNK_SIZE,
            child_chunk_overlap=CHILD_CHUNK_OVERLAP,
            raw_dir=PARSED_DOCS_DIR,
            parsed_dir=PARSED_DOCS_DIR,
            file_names=[],
            created_at="test",
        ),
        route_result={"route": "simple", "router": "rule", "reason": "ok"},
        state={
            "question": "How should credentials be handled?",
            "documents": [Document(page_content="Use approved storage.", metadata={"source": "security.md", "parent_id": "parent-2"})],
            "risk_level": "high",
            "node_timings_ms": {"retrieve": 1.0},
        },
        evidence={"facts": ["Use approved storage."], "missing_information": []},
        answer_docs=[Document(page_content="Use approved storage.", metadata={"source": "security.md", "parent_id": "parent-2"})],
        started=0.0,
        route_ms=1.0,
        grade_result={"grounded": True, "reason": "Supported."},
    )

    result = prepared.finish("Use approved storage.")

    assert result["route"] == "simple"
    assert result["answer"].startswith("Use approved storage.")
    assert "Sources:" in result["answer"]
    assert result["raw_result"]["final_status"] == "answered_with_evidence_streamed"
    assert result["raw_result"]["grounding_checked"] is True
    assert result["raw_result"]["grounded"] is True


def test_prepared_complex_answer_stream_emits_sub_results_and_final(monkeypatch):
    kb_config = KnowledgeBaseConfig(
        kb_id="kb",
        name="KB",
        collection_name="collection",
        embedding_model="model",
        language_strategy="en",
        parent_chunk_size=PARENT_CHUNK_SIZE,
        parent_chunk_overlap=PARENT_CHUNK_OVERLAP,
        child_chunk_size=CHILD_CHUNK_SIZE,
        child_chunk_overlap=CHILD_CHUNK_OVERLAP,
        raw_dir=PARSED_DOCS_DIR,
        parsed_dir=PARSED_DOCS_DIR,
        file_names=[],
        created_at="test",
    )

    monkeypatch.setattr("src.streaming_agentic_rag.decompose_question", lambda question: ["Q1", "Q2"])
    monkeypatch.setattr("src.streaming_agentic_rag.build_graph", lambda: object())

    def fake_run_single_question_graph(question, graph_app, kb_config=None):
        return {
            "answer": f"Answer {question}",
            "documents": [
                Document(
                    page_content=f"Evidence for {question}",
                    metadata={"source": f"{question}.md", "parent_id": "parent-1"},
                )
            ],
            "retrieved_sources": [f"{question}.md"],
            "risk_level": "low",
            "final_status": "answered_fast",
            "grounded": True,
            "grounding_checked": False,
            "node_timings_ms": {},
        }

    monkeypatch.setattr("src.streaming_agentic_rag.run_single_question_graph", fake_run_single_question_graph)
    monkeypatch.setattr("src.streaming_agentic_rag.aggregate_answers_stream", lambda question, sub_results: iter(["Final"]))

    prepared = PreparedComplexAnswerStream(
        question="Compare A and B.",
        kb_config=kb_config,
        route_result={"route": "complex", "router": "rule", "reason": "compare"},
        started=0.0,
        route_ms=1.0,
    )

    events = list(prepared.stream_events())

    assert [event["type"] for event in events] == [
        "sub_questions",
        "sub_result",
        "sub_result",
        "aggregate_start",
        "aggregate_chunk",
        "final",
    ]
    assert events[-1]["result"]["route"] == "complex"
    assert events[-1]["result"]["answer"].startswith("Final")
    assert "Sources:" in events[-1]["result"]["answer"]
    assert len(events[-1]["result"]["raw_result"]["sub_results"]) == 2
    assert events[-1]["result"]["raw_result"]["source_snippets"]
