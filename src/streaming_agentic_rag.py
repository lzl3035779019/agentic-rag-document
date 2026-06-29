from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from langchain_core.documents import Document

from src.config import (
    ANSWER_CONTEXT_TOP_K,
    ANSWER_HIGH_RISK_CONTEXT_TOP_K,
    ANSWER_MEDIUM_RISK_CONTEXT_TOP_K,
)
from src.evidence import answer_from_evidence_stream, extract_evidence, fast_answer_stream
from src.agents.aggregator import aggregate_answers_stream
from src.agents.decomposer import decompose_question
from src.graph import (
    RAGState,
    _append_sources,
    _evidence_context,
    _source_snippets,
    build_graph,
    clarify_node,
    retrieve_node,
    run_single_question_graph,
    rewrite_node,
    route_after_retrieve,
)
from src.graders import grade_answer
from src.knowledge_base import KnowledgeBaseConfig, get_knowledge_base
from src.router import route_question


@dataclass
class PreparedFastAnswerStream:
    question: str
    kb_config: KnowledgeBaseConfig
    route_result: dict
    state: RAGState
    answer_docs: list[Document]
    started: float
    route_ms: float

    def stream_answer(self):
        yield from fast_answer_stream(self.question, self.answer_docs)

    def finish(self, answer_body: str) -> dict:
        timings = dict(self.state.get("node_timings_ms", {}))
        total_ms = round((time.perf_counter() - self.started) * 1000, 2)
        answer = _append_sources(answer_body.strip(), self.answer_docs)
        raw_result = dict(self.state)
        raw_result.update(
            {
                "answer": answer,
                "source_snippets": _source_snippets(self.answer_docs),
                "needs_answer_grading": False,
                "grounding_checked": False,
                "grounded": bool(answer_body.strip()),
                "grade_reason": "Skipped strict evidence extraction and answer grading by risk-aware streaming fast path.",
                "final_status": "answered_fast_streamed" if answer_body.strip() else "insufficient_context",
                "node_timings_ms": timings,
            }
        )

        return {
            "question": self.question,
            "route": self.route_result["route"],
            "router": self.route_result["router"],
            "route_reason": self.route_result["reason"],
            "answer": answer,
            "knowledge_base": {
                "kb_id": self.kb_config.kb_id,
                "name": self.kb_config.name,
                "collection_name": self.kb_config.collection_name,
                "embedding_model": self.kb_config.embedding_model,
            },
            "timings_ms": {
                "route_ms": self.route_ms,
                "total_ms": total_ms,
                "rag_ms": round(total_ms - self.route_ms, 2),
            },
            "raw_result": raw_result,
        }


@dataclass
class PreparedEvidenceAnswerStream:
    question: str
    kb_config: KnowledgeBaseConfig
    route_result: dict
    state: RAGState
    evidence: dict[str, list[str]]
    answer_docs: list[Document]
    started: float
    route_ms: float
    grade_result: dict | None = None

    def stream_answer(self):
        yield from answer_from_evidence_stream(self.question, self.evidence)

    def finish(self, answer_body: str) -> dict:
        timings = dict(self.state.get("node_timings_ms", {}))
        grade_result = self.grade_result
        if grade_result is None:
            grade_started = time.perf_counter()
            grade_result = grade_answer(
                question=self.question,
                context=_evidence_context(self.evidence),
                answer=answer_body.strip(),
            )
            timings["grade_answer"] = round((time.perf_counter() - grade_started) * 1000, 2)

        grounded = bool(grade_result.get("grounded"))
        total_ms = round((time.perf_counter() - self.started) * 1000, 2)
        answer = _append_sources(answer_body.strip(), self.answer_docs)
        raw_result = dict(self.state)
        raw_result.update(
            {
                "evidence": self.evidence,
                "answer": answer,
                "source_snippets": _source_snippets(self.answer_docs),
                "needs_answer_grading": True,
                "grounding_checked": True,
                "grounded": grounded,
                "grade_reason": str(grade_result.get("reason", "")),
                "final_status": "answered_with_evidence_streamed" if grounded else "streamed_answer_failed_grounding",
                "node_timings_ms": timings,
            }
        )

        return {
            "question": self.question,
            "route": self.route_result["route"],
            "router": self.route_result["router"],
            "route_reason": self.route_result["reason"],
            "answer": answer,
            "knowledge_base": {
                "kb_id": self.kb_config.kb_id,
                "name": self.kb_config.name,
                "collection_name": self.kb_config.collection_name,
                "embedding_model": self.kb_config.embedding_model,
            },
            "timings_ms": {
                "route_ms": self.route_ms,
                "total_ms": total_ms,
                "rag_ms": round(total_ms - self.route_ms, 2),
            },
            "raw_result": raw_result,
        }


@dataclass
class PreparedComplexAnswerStream:
    question: str
    kb_config: KnowledgeBaseConfig
    route_result: dict
    started: float
    route_ms: float

    def _run_sub_question(self, index: int, sub_question: str, graph_app) -> dict:
        started = time.perf_counter()
        result = run_single_question_graph(sub_question, graph_app, kb_config=self.kb_config)
        documents = result.get("documents", [])
        return {
            "index": index,
            "sub_question": sub_question,
            "answer": result.get("answer", ""),
            "documents": documents,
            "sources": result.get("retrieved_sources", []),
            "source_count": len(documents),
            "risk_level": result.get("risk_level"),
            "final_status": result.get("final_status"),
            "grounded": result.get("grounded"),
            "grounding_checked": result.get("grounding_checked"),
            "node_timings_ms": result.get("node_timings_ms", {}),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }

    def _source_docs(self, sub_results: list[dict]) -> list[Document]:
        docs: list[Document] = []
        seen: set[tuple[str, str]] = set()
        for item in sub_results:
            for doc in item.get("documents", []) or []:
                source = str(doc.metadata.get("source", ""))
                parent_id = str(doc.metadata.get("parent_id", ""))
                key = (source, parent_id)
                if not source or not parent_id or key in seen:
                    continue
                seen.add(key)
                docs.append(doc)
        return docs

    def _finish(self, sub_questions: list[str], sub_results: list[dict], final_answer: str, timings: dict[str, float]) -> dict:
        total_ms = round((time.perf_counter() - self.started) * 1000, 2)
        timings["total_ms"] = total_ms
        source_docs = self._source_docs(sub_results)
        answer = _append_sources(final_answer, source_docs)
        raw_result = {
            "question": self.question,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
            "answer": answer,
            "source_snippets": _source_snippets(source_docs),
            "timings_ms": timings,
        }
        return {
            "question": self.question,
            "route": self.route_result["route"],
            "router": self.route_result["router"],
            "route_reason": self.route_result["reason"],
            "answer": answer,
            "knowledge_base": {
                "kb_id": self.kb_config.kb_id,
                "name": self.kb_config.name,
                "collection_name": self.kb_config.collection_name,
                "embedding_model": self.kb_config.embedding_model,
            },
            "timings_ms": {
                "route_ms": self.route_ms,
                "total_ms": total_ms,
                "rag_ms": round(total_ms - self.route_ms, 2),
            },
            "raw_result": raw_result,
        }

    def stream_events(self):
        timings: dict[str, float] = {}

        started = time.perf_counter()
        sub_questions = decompose_question(self.question)
        timings["decompose_ms"] = round((time.perf_counter() - started) * 1000, 2)
        yield {"type": "sub_questions", "sub_questions": sub_questions}

        graph_app = build_graph()
        sub_results: list[dict] = []
        started = time.perf_counter()
        max_workers = max(1, min(len(sub_questions), 4))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(self._run_sub_question, index, sub_question, graph_app)
                for index, sub_question in enumerate(sub_questions, start=1)
            ]
            for future in as_completed(futures):
                sub_result = future.result()
                sub_results.append(sub_result)
                yield {"type": "sub_result", "result": sub_result}
        sub_results.sort(key=lambda item: item["index"])
        timings["parallel_sub_questions_ms"] = round((time.perf_counter() - started) * 1000, 2)

        yield {"type": "aggregate_start"}
        started = time.perf_counter()
        final_chunks: list[str] = []
        for chunk in aggregate_answers_stream(self.question, sub_results):
            final_chunks.append(chunk)
            yield {"type": "aggregate_chunk", "text": chunk}
        timings["aggregate_ms"] = round((time.perf_counter() - started) * 1000, 2)

        final_answer = "".join(final_chunks).strip()
        yield {
            "type": "final",
            "result": self._finish(sub_questions, sub_results, final_answer, timings),
        }


def prepare_fast_answer_stream(question: str, kb_id: str | None = None) -> PreparedFastAnswerStream | None:
    started = time.perf_counter()
    kb_config = get_knowledge_base(kb_id)

    route_started = time.perf_counter()
    route_result = route_question(question)
    route_ms = round((time.perf_counter() - route_started) * 1000, 2)
    if route_result["route"] != "simple":
        return None

    state: RAGState = {"question": question, "kb_config": kb_config}
    state.update(clarify_node(state))
    if state.get("need_clarification"):
        return None

    while True:
        state.update(retrieve_node(state))
        next_node = route_after_retrieve(state)

        if next_node == "rewrite":
            state.update(rewrite_node(state))
            continue
        if next_node != "fast_answer":
            return None

        risk_level = state.get("risk_level", "low")
        context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K if risk_level == "medium" else ANSWER_CONTEXT_TOP_K
        answer_docs = state.get("documents", [])[:context_top_k]
        return PreparedFastAnswerStream(
            question=question,
            kb_config=kb_config,
            route_result=route_result,
            state=state,
            answer_docs=answer_docs,
            started=started,
            route_ms=route_ms,
        )


def prepare_evidence_answer_stream(question: str, kb_id: str | None = None) -> PreparedEvidenceAnswerStream | None:
    started = time.perf_counter()
    kb_config = get_knowledge_base(kb_id)

    route_started = time.perf_counter()
    route_result = route_question(question)
    route_ms = round((time.perf_counter() - route_started) * 1000, 2)
    if route_result["route"] != "simple":
        return None

    state: RAGState = {"question": question, "kb_config": kb_config}
    state.update(clarify_node(state))
    if state.get("need_clarification"):
        return None

    while True:
        state.update(retrieve_node(state))
        next_node = route_after_retrieve(state)

        if next_node == "rewrite":
            state.update(rewrite_node(state))
            continue
        if next_node != "answer_with_evidence":
            return None

        risk_level = state.get("risk_level", "high")
        if risk_level == "high":
            context_top_k = ANSWER_HIGH_RISK_CONTEXT_TOP_K
        elif risk_level == "medium":
            context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K
        else:
            context_top_k = ANSWER_CONTEXT_TOP_K
        answer_docs = state.get("documents", [])[:context_top_k]

        evidence_started = time.perf_counter()
        evidence = extract_evidence(question, answer_docs)
        timings = dict(state.get("node_timings_ms", {}))
        timings["extract_evidence"] = round((time.perf_counter() - evidence_started) * 1000, 2)
        state["node_timings_ms"] = timings

        return PreparedEvidenceAnswerStream(
            question=question,
            kb_config=kb_config,
            route_result=route_result,
            state=state,
            evidence=evidence,
            answer_docs=answer_docs,
            started=started,
            route_ms=route_ms,
        )


def prepare_simple_answer_stream(
    question: str,
    kb_id: str | None = None,
) -> PreparedFastAnswerStream | PreparedEvidenceAnswerStream | None:
    started = time.perf_counter()
    kb_config = get_knowledge_base(kb_id)

    route_started = time.perf_counter()
    route_result = route_question(question)
    route_ms = round((time.perf_counter() - route_started) * 1000, 2)
    if route_result["route"] != "simple":
        return None

    state: RAGState = {"question": question, "kb_config": kb_config}
    state.update(clarify_node(state))
    if state.get("need_clarification"):
        return None

    while True:
        state.update(retrieve_node(state))
        next_node = route_after_retrieve(state)

        if next_node == "rewrite":
            state.update(rewrite_node(state))
            continue
        if next_node == "fast_answer":
            risk_level = state.get("risk_level", "low")
            context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K if risk_level == "medium" else ANSWER_CONTEXT_TOP_K
            answer_docs = state.get("documents", [])[:context_top_k]
            return PreparedFastAnswerStream(
                question=question,
                kb_config=kb_config,
                route_result=route_result,
                state=state,
                answer_docs=answer_docs,
                started=started,
                route_ms=route_ms,
            )
        if next_node == "answer_with_evidence":
            risk_level = state.get("risk_level", "high")
            if risk_level == "high":
                context_top_k = ANSWER_HIGH_RISK_CONTEXT_TOP_K
            elif risk_level == "medium":
                context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K
            else:
                context_top_k = ANSWER_CONTEXT_TOP_K
            answer_docs = state.get("documents", [])[:context_top_k]

            evidence_started = time.perf_counter()
            evidence = extract_evidence(question, answer_docs)
            timings = dict(state.get("node_timings_ms", {}))
            timings["extract_evidence"] = round((time.perf_counter() - evidence_started) * 1000, 2)
            state["node_timings_ms"] = timings

            return PreparedEvidenceAnswerStream(
                question=question,
                kb_config=kb_config,
                route_result=route_result,
                state=state,
                evidence=evidence,
                answer_docs=answer_docs,
                started=started,
                route_ms=route_ms,
            )
        return None


def prepare_complex_answer_stream(question: str, kb_id: str | None = None) -> PreparedComplexAnswerStream | None:
    started = time.perf_counter()
    kb_config = get_knowledge_base(kb_id)

    route_started = time.perf_counter()
    route_result = route_question(question)
    route_ms = round((time.perf_counter() - route_started) * 1000, 2)
    if route_result["route"] != "complex":
        return None

    return PreparedComplexAnswerStream(
        question=question,
        kb_config=kb_config,
        route_result=route_result,
        started=started,
        route_ms=route_ms,
    )
