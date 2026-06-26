import time
from typing import TypedDict
import re

from langchain_core.documents import Document
from langgraph.graph import END, StateGraph

from src.clarification import check_clarification
from src.config import (
    ANSWER_HIGH_RISK_CONTEXT_TOP_K,
    ANSWER_MEDIUM_RISK_CONTEXT_TOP_K,
    ANSWER_CONTEXT_TOP_K,
    MAX_REGENERATE_RETRY,
    MAX_REWRITE_RETRY,
    MIN_RELEVANT_DOCS,
    REGENERATE_CONTEXT_TOP_K,
)
from src.evidence import answer_from_evidence, answer_with_evidence, extract_evidence, fast_answer
from src.graders import grade_answer, grade_documents
from src.hybrid_retriever import HybridRetriever
from src.knowledge_base import KnowledgeBaseConfig
from src.query_rewritter import rewrite_query, rewrite_query_with_feedback


HARD_HIGH_RISK_TERMS = {
    "legal",
    "compliance",
    "lawsuit",
    "subpoena",
    "liability",
    "violation",
    "security",
    "security incident",
    "secret",
    "secrets",
    "credential",
    "credentials",
    "password",
    "token",
    "api key",
    "vpn",
    "privacy",
    "personal data",
    "sensitive data",
    "permission",
    "access control",
    "delete",
    "remove",
    "destroy",
    "overwrite",
    "irreversible",
    "payment",
    "billing",
    "invoice",
    "financial",
    "medical",
    "health",
    "diagnosis",
    "drug",
    "investment",
    "trade",
    "trading",
    "法律",
    "合规",
    "诉讼",
    "传票",
    "责任",
    "违规",
    "违法",
    "安全",
    "安全事故",
    "凭证",
    "密钥",
    "密码",
    "令牌",
    "个人信息",
    "敏感信息",
    "敏感数据",
    "隐私",
    "权限",
    "访问控制",
    "删除",
    "移除",
    "销毁",
    "覆盖",
    "不可逆",
    "支付",
    "付款",
    "账单",
    "发票",
    "财务",
    "金额",
    "医疗",
    "健康",
    "诊断",
    "用药",
    "投资",
    "交易",
}

SOFT_POLICY_TERMS = {
    "policy",
    "rule",
    "requirement",
    "requirements",
    "condition",
    "conditions",
    "criteria",
    "eligibility",
    "allowed",
    "prohibited",
    "exception",
    "exceptions",
    "limit",
    "limits",
    "boundary",
    "boundaries",
    "constraint",
    "constraints",
    "process",
    "procedure",
    "workflow",
    "steps",
    "should",
    "must",
    "need to",
    "规则",
    "要求",
    "条件",
    "标准",
    "资格",
    "是否允许",
    "能否",
    "可不可以",
    "禁止",
    "例外",
    "限制",
    "边界",
    "约束",
    "流程",
    "步骤",
    "过程",
    "应该",
    "必须",
    "需要",
}

def _prefers_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _is_high_risk_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in HARD_HIGH_RISK_TERMS)


def _is_medium_risk_question(question: str) -> bool:
    lowered = question.lower()
    return any(term in lowered for term in SOFT_POLICY_TERMS)


def _risk_level(question: str) -> str:
    if _is_high_risk_question(question):
        return "high"
    if _is_medium_risk_question(question):
        return "medium"
    return "low"


def _with_timing(state: "RAGState", update: "RAGState", node_name: str, started: float) -> "RAGState":
    timings = dict(state.get("node_timings_ms", {}))
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    timings[node_name] = round(timings.get(node_name, 0.0) + elapsed_ms, 2)
    update["node_timings_ms"] = timings
    return update


class RAGState(TypedDict, total=False):
    question: str
    kb_config: KnowledgeBaseConfig
    rewritten_question: str
    need_clarification: bool
    clarification_message: str
    documents: list[Document]
    retrieved_sources: list[str]
    source_snippets: list[dict[str, str]]
    evidence: dict[str, list[str]]
    answer: str
    rewrite_count: int
    regenerate_count: int
    need_rewrite: bool
    need_regenerate: bool
    needs_document_grading: bool
    needs_answer_grading: bool
    grounding_checked: bool
    grounded: bool
    grade_reason: str
    final_status: str
    risk_level: str
    node_timings_ms: dict[str, float]


def clarify_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    result = check_clarification(state["question"])
    if result["need_clarification"]:
        return _with_timing(state, {
            "need_clarification": True,
            "clarification_message": result["raw"],
            "grounding_checked": False,
            "grounded": False,
            "final_status": "needs_clarification",
            "risk_level": _risk_level(state["question"]),
        }, "clarify", started)

    return _with_timing(state, {
        "need_clarification": False,
        "rewritten_question": state["question"],
        "rewrite_count": 0,
        "regenerate_count": 0,
        "grounding_checked": False,
        "risk_level": _risk_level(state["question"]),
    }, "clarify", started)


def rewrite_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    rewrite_count = state.get("rewrite_count", 0)
    if rewrite_count == 0:
        rewritten = rewrite_query(state["question"])
    else:
        rewritten = rewrite_query_with_feedback(
            question=state["question"],
            previous_query=state.get("rewritten_question", state["question"]),
            grade_reason=state.get("grade_reason", ""),
            retrieved_sources=state.get("retrieved_sources", []),
        )

    return _with_timing(state, {
        "rewritten_question": rewritten,
        "rewrite_count": rewrite_count + 1,
        "need_rewrite": False,
    }, "rewrite", started)


def retrieve_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    retriever = HybridRetriever(kb_config=state.get("kb_config"))
    docs = retriever.retrieve_parents(state["rewritten_question"])
    sources = [str(doc.metadata.get("source", "")) for doc in docs]
    return _with_timing(state, {
        "documents": docs,
        "retrieved_sources": sources,
        "need_rewrite": len(docs) < MIN_RELEVANT_DOCS,
        "needs_document_grading": False,
        "grade_reason": "No documents retrieved." if len(docs) < MIN_RELEVANT_DOCS else "Retrieved documents are available.",
    }, "retrieve", started)


def grade_documents_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    docs = state.get("documents", [])
    relevant_docs = grade_documents(state["rewritten_question"], docs)
    need_rewrite = len(relevant_docs) < MIN_RELEVANT_DOCS

    if need_rewrite:
        grade_reason = "No retrieved document was judged relevant enough to answer the rewritten query."
    else:
        reasons = [
            str(doc.metadata.get("relevance_reason", "")).strip()
            for doc in relevant_docs
            if doc.metadata.get("relevance_reason")
        ]
        grade_reason = "Retrieved documents passed relevance grading."
        if reasons:
            grade_reason += " " + " | ".join(reasons[:3])

    return _with_timing(state, {
        "documents": relevant_docs,
        "need_rewrite": need_rewrite,
        "grade_reason": grade_reason,
    }, "grade_documents", started)


def _source_docs_for_display(docs: list[Document]) -> list[Document]:
    if docs and docs[0].metadata.get("expanded_from_heading"):
        primary_source = docs[0].metadata.get("source")
        return [doc for doc in docs if doc.metadata.get("source") == primary_source]
    return docs


def _source_reference(doc: Document) -> str | None:
    source = doc.metadata.get("source")
    parent_id = doc.metadata.get("parent_id")
    if not source or not parent_id:
        return None
    if doc.metadata.get("expanded_parent_ids"):
        parent_id = ",".join(doc.metadata["expanded_parent_ids"])
    return f"{source}#{parent_id}"


def _source_snippet(doc: Document, max_chars: int = 700) -> str:
    text = re.sub(r"\s+", " ", doc.page_content).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _source_snippets(docs: list[Document]) -> list[dict[str, str]]:
    snippets = []
    seen = set()
    for doc in _source_docs_for_display(docs):
        reference = _source_reference(doc)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        snippets.append({"source": reference, "snippet": _source_snippet(doc)})
    return snippets


def _append_sources(answer: str, docs: list[Document]) -> str:
    sources = []
    seen = set()
    for doc in _source_docs_for_display(docs):
        reference = _source_reference(doc)
        if not reference or reference in seen:
            continue
        seen.add(reference)
        sources.append(f"- {reference}")
    if sources:
        return answer + "\n\nSources:\n" + "\n".join(sources)
    return answer


def _answer_body(answer: str) -> str:
    return answer.split("\n\nSources:", 1)[0].strip()


def _evidence_context(evidence: dict[str, list[str]]) -> str:
    facts = evidence.get("facts", []) or []
    missing = evidence.get("missing_information", []) or []
    lines = ["Evidence facts:"]
    lines.extend(f"- {fact}" for fact in facts)
    if missing:
        lines.append("\nMissing information:")
        lines.extend(f"- {item}" for item in missing)
    return "\n".join(lines)


def _should_grade_answer(question: str, evidence: dict[str, list[str]], answer: str) -> bool:
    risk_level = _risk_level(question)
    if risk_level == "high":
        return True
    if not answer.strip():
        return True
    if len(evidence.get("facts", []) or []) < 2:
        return True
    return False


def answer_with_evidence_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    risk_level = state.get("risk_level") or _risk_level(state["question"])
    if risk_level == "high":
        context_top_k = ANSWER_HIGH_RISK_CONTEXT_TOP_K
    elif risk_level == "medium":
        context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K
    else:
        context_top_k = ANSWER_CONTEXT_TOP_K
    answer_docs = state.get("documents", [])[:context_top_k]
    result = answer_with_evidence(state["question"], answer_docs)
    evidence = {
        "facts": result.get("facts", []),
        "missing_information": result.get("missing_information", []),
    }
    answer = result.get("answer", "")
    needs_answer_grading = _should_grade_answer(state["question"], evidence, answer)

    return _with_timing(state, {
        "evidence": evidence,
        "answer": _append_sources(answer, answer_docs),
        "source_snippets": _source_snippets(answer_docs),
        "needs_answer_grading": needs_answer_grading,
        "grounding_checked": False,
        "grounded": not needs_answer_grading,
        "grade_reason": "Skipped answer grading by fast-path policy." if not needs_answer_grading else state.get("grade_reason", ""),
        "final_status": "answered_fast" if not needs_answer_grading else "answered",
        "risk_level": risk_level,
    }, "answer_with_evidence", started)


def fast_answer_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    risk_level = state.get("risk_level") or _risk_level(state["question"])
    context_top_k = ANSWER_MEDIUM_RISK_CONTEXT_TOP_K if risk_level == "medium" else ANSWER_CONTEXT_TOP_K
    answer_docs = state.get("documents", [])[:context_top_k]
    answer = fast_answer(state["question"], answer_docs)
    final_status = "answered_fast" if answer.strip() else "insufficient_context"

    return _with_timing(state, {
        "answer": _append_sources(answer, answer_docs),
        "source_snippets": _source_snippets(answer_docs),
        "needs_answer_grading": False,
        "grounding_checked": False,
        "grounded": bool(answer.strip()),
        "grade_reason": "Skipped strict evidence extraction and answer grading by risk-aware fast path.",
        "final_status": final_status,
        "risk_level": risk_level,
    }, "fast_answer", started)


def regenerate_answer_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    regenerate_count = state.get("regenerate_count", 0)
    answer_docs = state.get("documents", [])[:REGENERATE_CONTEXT_TOP_K]
    evidence = extract_evidence(state["question"], answer_docs)
    strict_question = (
        f"{state['question']}\n\n"
        "The previous answer failed the grounding check. "
        f"Failure reason: {state.get('grade_reason', '')}. "
        "Regenerate a stricter answer using only the extracted evidence."
    )
    answer = answer_from_evidence(strict_question, evidence)

    return _with_timing(state, {
        "evidence": evidence,
        "answer": _append_sources(answer, answer_docs),
        "source_snippets": _source_snippets(answer_docs),
        "regenerate_count": regenerate_count + 1,
        "need_regenerate": False,
        "needs_answer_grading": True,
        "grounding_checked": False,
        "final_status": "regenerated",
    }, "regenerate_answer", started)


def grade_answer_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    result = grade_answer(
        question=state["question"],
        context=_evidence_context(state.get("evidence", {})),
        answer=_answer_body(state["answer"]),
    )

    return _with_timing(state, {
        "grounded": result["grounded"],
        "grounding_checked": True,
        "grade_reason": result["reason"],
        "need_regenerate": not result["grounded"],
    }, "grade_answer", started)


def fallback_answer_node(state: RAGState) -> RAGState:
    started = time.perf_counter()
    if _prefers_chinese(state["question"]):
        answer = (
            "我没有在当前知识库检索结果中找到足够可靠的信息来回答这个问题。"
            f"\n\n原因：{state.get('grade_reason', '答案没有通过依据校验。')}"
        )
    else:
        answer = (
            "I could not find enough reliable information in the current knowledge base results "
            "to answer this question."
            f"\n\nReason: {state.get('grade_reason', 'The answer did not pass grounding check.')}"
        )

    return _with_timing(state, {
        "answer": answer,
        "grounded": False,
        "grounding_checked": True,
        "need_rewrite": False,
        "need_regenerate": False,
        "final_status": "insufficient_context",
    }, "fallback_answer", started)



def route_after_clarify(state: RAGState) -> str:
    if state["need_clarification"]:
        return "need_clarification"
    return "retrieve"


def route_after_document_grading(state: RAGState) -> str:
    if state.get("need_rewrite") and state.get("rewrite_count", 0) < MAX_REWRITE_RETRY:
        return "rewrite"
    if state.get("need_rewrite"):
        return "fallback"
    if state.get("risk_level") == "high":
        return "answer_with_evidence"
    return "fast_answer"


def route_after_retrieve(state: RAGState) -> str:
    if state.get("need_rewrite") and state.get("rewrite_count", 0) < MAX_REWRITE_RETRY:
        return "rewrite"
    if state.get("need_rewrite"):
        return "fallback"
    if state.get("needs_document_grading"):
        return "grade_documents"
    if state.get("risk_level") == "high":
        return "answer_with_evidence"
    return "fast_answer"


def route_after_answer_with_evidence(state: RAGState) -> str:
    if state.get("needs_answer_grading"):
        return "grade_answer"
    return "done"


def route_after_answer_grading(state: RAGState) -> str:
    if state.get("grounded"):
        return "done"
    if state.get("regenerate_count", 0) < MAX_REGENERATE_RETRY:
        return "regenerate"
    return "fallback"


def build_graph():
    graph = StateGraph(RAGState)

    graph.add_node("clarify", clarify_node)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade_documents", grade_documents_node)
    graph.add_node("fast_answer", fast_answer_node)
    graph.add_node("answer_with_evidence", answer_with_evidence_node)
    graph.add_node("regenerate_answer", regenerate_answer_node)
    graph.add_node("grade_answer", grade_answer_node)
    graph.add_node("fallback_answer", fallback_answer_node)

    graph.set_entry_point("clarify")
    graph.add_conditional_edges(
        "clarify",
        route_after_clarify,
        {
            "need_clarification": END,
            "retrieve": "retrieve",
        },
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "rewrite": "rewrite",
            "fallback": "fallback_answer",
            "grade_documents": "grade_documents",
            "fast_answer": "fast_answer",
            "answer_with_evidence": "answer_with_evidence",
        },
    )
    graph.add_conditional_edges(
        "grade_documents",
        route_after_document_grading,
        {
            "rewrite": "rewrite",
            "fallback": "fallback_answer",
            "fast_answer": "fast_answer",
            "answer_with_evidence": "answer_with_evidence",
        },
    )
    graph.add_edge("fast_answer", END)
    graph.add_conditional_edges(
        "answer_with_evidence",
        route_after_answer_with_evidence,
        {
            "grade_answer": "grade_answer",
            "done": END,
        },
    )
    graph.add_edge("regenerate_answer", "grade_answer")
    graph.add_conditional_edges(
        "grade_answer",
        route_after_answer_grading,
        {
            "regenerate": "regenerate_answer",
            "fallback": "fallback_answer",
            "done": END,
        },
    )
    graph.add_edge("fallback_answer", END)
    return graph.compile()


def run_single_question_graph(
    question: str,
    app=None,
    kb_config: KnowledgeBaseConfig | None = None,
) -> RAGState:
    """Run the complete single-question Agentic RAG graph.

    Multi-question workflows should call this function for each decomposed
    sub-question instead of bypassing the graph with a separate retriever flow.
    Passing a compiled app lets callers reuse one graph across parallel tasks.
    """
    graph_app = app or build_graph()
    return graph_app.invoke({"question": question, "kb_config": kb_config})


if __name__ == "__main__":
    app = build_graph()
    questions = [
        "What are the main workflow steps?",
        "How does this workflow connect with downstream validation?",
        "How does this work?",
    ]

    for question in questions:
        print("=" * 60)
        result = run_single_question_graph(question, app)
        print("question:", result.get("question"))
        print("rewritten:", result.get("rewritten_question"))
        print("rewrite_count:", result.get("rewrite_count"))
        print("regenerate_count:", result.get("regenerate_count"))
        print("grounded:", result.get("grounded"))
        print("grounding_checked:", result.get("grounding_checked"))
        print("final_status:", result.get("final_status"))
        print("grade reason:", result.get("grade_reason"))
        print("answer:", result.get("answer"))
