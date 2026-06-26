import time

from src.agents.muti_agent_rag import run_multi_agent_rag
from src.graph import build_graph, run_single_question_graph
from src.knowledge_base import get_knowledge_base
from src.router import route_question


def run_agentic_rag(question: str, kb_id: str | None = None) -> dict:
    started = time.perf_counter()
    kb_config = get_knowledge_base(kb_id)

    route_started = time.perf_counter()
    route_result = route_question(question)
    route_ms = round((time.perf_counter() - route_started) * 1000, 2)

    if route_result["route"] == "complex":
        rag_result = run_multi_agent_rag(question, kb_config=kb_config)
    else:
        graph_app = build_graph()
        rag_result = run_single_question_graph(question, graph_app, kb_config=kb_config)

    total_ms = round((time.perf_counter() - started) * 1000, 2)
    answer = rag_result.get("answer", "")

    return {
        "question": question,
        "route": route_result["route"],
        "router": route_result["router"],
        "route_reason": route_result["reason"],
        "answer": answer,
        "knowledge_base": {
            "kb_id": kb_config.kb_id,
            "name": kb_config.name,
            "collection_name": kb_config.collection_name,
            "embedding_model": kb_config.embedding_model,
        },
        "timings_ms": {
            "route_ms": route_ms,
            "total_ms": total_ms,
            "rag_ms": round(total_ms - route_ms, 2),
        },
        "raw_result": rag_result,
    }


if __name__ == "__main__":
    examples = [
        "What is the retry policy?",
        "Compare synchronous and asynchronous processing.",
        "What should I know before changing this configuration?",
    ]

    for question in examples:
        print("=" * 80)
        result = run_agentic_rag(question)
        print("question:", result["question"])
        print("route:", result["route"])
        print("router:", result["router"])
        print("route_reason:", result["route_reason"])
        print("timings_ms:", result["timings_ms"])
        print("answer:", result["answer"][:1000])
