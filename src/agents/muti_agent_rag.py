import asyncio
import time

from src.agents.aggregator import aggregate_answers
from src.agents.decomposer import decompose_question
from src.graph import build_graph, run_single_question_graph
from src.knowledge_base import KnowledgeBaseConfig
from src.observability import langfuse, trace_step


async def answer_sub_question_async(
    sub_question: str,
    graph_app,
    kb_config: KnowledgeBaseConfig | None = None,
) -> dict:
    """Run one decomposed sub-question through the full graph.py RAG flow."""
    started = time.perf_counter()

    with trace_step("graph.answer_sub_question", {"sub_question": sub_question}) as span:
        # Do not bypass graph.py here. Each sub-question should still use the
        # same rewrite, retrieval, risk routing, cache, and grounding policy.
        result = await asyncio.to_thread(
            run_single_question_graph,
            sub_question,
            graph_app,
            kb_config,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)

        documents = result.get("documents", [])
        sub_result = {
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
            "latency_ms": latency_ms,
        }

        span.update(
            output={
                "answer": sub_result["answer"],
                "source_count": sub_result["source_count"],
                "risk_level": sub_result["risk_level"],
                "final_status": sub_result["final_status"],
                "latency_ms": latency_ms,
            }
        )

    return sub_result


async def run_multi_agent_rag_async(
    question: str,
    kb_config: KnowledgeBaseConfig | None = None,
) -> dict:
    """Decompose a complex question, answer sub-questions in parallel, then aggregate."""
    total_started = time.perf_counter()
    timings: dict[str, float] = {}

    with trace_step("multi_agent_rag", {"question": question}) as root_span:
        with trace_step("decompose", {"question": question}) as span:
            started = time.perf_counter()
            sub_questions = decompose_question(question)
            timings["decompose_ms"] = round((time.perf_counter() - started) * 1000, 2)
            span.update(output={"sub_questions": sub_questions})

        graph_app = build_graph()

        with trace_step("parallel_graph_sub_questions", {"sub_questions": sub_questions}) as span:
            started = time.perf_counter()
            tasks = [
                answer_sub_question_async(sub_question, graph_app, kb_config)
                for sub_question in sub_questions
            ]
            sub_results = await asyncio.gather(*tasks)
            timings["parallel_sub_questions_ms"] = round(
                (time.perf_counter() - started) * 1000,
                2,
            )
            span.update(
                output={
                    "sub_result_count": len(sub_results),
                    "sub_questions": [item["sub_question"] for item in sub_results],
                    "sub_statuses": [item["final_status"] for item in sub_results],
                    "sub_latencies_ms": [item["latency_ms"] for item in sub_results],
                }
            )

        with trace_step("aggregate", {"sub_question_count": len(sub_questions)}) as span:
            started = time.perf_counter()
            final_answer = aggregate_answers(question, sub_results)
            timings["aggregate_ms"] = round((time.perf_counter() - started) * 1000, 2)
            span.update(output={"answer": final_answer})

        timings["total_ms"] = round((time.perf_counter() - total_started) * 1000, 2)
        root_span.update(
            output={
                "sub_questions": sub_questions,
                "timings_ms": timings,
                "final_answer": final_answer,
            }
        )

    langfuse.flush()

    return {
        "question": question,
        "sub_questions": sub_questions,
        "sub_results": sub_results,
        "answer": final_answer,
        "timings_ms": timings,
    }


def run_multi_agent_rag(
    question: str,
    kb_config: KnowledgeBaseConfig | None = None,
) -> dict:
    return asyncio.run(run_multi_agent_rag_async(question, kb_config=kb_config))


if __name__ == "__main__":
    question = "What are the main workflow steps, and what constraints affect them?"
    result = run_multi_agent_rag(question)

    print("question:", result["question"])
    print("sub questions:", result["sub_questions"])
    print("timings_ms:", result["timings_ms"])
    print("sub results:")
    for item in result["sub_results"]:
        print(
            {
                "sub_question": item["sub_question"],
                "source_count": item["source_count"],
                "risk_level": item["risk_level"],
                "final_status": item["final_status"],
                "latency_ms": item["latency_ms"],
                "node_timings_ms": item["node_timings_ms"],
            }
        )
    print("answer:", result["answer"])
