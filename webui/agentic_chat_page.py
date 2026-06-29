from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agentic_rag import run_agentic_rag
from src.kb_routing import resolve_query_kb_id
from src.streaming_agentic_rag import prepare_complex_answer_stream, prepare_simple_answer_stream
from webui.knowledge_base_panel import render_knowledge_base_panel
from webui.workflow_visualizer import build_single_question_steps, render_mermaid, render_steps


INTRO = "你好，我是 Agentic RAG 可视化助手。输入问题后，我会展示答案和每一步执行状态。"


def _split_answer_sources(answer: str) -> tuple[str, list[str]]:
    body, marker, sources_text = answer.partition("\n\nSources:\n")
    if not marker:
        body, marker, sources_text = answer.partition("\nSources:\n")
    if not marker:
        return answer.strip(), []

    sources: list[str] = []
    for line in sources_text.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if cleaned:
            sources.append(cleaned)
    return body.strip(), sources


def _source_label(source: str) -> str:
    path_part, _, anchor = source.partition("#")
    name = Path(path_part).name or path_part
    if anchor:
        compact_anchor = anchor.replace("parent-", "p")
        return f"{name} · {compact_anchor}"
    return name


def _normalize_markdown_spacing(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\n?Source:\s*`?[^`\n]+`?\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\n)([-*]\s+\*\*)", r"\n\1", text)
    text = re.sub(r"(?<!\\)\$", r"\\$", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _source_snippet_map(source_snippets: list[dict] | None) -> dict[str, str]:
    if not source_snippets:
        return {}
    return {
        str(item.get("source", "")): str(item.get("snippet", "")).strip()
        for item in source_snippets
        if item.get("source") and item.get("snippet")
    }


def _render_sources(sources: list[str], snippets: dict[str, str]) -> None:
    if not sources:
        return

    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for source in sources:
            st.markdown(f"- `{_source_label(source)}`")
            if snippets.get(source):
                st.caption("证据片段")
                st.code(snippets[source], language=None)

        with st.popover("完整路径"):
            for source in sources:
                st.code(source, language=None)


def _assistant_message(answer: str, result: dict[str, Any]) -> dict[str, Any]:
    raw_result = result.get("raw_result", {}) or {}
    return {
        "role": "assistant",
        "content": answer,
        "source_snippets": raw_result.get("source_snippets", []),
        "trace_result": result,
    }


def _trace_result_from_message(message: dict[str, Any]) -> dict[str, Any] | None:
    if message.get("role") != "assistant":
        return None
    trace = message.get("trace_result")
    return trace if isinstance(trace, dict) and trace else None


def _render_answer(
    answer: str,
    source_snippets: list[dict] | None = None,
    *,
    collapsible: bool = True,
    show_sources: bool = True,
) -> None:
    body, sources = _split_answer_sources(answer)
    body = _normalize_markdown_spacing(body)
    snippets = _source_snippet_map(source_snippets)

    if collapsible:
        with st.expander("回答", expanded=True):
            st.markdown(body)
    else:
        with st.container(border=True):
            st.markdown(body)

    if show_sources:
        _render_sources(sources, snippets)


def _render_streamed_sources(result: dict[str, Any]) -> None:
    answer = result.get("answer", "")
    _, sources = _split_answer_sources(answer)
    snippets = _source_snippet_map(result.get("raw_result", {}).get("source_snippets", []))
    _render_sources(sources, snippets)


def _init_state() -> None:
    st.session_state.setdefault("messages", [{"role": "assistant", "content": INTRO}])


def _clear() -> None:
    st.session_state["messages"] = [{"role": "assistant", "content": INTRO}]


def _render_summary(result: dict[str, Any]) -> None:
    raw = result.get("raw_result", {}) or {}
    timings = result.get("timings_ms", {}) or {}

    cols = st.columns(5)
    cols[0].metric("Route", result.get("route", "-"))
    cols[1].metric("Router", result.get("router", "-"))
    cols[2].metric("Risk", raw.get("risk_level", "-"))
    cols[3].metric("Status", raw.get("final_status", "-"))
    cols[4].metric("Total", f"{timings.get('total_ms', 0):.0f} ms")

    route_reason = result.get("route_reason", "")
    if route_reason:
        st.caption(route_reason)

    kb = result.get("knowledge_base")
    if kb:
        st.caption(
            "Knowledge base: "
            f"{kb.get('name', '-')} | `{kb.get('collection_name', '-')}` | "
            f"`{kb.get('embedding_model', '-')}`"
        )


def _render_complex(raw: dict[str, Any]) -> None:
    sub_results = raw.get("sub_results", []) or []
    if not sub_results:
        return

    st.markdown("#### 子问题执行")
    for index, item in enumerate(sub_results, start=1):
        with st.container(border=True):
            st.markdown(f"**子问题 {index}: {item.get('sub_question', '')}**")
            cols = st.columns(4)
            cols[0].metric("Risk", item.get("risk_level", "-"))
            cols[1].metric("Status", item.get("final_status", "-"))
            cols[2].metric("Sources", item.get("source_count", 0))
            cols[3].metric("Latency", f"{item.get('latency_ms', 0):.0f} ms")
            _render_answer(item.get("answer", ""), collapsible=False, show_sources=False)
            sources = item.get("sources") or []
            if sources:
                st.caption("Sources: " + ", ".join(str(source) for source in sources[:5]))


def _render_complex_progress(sub_questions: list[str], sub_results: list[dict[str, Any]]) -> None:
    if not sub_questions:
        st.caption("正在分解问题...")
        return

    completed = {item.get("index"): item for item in sub_results}
    st.caption(f"子问题进度：{len(sub_results)}/{len(sub_questions)}")
    for index, sub_question in enumerate(sub_questions, start=1):
        item = completed.get(index)
        st.markdown(f"**{index}. {sub_question}**")
        if item:
            st.caption(
                f"Status: {item.get('final_status', '-')} | "
                f"Risk: {item.get('risk_level', '-')} | "
                f"Latency: {item.get('latency_ms', 0):.0f} ms"
            )
        else:
            st.caption("等待执行...")


def _render_trace_body(result: dict[str, Any]) -> None:
    st.markdown("#### 执行概览")
    _render_summary(result)

    st.markdown("#### 流程图")
    render_mermaid(result)

    raw = result.get("raw_result", {}) or {}
    if result.get("route") == "complex":
        _render_complex(raw)
    else:
        st.markdown("#### 节点状态")
        render_steps(build_single_question_steps(raw))

    with st.popover("查看 raw_result"):
        st.json(raw, expanded=False)


def _render_trace(result: dict[str, Any]) -> None:
    with st.expander("模型思考过程", expanded=False):
        _render_trace_body(result)


def _render_history() -> None:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                trace = _trace_result_from_message(message)
                if trace:
                    _render_trace(trace)
                _render_answer(
                    message["content"],
                    message.get("source_snippets"),
                    collapsible=bool(trace),
                )
            else:
                st.write(message["content"])


def _resolve_query_kb_id(question: str, selected_kb: Any) -> str:
    return resolve_query_kb_id(
        question=question,
        manual_kb_id=selected_kb.kb_id,
        auto_route_enabled=bool(st.session_state.get("auto_route_kb")),
        chinese_kb_id=st.session_state.get("chinese_kb_id"),
        english_kb_id=st.session_state.get("english_kb_id"),
    )


def _run_question(question: str, selected_kb: Any) -> tuple[str, dict[str, Any]]:
    try:
        query_kb_id = _resolve_query_kb_id(question, selected_kb)
        result = run_agentic_rag(question, kb_id=query_kb_id)
        answer = result.get("answer", "未生成答案。")
        return answer, result
    except Exception as exc:
        result = {
            "question": question,
            "route": "error",
            "router": "error",
            "route_reason": str(exc),
            "answer": f"运行失败：{type(exc).__name__}: {exc}",
            "timings_ms": {},
            "raw_result": {"final_status": "error", "error": str(exc)},
        }
        return result["answer"], result


def _try_stream_complex_question(question: str, query_kb_id: str) -> tuple[str, dict[str, Any]] | None:
    prepared = prepare_complex_answer_stream(question, kb_id=query_kb_id)
    if prepared is None:
        return None

    with st.chat_message("assistant"):
        answer = ""
        result: dict[str, Any] | None = None
        sub_questions: list[str] = []
        sub_results: list[dict[str, Any]] = []

        with st.expander("模型思考过程", expanded=False):
            trace_body = st.empty()
            trace_body.caption("正在分解问题...")

        with st.expander("回答", expanded=True):
            answer_slot = st.empty()
            answer_slot.caption("正在执行子问题...")
            for event in prepared.stream_events():
                event_type = event.get("type")
                if event_type == "sub_questions":
                    sub_questions = event.get("sub_questions", [])
                    with trace_body.container():
                        _render_complex_progress(sub_questions, sub_results)
                elif event_type == "sub_result":
                    sub_results.append(event["result"])
                    with trace_body.container():
                        _render_complex_progress(sub_questions, sub_results)
                elif event_type == "aggregate_start":
                    answer_slot.caption("正在汇总最终答案...")
                elif event_type == "aggregate_chunk":
                    answer += event.get("text", "")
                    answer_slot.markdown(_normalize_markdown_spacing(answer + "▌"))
                elif event_type == "final":
                    result = event["result"]

            answer_slot.markdown(_normalize_markdown_spacing(answer))

        if result is None:
            raise RuntimeError("复杂问题流式执行未返回最终结果。")

        with trace_body.container():
            _render_trace_body(result)
        _render_streamed_sources(result)

    return result["answer"], result


def _try_stream_simple_question(question: str, query_kb_id: str) -> tuple[str, dict[str, Any]] | None:
    prepared = prepare_simple_answer_stream(question, kb_id=query_kb_id)
    if prepared is None:
        return None

    with st.chat_message("assistant"):
        with st.expander("模型思考过程", expanded=False):
            trace_body = st.empty()
            trace_body.caption("已完成路由和检索，正在生成答案...")

        with st.expander("回答", expanded=True):
            answer_body = st.write_stream(prepared.stream_answer())

        result = prepared.finish(answer_body)
        with trace_body.container():
            _render_trace_body(result)
        if result.get("raw_result", {}).get("final_status") == "streamed_answer_failed_grounding":
            st.warning("这次流式答案没有通过依据校验，请以模型思考过程中的 grounding 结果为准。")
        _render_streamed_sources(result)

    return result["answer"], result


def _try_stream_question(question: str, selected_kb: Any) -> tuple[str, dict[str, Any]] | None:
    query_kb_id = _resolve_query_kb_id(question, selected_kb)
    return _try_stream_complex_question(question, query_kb_id) or _try_stream_simple_question(question, query_kb_id)


def agentic_chat_page() -> None:
    _init_state()

    st.title("Agentic RAG Visualized")
    st.caption("保留上个项目的 Agentic RAG 核心，用 Streamlit 展示路由、检索、改写、证据、校验和耗时。")

    with st.sidebar:
        st.header("知识库")
        selected_kb = render_knowledge_base_panel()
        st.divider()
        st.header("运行前提")
        st.markdown(
            "- Qdrant 已启动：`docker compose up -d`\n"
            "- 已构建向量库：`python -m src.qdrant_store`\n"
            "- `src/.env` 中配置 `QWEN_API_KEY`"
        )
        st.button("清空对话", on_click=_clear, use_container_width=True)

    _render_history()

    question = st.chat_input("请输入问题")
    if not question:
        return

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Agentic RAG 正在准备流式执行..."):
            streamed = _try_stream_question(question, selected_kb)

        if streamed is None:
            with st.spinner("Agentic RAG 正在执行..."):
                answer, result = _run_question(question, selected_kb)
            with st.chat_message("assistant"):
                _render_trace(result)
                _render_answer(answer, result.get("raw_result", {}).get("source_snippets", []))
        else:
            answer, result = streamed
    except Exception as exc:
        result = {
            "question": question,
            "route": "error",
            "router": "error",
            "route_reason": str(exc),
            "answer": f"运行失败：{type(exc).__name__}: {exc}",
            "timings_ms": {},
            "raw_result": {"final_status": "error", "error": str(exc)},
        }
        answer = result["answer"]
        with st.chat_message("assistant"):
            _render_answer(answer)

    st.session_state["messages"].append(_assistant_message(answer, result))
