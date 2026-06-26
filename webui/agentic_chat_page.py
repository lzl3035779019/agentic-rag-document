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
from webui.knowledge_base_panel import render_knowledge_base_panel
from webui.workflow_visualizer import (
    build_single_question_steps,
    render_mermaid,
    render_steps,
)


INTRO = "你好，我是 Agentic RAG 可视化助手。输入问题后，我会展示答案和每一步执行状态。"


def _split_answer_sources(answer: str) -> tuple[str, list[str]]:
    body, marker, sources_text = answer.partition("\n\nSources:\n")
    if not marker:
        body, marker, sources_text = answer.partition("\nSources:\n")
    if not marker:
        return answer.strip(), []

    sources = []
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

    if show_sources and sources:
        with st.expander(f"Sources ({len(sources)})", expanded=False):
            for source in sources:
                st.markdown(f"- `{_source_label(source)}`")
                if snippets.get(source):
                    st.caption("证据片段")
                    st.code(snippets[source], language=None)
            with st.popover("完整路径"):
                for source in sources:
                    st.code(source, language=None)


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

    st.caption(result.get("route_reason", ""))
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
            if item.get("sources"):
                st.caption("Sources: " + ", ".join(str(s) for s in item.get("sources", [])[:5]))


def _render_trace(result: dict[str, Any]) -> None:
    with st.expander("模型思考过程", expanded=False):
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


def _render_history() -> None:
    for message in st.session_state["messages"]:
        trace = _trace_result_from_message(message)
        if trace:
            _render_trace(trace)

        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                _render_answer(
                    message["content"],
                    message.get("source_snippets"),
                    collapsible=bool(trace),
                )
            else:
                st.write(message["content"])


def _run_question(question: str, selected_kb: Any) -> tuple[str, dict[str, Any]]:
    try:
        query_kb_id = resolve_query_kb_id(
            question=question,
            manual_kb_id=selected_kb.kb_id,
            auto_route_enabled=bool(st.session_state.get("auto_route_kb")),
            chinese_kb_id=st.session_state.get("chinese_kb_id"),
            english_kb_id=st.session_state.get("english_kb_id"),
        )
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

    with st.spinner("Agentic RAG 正在执行..."):
        answer, result = _run_question(question, selected_kb)

    _render_trace(result)
    with st.chat_message("assistant"):
        _render_answer(answer, result.get("raw_result", {}).get("source_snippets", []))

    st.session_state["messages"].append(_assistant_message(answer, result))
