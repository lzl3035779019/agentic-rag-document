from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st


@dataclass
class StepView:
    name: str
    status: str
    detail: str = ""
    elapsed_ms: float | None = None


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.2f} ms"


def _status_icon(status: str) -> str:
    return {
        "done": "✅",
        "skipped": "⚪",
        "warning": "⚠️",
        "failed": "❌",
        "active": "🔵",
    }.get(status, "⚪")


def _documents_summary(raw_result: dict[str, Any]) -> str:
    docs = raw_result.get("documents", []) or []
    if not docs:
        return "未返回文档"
    sources = raw_result.get("retrieved_sources", []) or []
    unique_sources = []
    for source in sources:
        if source and source not in unique_sources:
            unique_sources.append(source)
    return f"{len(docs)} 个文档；来源 {len(unique_sources)} 个"


def build_single_question_steps(raw_result: dict[str, Any]) -> list[StepView]:
    timings = raw_result.get("node_timings_ms", {}) or {}
    need_clarification = bool(raw_result.get("need_clarification"))
    rewrite_count = int(raw_result.get("rewrite_count", 0) or 0)
    final_status = str(raw_result.get("final_status", "") or "")
    risk_level = str(raw_result.get("risk_level", "") or "unknown")
    grounded = raw_result.get("grounded")
    grounding_checked = bool(raw_result.get("grounding_checked"))

    steps = [
        StepView(
            "clarify",
            "warning" if need_clarification else "done",
            raw_result.get("clarification_message", "问题足够明确，进入检索。")
            if need_clarification
            else "问题足够明确，进入检索。",
            timings.get("clarify"),
        )
    ]
    if need_clarification:
        return steps

    steps.append(
        StepView(
            "retrieve",
            "done" if raw_result.get("documents") else "warning",
            _documents_summary(raw_result),
            timings.get("retrieve"),
        )
    )
    steps.append(
        StepView(
            "rewrite",
            "done" if rewrite_count else "skipped",
            f"改写次数：{rewrite_count}" if rewrite_count else "检索未触发改写。",
            timings.get("rewrite"),
        )
    )
    steps.append(
        StepView(
            "risk",
            "done",
            f"风险等级：{risk_level}",
            None,
        )
    )

    if timings.get("grade_documents") is not None:
        steps.append(
            StepView(
                "grade_documents",
                "done",
                raw_result.get("grade_reason", "已执行文档相关性评分。"),
                timings.get("grade_documents"),
            )
        )
    else:
        steps.append(
            StepView(
                "grade_documents",
                "skipped",
                "当前主流程默认未开启文档评分。",
                None,
            )
        )

    if timings.get("fast_answer") is not None:
        steps.append(
            StepView(
                "fast_answer",
                "done" if final_status != "insufficient_context" else "warning",
                "低/中风险快速回答路径。",
                timings.get("fast_answer"),
            )
        )
    if timings.get("answer_with_evidence") is not None:
        steps.append(
            StepView(
                "answer_with_evidence",
                "done",
                "高风险证据抽取与回答路径。",
                timings.get("answer_with_evidence"),
            )
        )
    if timings.get("grade_answer") is not None:
        steps.append(
            StepView(
                "grade_answer",
                "done" if grounded else "warning",
                raw_result.get("grade_reason", "已执行 grounding 校验。"),
                timings.get("grade_answer"),
            )
        )
    elif not grounding_checked:
        steps.append(
            StepView(
                "grade_answer",
                "skipped",
                "当前路径未执行严格答案校验。",
                None,
            )
        )
    if timings.get("regenerate_answer") is not None:
        steps.append(
            StepView(
                "regenerate_answer",
                "done",
                f"重生成次数：{raw_result.get('regenerate_count', 0)}",
                timings.get("regenerate_answer"),
            )
        )
    if timings.get("fallback_answer") is not None or final_status == "insufficient_context":
        steps.append(
            StepView(
                "fallback_answer",
                "failed" if final_status == "insufficient_context" else "warning",
                raw_result.get("grade_reason", "信息不足，返回保守答案。"),
                timings.get("fallback_answer"),
            )
        )

    return steps


def render_steps(steps: list[StepView]) -> None:
    for step in steps:
        with st.container(border=True):
            left, mid, right = st.columns([1.6, 6, 1.5])
            left.markdown(f"**{_status_icon(step.status)} {step.name}**")
            mid.write(step.detail or "无详情")
            right.caption(_fmt_ms(step.elapsed_ms))


def render_mermaid(result: dict[str, Any]) -> None:
    route = result.get("route", "simple")
    raw = result.get("raw_result", {}) or {}
    final_status = raw.get("final_status", "")
    risk = raw.get("risk_level", "")
    rewrite_count = raw.get("rewrite_count", 0)

    if route == "complex":
        sub_questions = raw.get("sub_questions", []) or []
        sub_nodes = "\n".join(
            f'    Q{i}["子问题 {i}: {question[:28]}"] --> A{i}["跑完整 graph.py"]'
            for i, question in enumerate(sub_questions, start=1)
        )
        joins = "\n".join(f"    A{i} --> G" for i in range(1, len(sub_questions) + 1))
        chart = f"""
flowchart TD
    U["用户问题"] --> R["router: complex"]
    R --> D["decompose_question"]
{sub_nodes}
{joins}
    G["aggregate_answers"] --> F["最终答案"]
"""
    else:
        answer_node = "fast_answer"
        if raw.get("node_timings_ms", {}).get("answer_with_evidence") is not None:
            answer_node = "answer_with_evidence"
        chart = f"""
flowchart TD
    U["用户问题"] --> C["clarify"]
    C --> R["retrieve"]
    R --> W{{"rewrite_count={rewrite_count}"}}
    W --> K["risk_level={risk}"]
    K --> A["{answer_node}"]
    A --> G["grade/fallback 状态: {final_status}"]
    G --> F["最终答案"]
"""

    st.code(f"```mermaid\n{chart.strip()}\n```", language="markdown")
    st.caption("上方是 Mermaid 源码，可复制到支持 Mermaid 的 Markdown 查看图形。")


def render_raw_result(result: dict[str, Any]) -> None:
    raw = result.get("raw_result", {}) or {}
    with st.expander("查看 raw_result", expanded=False):
        st.json(raw, expanded=False)
