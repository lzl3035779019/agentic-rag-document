from __future__ import annotations


def infer_query_language(question: str) -> str:
    return "zh" if any("\u4e00" <= char <= "\u9fff" for char in question) else "en"


def resolve_query_kb_id(
    *,
    question: str,
    manual_kb_id: str,
    auto_route_enabled: bool,
    chinese_kb_id: str | None,
    english_kb_id: str | None,
) -> str:
    if not auto_route_enabled:
        return manual_kb_id

    language = infer_query_language(question)
    if language == "zh" and chinese_kb_id:
        return chinese_kb_id
    if language == "en" and english_kb_id:
        return english_kb_id
    return manual_kb_id
