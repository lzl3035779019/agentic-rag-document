from webui.agentic_chat_page import (
    _assistant_message,
    _normalize_markdown_spacing,
    _source_snippet_map,
    _trace_result_from_message,
)


def test_normalize_markdown_spacing_does_not_split_dates():
    text = "Summer Hours: 4-day work weeks from May 1 through August 31.\n\n* Sick Time: Untracked."

    assert _normalize_markdown_spacing(text) == text


def test_normalize_markdown_spacing_escapes_dollar_amounts():
    text = "FSA: Optional medical account ($3,300/year)."

    assert r"\$3,300/year" in _normalize_markdown_spacing(text)


def test_source_snippet_map_uses_source_reference_keys():
    snippets = [
        {"source": "doc.md#parent-1", "snippet": "原文片段"},
        {"source": "", "snippet": "ignored"},
    ]

    assert _source_snippet_map(snippets) == {"doc.md#parent-1": "原文片段"}
def test_assistant_message_keeps_trace_with_its_own_answer():
    result = {
        "answer": "answer",
        "raw_result": {"source_snippets": [{"source": "doc.md#parent-1", "snippet": "text"}]},
    }

    message = _assistant_message("answer", result)

    assert message["role"] == "assistant"
    assert message["content"] == "answer"
    assert message["source_snippets"] == [{"source": "doc.md#parent-1", "snippet": "text"}]
    assert message["trace_result"] is result


def test_trace_result_from_message_ignores_intro_messages():
    assert _trace_result_from_message({"role": "assistant", "content": "hello"}) is None
    assert _trace_result_from_message({"role": "user", "content": "hello", "trace_result": {}}) is None
