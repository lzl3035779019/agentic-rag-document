from src.evidence import (
    ANSWER_WITH_EVIDENCE_PROMPT_EN,
    ANSWER_WITH_EVIDENCE_PROMPT_ZH,
    EVIDENCE_ANSWER_PROMPT_EN,
    EVIDENCE_ANSWER_PROMPT_ZH,
    EVIDENCE_PROMPT_EN,
    EVIDENCE_PROMPT_ZH,
    FAST_ANSWER_CONTEXT_CHAR_LIMIT,
    FAST_ANSWER_PROMPT_EN,
    FAST_ANSWER_PROMPT_ZH,
    _select_answer_with_evidence_prompt,
    _select_evidence_answer_prompt,
    _select_evidence_prompt,
    _select_fast_answer_prompt,
)


def test_chinese_question_uses_chinese_fast_answer_prompt():
    assert _select_fast_answer_prompt("Agent 的工作模式或者有什么？") is FAST_ANSWER_PROMPT_ZH


def test_english_question_uses_english_fast_answer_prompt():
    assert _select_fast_answer_prompt("What are the Agent working modes?") is FAST_ANSWER_PROMPT_EN


def test_fast_answer_uses_medium_detail_context_limit():
    assert FAST_ANSWER_CONTEXT_CHAR_LIMIT >= 12000


def test_chinese_question_uses_chinese_evidence_prompts():
    question = "\u5ba2\u6237\u6570\u636e\u6743\u9650\u600e\u4e48\u5904\u7406\uff1f"

    assert _select_evidence_prompt(question) is EVIDENCE_PROMPT_ZH
    assert _select_evidence_answer_prompt(question) is EVIDENCE_ANSWER_PROMPT_ZH
    assert _select_answer_with_evidence_prompt(question) is ANSWER_WITH_EVIDENCE_PROMPT_ZH


def test_english_question_uses_english_evidence_prompts():
    question = "How should customer data permissions be handled?"

    assert _select_evidence_prompt(question) is EVIDENCE_PROMPT_EN
    assert _select_evidence_answer_prompt(question) is EVIDENCE_ANSWER_PROMPT_EN
    assert _select_answer_with_evidence_prompt(question) is ANSWER_WITH_EVIDENCE_PROMPT_EN
