from src.clarification import (
    CLARIFICATION_PROMPT_EN,
    CLARIFICATION_PROMPT_ZH,
    _needs_llm_clarification,
    _select_clarification_prompt,
)


def test_specific_chinese_question_does_not_need_clarification():
    question = "Agent \u7684\u5de5\u4f5c\u6a21\u5f0f\u6216\u8005\u6709\u4ec0\u4e48\uff1f"

    assert not _needs_llm_clarification(question)


def test_specific_chinese_policy_question_does_not_need_clarification():
    question = "\u5458\u5de5\u798f\u5229\u6709\u54ea\u4e9b\uff1f"

    assert not _needs_llm_clarification(question)


def test_vague_chinese_reference_needs_clarification():
    question = "\u8fd9\u4e2a\u5462\uff1f"

    assert _needs_llm_clarification(question)


def test_chinese_question_uses_chinese_clarification_prompt():
    question = "\u8fd9\u4e2a\u600e\u4e48\u5904\u7406\uff1f"

    assert _select_clarification_prompt(question) is CLARIFICATION_PROMPT_ZH


def test_english_question_uses_english_clarification_prompt():
    assert _select_clarification_prompt("How does this work?") is CLARIFICATION_PROMPT_EN
