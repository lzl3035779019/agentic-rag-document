from src.query_rewritter import FEEDBACK_REWRITE_PROMPT, REWRITE_PROMPT


def test_rewrite_prompt_preserves_user_question_language():
    prompt_text = REWRITE_PROMPT.format(question="Agent 的工作模式有哪些？")

    assert "same language as the user question" in prompt_text
    assert "translate it into English" not in prompt_text


def test_feedback_rewrite_prompt_preserves_user_question_language():
    prompt_text = FEEDBACK_REWRITE_PROMPT.format(
        question="Agent 的工作模式有哪些？",
        previous_query="Agent 工作模式",
        grade_reason="No relevant documents.",
        retrieved_sources="",
    )

    assert "same language as the original question" in prompt_text
    assert "English retrieval query" not in prompt_text
