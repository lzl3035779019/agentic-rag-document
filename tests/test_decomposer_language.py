from src.agents.decomposer import DECOMPOSE_PROMPT


def test_decompose_prompt_preserves_user_question_language():
    prompt_text = DECOMPOSE_PROMPT.format(question="Workflow 和 Agent 分别适合什么场景？")

    assert "same language as the user question" in prompt_text
    assert "translate sub-questions into English" not in prompt_text
