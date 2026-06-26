import json

from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


DECOMPOSE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a query decomposition agent for a general enterprise knowledge base RAG system.

Break the user question into 1 to 4 clear retrieval sub-questions.

Rules:
1. If the question is simple, return one sub-question.
2. Write every sub-question in the same language as the user question.
3. If the question is Chinese, the sub-questions must be Chinese. Do not translate them into English.
4. Do not answer the question.
5. Return JSON only.

JSON format:
{{"sub_questions": ["question 1", "question 2"]}}

User question:
{question}
"""
)


def decompose_question(question: str) -> list[str]:
    prompt = DECOMPOSE_PROMPT.invoke({"question": question})
    text = invoke_llm_cached("decompose_question", prompt).strip()

    try:
        data = json.loads(text)
        sub_questions = data.get("sub_questions", [])
        if isinstance(sub_questions, list) and sub_questions:
            return [str(item).strip() for item in sub_questions if str(item).strip()]
    except json.JSONDecodeError:
        pass

    return [question]


if __name__ == "__main__":
    examples = [
        "What are the main workflow steps?",
        "What are the main workflow steps, and what constraints affect them?",
        "这个流程有哪些步骤？有哪些限制？",
    ]

    for example in examples:
        print("=" * 60)
        print("question:", example)
        print("sub questions:", decompose_question(example))
