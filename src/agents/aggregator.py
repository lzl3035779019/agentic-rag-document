from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


AGGREGATE_PROMPT = ChatPromptTemplate.from_template(
    """
You are an answer aggregation agent.

Use the sub-answers to answer the original question.

Rules:
1. If the original question is Chinese, answer in Chinese.
2. If the original question is English, answer in English.
3. Do not invent information not present in the sub-answers.
4. If sub-answers are insufficient, clearly say what is missing.

Original question:
{question}

Sub-answers:
{sub_answers}

Final answer:
"""
)


def aggregate_answers(question: str, sub_results: list[dict]) -> str:
    formatted = []
    for index, result in enumerate(sub_results, start=1):
        formatted.append(
            f"Sub-question {index}: {result['sub_question']}\n"
            f"Sub-answer {index}: {result['answer']}"
        )

    prompt = AGGREGATE_PROMPT.invoke(
        {
            "question": question,
            "sub_answers": "\n\n".join(formatted),
        }
    )
    return invoke_llm_cached("aggregate_answers", prompt)


if __name__ == "__main__":
    fake_results = [
        {
            "sub_question": "What are the main steps in the workflow?",
            "answer": "The workflow has several steps described in the source documents.",
        },
        {
            "sub_question": "What constraints affect the workflow?",
            "answer": "The workflow has constraints described in the source documents.",
        },
    ]

    print(
        aggregate_answers(
            "What are the main workflow steps, and what constraints affect them?",
            fake_results,
        )
    )
