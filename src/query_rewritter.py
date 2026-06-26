from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a query rewriting module for a general enterprise knowledge base.

Your task is to rewrite the user's question into one retrieval query in the same language as the user question.

Rules:
1. Keep the rewritten query in the same language as the user question.
2. If the question is Chinese, the rewritten query must be Chinese.
3. Preserve the user's intent exactly.
4. Preserve named entities, numbers, dates, product names, system names, policy names, document names, and technical terms.
5. Add only necessary synonyms or related terms that are directly implied by the user's question.
6. Do not add domain-specific topics that are not implied by the question.
7. Do not specialize the query for a single company, handbook, department, or document type.
8. Do not answer the question.
9. Do not ask for clarification.
10. Output only the rewritten query.

Good rewrites:
- startup configuration timeout retry behavior
- access permission restriction sensitive data internal system
- monitoring alerting operational insight logs application health
- data retention deletion rollback irreversible operation

Bad rewrites:
- general document policies
- please provide a more specific question
- all topics tools systems

User question:
{question}
"""
)


def rewrite_query(question: str) -> str:
    prompt = REWRITE_PROMPT.invoke({"question": question})
    return invoke_llm_cached("rewrite_query", prompt).strip()


FEEDBACK_REWRITE_PROMPT = ChatPromptTemplate.from_template(
    """
You are improving a failed retrieval query for a general enterprise knowledge base.

Rewrite the query again using the failure feedback.

Rules:
1. Output one retrieval query only, in the same language as the original question.
2. If the original question is Chinese, the improved query must be Chinese.
3. Preserve the original user intent.
4. Use the previous rewritten query only as a starting point.
5. Use the failure reason to avoid repeating the same mistake.
6. If retrieved sources look irrelevant, add more specific entities, policy names, product names, numbers, dates, or technical terms from the original question.
7. If the previous query was too broad, make it more specific.
8. If the previous query was too narrow, add directly implied synonyms.
9. Do not add unrelated domain-specific topics.
10. Do not answer the question.

Original question:
{question}

Previous rewritten query:
{previous_query}

Failure reason:
{grade_reason}

Retrieved sources:
{retrieved_sources}

Improved retrieval query:
"""
)


def rewrite_query_with_feedback(
    question: str,
    previous_query: str,
    grade_reason: str = "",
    retrieved_sources: list[str] | None = None,
) -> str:
    prompt = FEEDBACK_REWRITE_PROMPT.invoke(
        {
            "question": question,
            "previous_query": previous_query,
            "grade_reason": grade_reason or "No explicit reason provided.",
            "retrieved_sources": "\n".join(retrieved_sources or []) or "No sources retrieved.",
        }
    )
    return invoke_llm_cached("rewrite_query_with_feedback", prompt).strip()


if __name__ == "__main__":
    questions = [
        "这个系统的核心流程有哪些？",
        "What are the main steps in this workflow?",
    ]
    for question in questions:
        print("=" * 40)
        print("original:", question)
        print("rewritten:", rewrite_query(question))
