import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


BATCH_RETRIEVAL_GRADER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a retrieval grader for a general enterprise knowledge base RAG system.

Decide whether each document may help answer the user question.
Be permissive. A document is relevant if it contains related policy, context,
terminology, entities, procedures, tools, numbers, dates, or partial information.
Return false only if the document is clearly unrelated.

Return JSON only:
{{
  "results": [
    {{"index": 0, "relevant": true, "reason": "short reason"}},
    {{"index": 1, "relevant": false, "reason": "short reason"}}
  ]
}}

Question:
{question}

Documents:
{documents}
"""
)


ANSWER_GRADER_PROMPT = ChatPromptTemplate.from_template(
    """
You are an answer grader for a general enterprise knowledge base RAG system.
Check whether the answer is supported by the context.

Rules:
1. Mark grounded=true only when all factual claims in the answer are supported by the context.
2. Mark grounded=false if the answer adds unsupported policies, procedures, tools, numbers, dates, people, departments, or document names.
3. Do not require the answer to mention information that is not in the context.
4. A conservative "not enough information" answer is grounded if the context is insufficient.

Return JSON only:
{{"grounded": true or false, "reason": "short reason"}}

Question:
{question}

Context:
{context}

Answer:
{answer}
"""
)


def _parse_bool_json(text: str, key: str) -> tuple[bool, str]:
    try:
        data = json.loads(text)
        return bool(data.get(key)), str(data.get("reason", ""))
    except json.JSONDecodeError:
        lowered = text.lower()
        return "true" in lowered, text


def _format_docs_for_batch_grading(docs: list[Document]) -> str:
    blocks = []
    for index, doc in enumerate(docs):
        source = doc.metadata.get("source", "")
        parent_id = doc.metadata.get("parent_id", "")
        blocks.append(
            f"[{index}] source={source} parent_id={parent_id}\n"
            f"{doc.page_content[:1800]}"
        )
    return "\n\n---\n\n".join(blocks)


def _parse_batch_grading(text: str, docs_count: int) -> list[tuple[int, bool, str]]:
    data = json.loads(text)
    results = data.get("results", [])
    parsed = []
    for item in results:
        index = int(item.get("index"))
        if 0 <= index < docs_count:
            parsed.append(
                (
                    index,
                    bool(item.get("relevant")),
                    str(item.get("reason", "")),
                )
            )
    return parsed


def grade_documents(question: str, docs: list[Document]) -> list[Document]:
    if not docs:
        return []

    prompt = BATCH_RETRIEVAL_GRADER_PROMPT.invoke(
        {
            "question": question,
            "documents": _format_docs_for_batch_grading(docs),
        }
    )
    text = invoke_llm_cached("grade_documents_batch", prompt)

    try:
        graded = _parse_batch_grading(text, len(docs))
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        for doc in docs:
            doc.metadata["relevance_reason"] = f"Batch grader returned invalid JSON; kept by fallback. Raw: {text[:300]}"
        return docs

    relevant_docs = []
    graded_indexes = set()
    for index, relevant, reason in graded:
        graded_indexes.add(index)
        doc = docs[index]
        doc.metadata["relevance_reason"] = reason

        if relevant:
            relevant_docs.append(doc)

    for index, doc in enumerate(docs):
        if index not in graded_indexes:
            doc.metadata["relevance_reason"] = "Batch grader did not return a result for this document."

    return relevant_docs


def grade_answer(question: str, context: str, answer: str) -> dict:
    prompt = ANSWER_GRADER_PROMPT.invoke(
        {
            "question": question,
            "context": context[:4000],
            "answer": answer,
        }
    )
    text = invoke_llm_cached("grade_answer", prompt)
    grounded, reason = _parse_bool_json(text, "grounded")

    return {
        "grounded": grounded,
        "reason": reason,
        "raw": text,
    }
