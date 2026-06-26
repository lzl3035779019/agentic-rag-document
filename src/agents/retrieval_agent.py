from langchain_core.prompts import ChatPromptTemplate

from src.basic_rag import format_docs
from src.graders import grade_documents
from src.hybrid_retriever import HybridRetriever
from src.llm import get_llm


SUB_ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a retrieval agent.
Answer the sub-question using only the context.

If the context does not contain enough information, say:
Not enough information found in the knowledge base.

Context:
{context}

Sub-question:
{sub_question}

Sub-answer:
"""
)


def answer_sub_question(sub_question: str, retriever: HybridRetriever) -> dict:
    docs = retriever.retrieve_parents(sub_question)
    relevant_docs = grade_documents(sub_question, docs)

    llm = get_llm()
    prompt = SUB_ANSWER_PROMPT.invoke(
        {
            "context": format_docs(relevant_docs),
            "sub_question": sub_question,
        }
    )
    response = llm.invoke(prompt)

    return {
        "sub_question": sub_question,
        "answer": response.content,
        "documents": relevant_docs,
        "source_count": len(relevant_docs),
    }


if __name__ == "__main__":
    retriever = HybridRetriever()
    result = answer_sub_question("What are the main steps in this workflow?", retriever)
    print(result["sub_question"])
    print(result["answer"])
    print("sources:", result["source_count"])
