from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


CLARIFICATION_PROMPT_EN = ChatPromptTemplate.from_template(
    """
You are a clarification checker for a general enterprise knowledge base RAG system.

Decide whether the user question is too ambiguous to retrieve documents.

Return JSON only:
{{"need_clarification": true or false, "question": "clarifying question if needed"}}

Rules:
1. Return true only when the question cannot be searched because it depends on missing context.
2. Short questions like "How does this work?", "What about that?", or "Explain it" usually need clarification.
3. Specific questions with clear entities, policies, tools, dates, numbers, or actions do not need clarification.

User question:
{question}
"""
)

CLARIFICATION_PROMPT_ZH = ChatPromptTemplate.from_template(
    """
你是企业知识库 RAG 系统的澄清判断器。

判断用户问题是否含糊到无法检索文档。

只返回 JSON：
{{"need_clarification": true or false, "question": "如果需要澄清，写一个中文澄清问题"}}

规则：
1. 只有当问题依赖缺失上下文、无法形成检索词时，才返回 true。
2. 像“这个怎么处理？”、“它是什么？”、“解释一下”这类缺少明确对象的问题通常需要澄清。
3. 如果问题里有清楚的实体、政策、工具、日期、数字、动作或主题，不需要澄清。
4. 如果用户问题是中文，澄清问题必须用中文。

用户问题：
{question}
"""
)

CLARIFICATION_PROMPT = CLARIFICATION_PROMPT_EN


AMBIGUOUS_REFERENCES = {"this", "that", "it", "they", "them", "those", "these"}
CHINESE_AMBIGUOUS_REFERENCES = {
    "这个",
    "那个",
    "这些",
    "那些",
    "它",
    "他们",
    "她们",
    "它们",
    "这",
    "那",
}
CHINESE_QUESTION_MARKERS = (
    "什么",
    "哪些",
    "哪",
    "怎么",
    "怎样",
    "如何",
    "多少",
    "是否",
    "有没有",
    "有什么",
)
CLEAR_QUESTION_STARTS = (
    "what is ",
    "what are ",
    "who is ",
    "who are ",
    "when is ",
    "when are ",
    "where is ",
    "where are ",
    "how much ",
    "how many ",
    "how does ",
    "how do ",
    "does ",
    "is ",
    "can ",
)
VAGUE_SHORT_QUESTIONS = {
    "help",
    "explain",
    "explain it",
    "what about",
    "what about this",
    "what about that",
    "how so",
}


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _select_clarification_prompt(question: str) -> ChatPromptTemplate:
    return CLARIFICATION_PROMPT_ZH if _contains_chinese(question) else CLARIFICATION_PROMPT_EN


def _needs_chinese_clarification(question: str) -> bool:
    compact = "".join(
        char for char in question.strip().lower()
        if char not in " \t\r\n，。！？?；;：:、,.()（）[]【】{}"
    )
    if not compact:
        return True

    if compact in CHINESE_AMBIGUOUS_REFERENCES:
        return True

    if any(compact.startswith(ref) and len(compact) <= len(ref) + 2 for ref in CHINESE_AMBIGUOUS_REFERENCES):
        return True

    if any(marker in compact for marker in CHINESE_QUESTION_MARKERS) and len(compact) >= 5:
        return False

    return len(compact) <= 3


def _needs_llm_clarification(question: str) -> bool:
    normalized = question.strip().lower()
    words = [word.strip(".,?!;:'\"()[]{}") for word in normalized.split()]

    if _contains_chinese(question):
        return _needs_chinese_clarification(question)

    if any(word in AMBIGUOUS_REFERENCES for word in words):
        return True

    if normalized.startswith(CLEAR_QUESTION_STARTS):
        return False

    if normalized in VAGUE_SHORT_QUESTIONS:
        return True

    if len(words) <= 2:
        return True

    return False


def check_clarification(question: str) -> dict:
    if not _needs_llm_clarification(question):
        return {
            "need_clarification": False,
            "raw": "Skipped LLM clarification by rule: question is specific enough.",
        }

    prompt = _select_clarification_prompt(question).invoke({"question": question})
    text = invoke_llm_cached("check_clarification", prompt).strip()
    need = "true" in text.lower()
    return {
        "need_clarification": need,
        "raw": text,
    }


if __name__ == "__main__":
    for question in ["How does this work?", "What are the main steps in this workflow?"]:
        result = check_clarification(question)
        print("=" * 40)
        print("question:", question)
        print(result)
