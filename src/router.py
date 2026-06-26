import json
import re

from langchain_core.prompts import ChatPromptTemplate

from src.llm_cache import invoke_llm_cached


COMPLEX_MARKERS = [
    "compare",
    "difference",
    "differences",
    "differ",
    "relationship",
    "connect",
    "connection",
    "both",
    "across",
    "versus",
    " vs ",
    "summarize all",
    "which items",
    "which documents",
    "which tools",
    "which rules apply",
    "which sections discuss",
    "multiple",
    "together",
    "coexist",
    "trade off",
    "tradeoff",
    "immediately and which require",
    "boundary conditions",
    " and why ",
    "allowed and not allowed",
    "ok and not ok",
    "pros and cons",
]

CHINESE_COMPLEX_MARKERS = [
    "比较",
    "对比",
    "相比",
    "区别",
    "差异",
    "不同",
    "关系",
    "联系",
    "影响",
    "分别",
    "同时",
    "多个",
    "多种",
    "哪些方面",
    "优缺点",
    "利弊",
    "取舍",
    "权衡",
    "适合什么场景",
    "适用场景",
    "各自",
]

SCENARIO_COMPLEX_MARKERS = [
    "what should i know before",
    "what should someone know before",
    "what happens when someone",
    "what should i consider before",
    "what should someone consider before",
    "what should i do if",
    "what happens if",
    "what are the steps for",
    "how should i handle",
    "how should we handle",
    "what should be considered",
]

CHINESE_SCENARIO_COMPLEX_MARKERS = [
    "应该了解什么",
    "需要了解什么",
    "需要知道什么",
    "应该考虑什么",
    "需要考虑什么",
    "应该注意什么",
    "需要注意什么",
    "如果",
    "发生",
    "怎么处理",
    "如何处理",
    "处理流程",
]

COMPLEX_PATTERNS = [
    r"\bfrom\s+[a-z0-9-]+\s+to\s+[a-z0-9-]+\b",
    r"\bchange\s+from\b",
    r"\bchanges\s+from\b",
    r"\bprogress\s+from\b",
    r"\bprogresses\s+from\b",
    r"\bprogression\s+from\b",
    r"\bwhich\s+.+\s+apply\b",
]

SIMPLE_PREFIXES = [
    "what is",
    "what are",
    "when",
    "where",
    "who",
    "how much",
    "how many",
    "how does",
    "does",
    "is",
    "tell me about",
    "explain",
]

CHINESE_SIMPLE_MARKERS = [
    "是什么",
    "有哪些",
    "有什么",
    "是多少",
    "什么时候",
    "在哪里",
    "谁",
    "如何",
    "怎么",
    "怎样",
    "是否",
    "能否",
    "可以吗",
]

SIMPLE_SINGLE_TOPIC_PATTERNS = [
    "how does",
    "how do",
    "tell me about",
    "what should i understand about",
    "what should we understand about",
    "explain",
]

ROUTER_PROMPT = ChatPromptTemplate.from_template(
    """
You are a query router for a general enterprise knowledge base RAG system.

Classify the user question into one route:

simple:
- answerable with one focused retrieval query
- asks for one fact, one policy, one definition, one number, one process, or one document section
- may be broad if it is still about one named feature, rule, process, error, configuration, metric, or document section

complex:
- requires comparing multiple policies
- asks about relationships across topics
- needs synthesis from multiple documents
- contains multiple user intents
- asks broadly what someone should know about a situation that likely spans several independent topics or workflows
- asks about a lifecycle scenario, organizational culture, or support system that may involve several steps, teams, or policies

Routing preference:
- If the question is broad but centered on one topic, prefer simple.
- If the question lists several items separated by commas or asks about a broad scenario, prefer complex.
- Choose complex only when the question clearly needs decomposition into multiple retrieval sub-questions.

Return JSON only.

JSON format:
{{"route": "simple" | "complex", "reason": "..."}}

User question:
{question}
"""
)


def _contains_chinese(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _chinese_char_count(text: str) -> int:
    return sum(1 for char in text if "\u4e00" <= char <= "\u9fff")


def rule_route(question: str) -> dict:
    q = f" {question.lower().strip()} "
    stripped = q.strip()
    word_count = len(question.split())
    has_chinese = _contains_chinese(question)

    if question.count("?") >= 2:
        return {
            "route": "complex",
            "reason": "Rule matched: multiple question marks indicate multiple intents.",
            "router": "rule",
        }

    if has_chinese and question.count("？") >= 2:
        return {
            "route": "complex",
            "reason": "Rule matched: multiple Chinese question marks indicate multiple intents.",
            "router": "rule",
        }

    if any(marker in q for marker in COMPLEX_MARKERS):
        return {
            "route": "complex",
            "reason": "Rule matched: question contains comparison, synthesis, or multi-topic marker.",
            "router": "rule",
        }

    if has_chinese and any(marker in stripped for marker in CHINESE_COMPLEX_MARKERS):
        return {
            "route": "complex",
            "reason": "Rule matched: Chinese comparison, synthesis, or multi-topic marker.",
            "router": "rule",
        }

    if any(re.search(pattern, stripped) for pattern in COMPLEX_PATTERNS):
        return {
            "route": "complex",
            "reason": "Rule matched: question asks about a range, progression, or program set.",
            "router": "rule",
        }

    if has_chinese and any(marker in stripped for marker in CHINESE_SCENARIO_COMPLEX_MARKERS):
        return {
            "route": "complex",
            "reason": "Rule matched: Chinese broad scenario likely spans multiple steps, teams, or policies.",
            "router": "rule",
        }

    if any(marker in q for marker in SCENARIO_COMPLEX_MARKERS):
        return {
            "route": "complex",
            "reason": "Rule matched: broad scenario likely spans multiple steps, teams, or policies.",
            "router": "rule",
        }

    if has_chinese and (question.count("，") + question.count("、") >= 2):
        return {
            "route": "complex",
            "reason": "Rule matched: Chinese list punctuation suggests multiple retrieval targets.",
            "router": "rule",
        }

    if stripped.startswith(("when are", "when is", "when do", "when does")):
        return {
            "route": "simple",
            "reason": "Rule matched: one timing or eligibility question.",
            "router": "rule",
        }

    if question.count(",") >= 2:
        return {
            "route": "complex",
            "reason": "Rule matched: comma-separated list suggests multiple retrieval targets.",
            "router": "rule",
        }

    if has_chinese and any(marker in stripped for marker in CHINESE_SIMPLE_MARKERS) and _chinese_char_count(question) <= 18:
        return {
            "route": "simple",
            "reason": "Rule matched: short Chinese single-topic factual or policy question.",
            "router": "rule",
        }

    if any(stripped.startswith(pattern) for pattern in SIMPLE_SINGLE_TOPIC_PATTERNS) and word_count <= 10:
        return {
            "route": "simple",
            "reason": "Rule matched: broad wording but appears to ask about one topic.",
            "router": "rule",
        }

    if any(stripped.startswith(prefix) for prefix in SIMPLE_PREFIXES) and word_count <= 16:
        return {
            "route": "simple",
            "reason": "Rule matched: short single-intent factual or policy question.",
            "router": "rule",
        }

    return {
        "route": "uncertain",
        "reason": "Rule could not confidently classify the question.",
        "router": "rule",
    }


def llm_route(question: str) -> dict:
    prompt = ROUTER_PROMPT.invoke({"question": question})
    text = invoke_llm_cached("route_question", prompt).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "route": "simple",
            "reason": "LLM router returned invalid JSON, defaulting to simple.",
            "router": "llm_fallback",
        }

    route = str(data.get("route", "")).strip().lower()
    if route not in {"simple", "complex"}:
        route = "simple"

    return {
        "route": route,
        "reason": str(data.get("reason", "")).strip() or "Classified by LLM router.",
        "router": "llm",
    }


def route_question(question: str) -> dict:
    result = rule_route(question)
    if result["route"] != "uncertain":
        return result
    return llm_route(question)


if __name__ == "__main__":
    examples = [
        "What is the retry policy?",
        "Compare synchronous and asynchronous processing.",
        "What should I know before changing this configuration?",
    ]
    for example in examples:
        print("=" * 60)
        print("question:", example)
        print(route_question(example))
