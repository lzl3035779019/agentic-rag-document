import json

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.basic_rag import format_docs
from src.llm_cache import invoke_llm_cached, invoke_llm_cached_stream


EVIDENCE_PROMPT_EN = ChatPromptTemplate.from_template(
    """
You are an evidence extraction module for a general enterprise knowledge base RAG system.

Extract only facts from the context that directly help answer the user question.

Rules:
1. Use only the provided context.
2. Do not infer, complete, or add facts.
3. Keep exact numbers, dates, policy names, tool names, and document names when they appear.
4. If a needed part is missing, list it in missing_information.
5. Return JSON only.

JSON schema:
{{
  "facts": ["fact 1", "fact 2"],
  "missing_information": ["missing item 1"]
}}

Question:
{question}

Context:
{context}
"""
)


EVIDENCE_PROMPT_ZH = ChatPromptTemplate.from_template(
    """
你是企业知识库 RAG 系统的证据抽取模块。

请只从 Context 中抽取能直接回答用户问题的事实。

规则：
1. 只能使用给定 Context。
2. 不要推断、补全或添加事实。
3. 保留 Context 中出现的准确数字、日期、政策名、工具名和文档名。
4. 如果缺少回答所需的重要信息，写入 missing_information。
5. 只返回 JSON。

JSON schema:
{{
  "facts": ["直接由 Context 支持的事实"],
  "missing_information": ["缺少的重要信息"]
}}

Question:
{question}

Context:
{context}
"""
)

EVIDENCE_PROMPT = EVIDENCE_PROMPT_EN


EVIDENCE_ANSWER_PROMPT_EN = ChatPromptTemplate.from_template(
    """
You are an enterprise knowledge base assistant.
Answer the question using only the extracted evidence.

Rules:
1. If the user asks in Chinese, answer in Chinese. Otherwise answer in English.
2. Use only facts listed in Evidence.
3. Do not use general knowledge.
4. Do not mention any policy, tool, date, number, department, person, or document unless it appears in Evidence.
5. If Missing information contains important gaps, clearly state what cannot be answered from the current knowledge base results.
6. Keep the answer concise.

Question:
{question}

Evidence:
{facts}

Missing information:
{missing_information}

Answer:
"""
)


EVIDENCE_ANSWER_PROMPT_ZH = ChatPromptTemplate.from_template(
    """
你是企业知识库问答助手。
请只依据抽取出的 Evidence 回答问题。

规则：
1. 必须用中文回答。
2. 只能使用 Evidence 中列出的事实。
3. 不要使用外部常识。
4. 不要提到 Evidence 中没有出现的政策、工具、日期、数字、部门、人物或文档。
5. 如果 Missing information 中存在重要缺口，需要清楚说明当前知识库结果无法回答什么。
6. 回答保持清晰、简洁。

Question:
{question}

Evidence:
{facts}

Missing information:
{missing_information}

Answer:
"""
)

EVIDENCE_ANSWER_PROMPT = EVIDENCE_ANSWER_PROMPT_EN


ANSWER_WITH_EVIDENCE_PROMPT_EN = ChatPromptTemplate.from_template(
    """
You are an enterprise knowledge base assistant.

Answer the question and extract supporting evidence in one pass.

Rules:
1. Use only the provided context.
2. Do not infer, invent, or complete policies, numbers, dates, tools, departments, people, or document names.
3. Keep exact numbers, dates, policy names, tool names, and document names when they appear in context.
4. If the context only partially answers the question, answer the supported part and list the missing information.
5. If the user asks in Chinese, answer in Chinese. Otherwise answer in English.
6. Return JSON only.

JSON schema:
{{
  "facts": ["fact directly supported by context"],
  "missing_information": ["important missing item"],
  "answer": "final user-facing answer"
}}

Question:
{question}

Context:
{context}
"""
)


ANSWER_WITH_EVIDENCE_PROMPT_ZH = ChatPromptTemplate.from_template(
    """
你是企业知识库问答助手。

请基于 Context 一次性完成证据抽取和回答。

规则：
1. 只能使用给定 Context。
2. 不要推断、编造或补全政策、数字、日期、工具、部门、人物或文档名。
3. 保留 Context 中出现的准确数字、日期、政策名、工具名和文档名。
4. 如果 Context 只能部分回答问题，先回答有依据的部分，再列出缺少的信息。
5. 必须用中文回答。
6. 只返回 JSON。

JSON schema:
{{
  "facts": ["直接由 Context 支持的事实"],
  "missing_information": ["缺少的重要信息"],
  "answer": "面向用户的最终中文回答"
}}

Question:
{question}

Context:
{context}
"""
)

ANSWER_WITH_EVIDENCE_PROMPT = ANSWER_WITH_EVIDENCE_PROMPT_EN


FAST_ANSWER_CONTEXT_CHAR_LIMIT = 12000


FAST_ANSWER_PROMPT_ZH = ChatPromptTemplate.from_template(
    """
你是企业知识库问答助手。

请只依据给定 Context 回答用户问题。

规则：
1. 必须用中文回答。
2. 默认采用“中等详细”风格：先用 1-2 句话直接概括，再分点展开。
3. 不要只列标题或名词；如果 Context 中有解释、流程、优缺点、适用场景、代表实现或示例，要保留关键内容。
4. 每个主要要点通常写 1-3 句，避免写成论文，也不要过度压缩。
5. 保留 Context 中出现的准确数字、日期、政策名、工具名、文档名和专有名词。
6. 不要使用外部常识，不要补充 Context 没有的信息。
7. 如果 Context 只能部分回答，先回答有依据的部分，再简短说明缺少哪些信息。
8. 只有当 Context 对问题没有任何有用事实时，才说无法找到足够信息。
9. 使用清晰 Markdown。解释型问题可以用小标题或加粗要点；列表型问题用项目符号。
10. 不要在正文里输出原始文件路径，来源会由系统单独追加。

Question:
{question}

Context:
{context}

Answer:
"""
)


FAST_ANSWER_PROMPT_EN = ChatPromptTemplate.from_template(
    """
You are an enterprise knowledge base assistant.

Answer the question using only the provided context.

Rules:
1. Answer in English.
2. Use a concise but complete structure: start with the direct answer, then use bullets or short sections.
3. Do not only list names or headings. Preserve key explanations, steps, tradeoffs, examples, constraints, and use cases when they appear in the context.
4. Keep exact numbers, dates, policy names, tool names, and document names when they appear in context.
5. Do not use general knowledge or add information that is not supported by the context.
6. If the context partially answers the question, answer the supported part first, then briefly state what is missing.
7. Say there is not enough information only when the context provides no useful facts for the question.
8. Format the answer in clean Markdown.
9. Do not include raw file paths in the answer body; source paths are appended separately.

Question:
{question}

Context:
{context}

Answer:
"""
)


def _parse_evidence_json(text: str) -> dict[str, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "facts": [],
            "missing_information": [f"Evidence extractor returned invalid JSON: {text[:300]}"],
        }

    facts = data.get("facts", [])
    missing = data.get("missing_information", [])
    if not isinstance(facts, list):
        facts = []
    if not isinstance(missing, list):
        missing = []

    return {
        "facts": [str(item).strip() for item in facts if str(item).strip()],
        "missing_information": [str(item).strip() for item in missing if str(item).strip()],
    }


def _parse_answer_with_evidence_json(text: str) -> dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "facts": [],
            "missing_information": [f"Answer-with-evidence returned invalid JSON: {text[:300]}"],
            "answer": "",
        }

    evidence = _parse_evidence_json(text)
    answer = str(data.get("answer", "")).strip()
    return {
        "facts": evidence["facts"],
        "missing_information": evidence["missing_information"],
        "answer": answer,
    }


def extract_evidence(question: str, docs: list[Document]) -> dict[str, list[str]]:
    prompt = _select_evidence_prompt(question).invoke(
        {
            "question": question,
            "context": format_docs(docs)[:6000],
        }
    )
    return _parse_evidence_json(invoke_llm_cached("extract_evidence", prompt))


def answer_from_evidence(question: str, evidence: dict[str, list[str]]) -> str:
    facts = evidence.get("facts", [])
    missing = evidence.get("missing_information", [])
    prompt = _select_evidence_answer_prompt(question).invoke(
        {
            "question": question,
            "facts": "\n".join(f"- {fact}" for fact in facts) or "No directly supported facts were extracted.",
            "missing_information": "\n".join(f"- {item}" for item in missing) or "None.",
        }
    )
    return invoke_llm_cached("answer_from_evidence", prompt)


def answer_from_evidence_stream(question: str, evidence: dict[str, list[str]]):
    facts = evidence.get("facts", [])
    missing = evidence.get("missing_information", [])
    prompt = _select_evidence_answer_prompt(question).invoke(
        {
            "question": question,
            "facts": "\n".join(f"- {fact}" for fact in facts) or "No directly supported facts were extracted.",
            "missing_information": "\n".join(f"- {item}" for item in missing) or "None.",
        }
    )
    yield from invoke_llm_cached_stream("answer_from_evidence", prompt)


def answer_with_evidence(question: str, docs: list[Document]) -> dict:
    prompt = _select_answer_with_evidence_prompt(question).invoke(
        {
            "question": question,
            "context": format_docs(docs)[:6000],
        }
    )
    return _parse_answer_with_evidence_json(invoke_llm_cached("answer_with_evidence", prompt))


def _prefers_chinese(question: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in question)


def _select_evidence_prompt(question: str) -> ChatPromptTemplate:
    return EVIDENCE_PROMPT_ZH if _prefers_chinese(question) else EVIDENCE_PROMPT_EN


def _select_evidence_answer_prompt(question: str) -> ChatPromptTemplate:
    return EVIDENCE_ANSWER_PROMPT_ZH if _prefers_chinese(question) else EVIDENCE_ANSWER_PROMPT_EN


def _select_answer_with_evidence_prompt(question: str) -> ChatPromptTemplate:
    return ANSWER_WITH_EVIDENCE_PROMPT_ZH if _prefers_chinese(question) else ANSWER_WITH_EVIDENCE_PROMPT_EN


def _select_fast_answer_prompt(question: str) -> ChatPromptTemplate:
    return FAST_ANSWER_PROMPT_ZH if _prefers_chinese(question) else FAST_ANSWER_PROMPT_EN


def fast_answer(question: str, docs: list[Document]) -> str:
    prompt_template = _select_fast_answer_prompt(question)
    prompt = prompt_template.invoke(
        {
            "question": question,
            "context": format_docs(docs)[:FAST_ANSWER_CONTEXT_CHAR_LIMIT],
        }
    )
    return invoke_llm_cached("fast_answer", prompt).strip()


def fast_answer_stream(question: str, docs: list[Document]):
    prompt_template = _select_fast_answer_prompt(question)
    prompt = prompt_template.invoke(
        {
            "question": question,
            "context": format_docs(docs)[:FAST_ANSWER_CONTEXT_CHAR_LIMIT],
        }
    )
    yield from invoke_llm_cached_stream("fast_answer", prompt)
