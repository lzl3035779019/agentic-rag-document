from dataclasses import dataclass
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHILD_CHUNK_OVERLAP,
    CHILD_CHUNK_SIZE,
    PARENT_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE,
)
from src.knowledge_base import KnowledgeBaseConfig, get_default_knowledge_base
from src.loader import load_documents


HEADING_ONLY_MAX_CHARS = 120


@dataclass(frozen=True)
class ChunkingConfig:
    language_strategy: str = "en"
    parent_chunk_size: int = PARENT_CHUNK_SIZE
    parent_chunk_overlap: int = PARENT_CHUNK_OVERLAP
    child_chunk_size: int = CHILD_CHUNK_SIZE
    child_chunk_overlap: int = CHILD_CHUNK_OVERLAP


def chunking_config_from_kb(kb_config: KnowledgeBaseConfig | None = None) -> ChunkingConfig:
    kb = kb_config or get_default_knowledge_base()
    return ChunkingConfig(
        language_strategy=kb.language_strategy,
        parent_chunk_size=kb.parent_chunk_size,
        parent_chunk_overlap=kb.parent_chunk_overlap,
        child_chunk_size=kb.child_chunk_size,
        child_chunk_overlap=kb.child_chunk_overlap,
    )


def get_language_separators(language_strategy: str) -> list[str]:
    if language_strategy == "zh":
        return ["\n## ", "\n### ", "\n\n", "\n", "。", "？", "！", "；", "，", "、", " "]
    return ["\n## ", "\n### ", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def resolve_language_strategy(docs: list[Document], language_strategy: str) -> str:
    if language_strategy != "auto":
        return language_strategy
    text = "\n".join(doc.page_content[:2000] for doc in docs)
    cjk_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    ascii_word_count = sum(1 for char in text if char.isascii() and char.isalpha())
    return "zh" if cjk_count >= max(5, ascii_word_count // 3) else "en"


def _is_heading_only(doc: Document) -> bool:
    content = doc.page_content.strip()
    if len(content) > HEADING_ONLY_MAX_CHARS or not content.startswith("#"):
        return False
    text_without_marks = content.lstrip("#").strip()
    return len(text_without_marks.split()) <= 8


def _same_source(left: Document, right: Document) -> bool:
    return left.metadata.get("source") == right.metadata.get("source")


def _merge_heading_only_parents(parent_docs: list[Document]) -> list[Document]:
    merged: list[Document] = []
    index = 0

    while index < len(parent_docs):
        doc = parent_docs[index]
        if not _is_heading_only(doc):
            merged.append(doc)
            index += 1
            continue

        parts = [doc.page_content.strip()]
        metadata = dict(doc.metadata)
        next_index = index + 1

        while (
            next_index < len(parent_docs)
            and _is_heading_only(parent_docs[next_index])
            and _same_source(doc, parent_docs[next_index])
        ):
            parts.append(parent_docs[next_index].page_content.strip())
            next_index += 1

        if next_index < len(parent_docs) and _same_source(doc, parent_docs[next_index]):
            parts.append(parent_docs[next_index].page_content.strip())
            merged.append(Document(page_content="\n\n".join(parts), metadata=metadata))
            index = next_index + 1
            continue

        if merged and merged[-1].metadata.get("source") == doc.metadata.get("source"):
            merged[-1].page_content = (
                merged[-1].page_content.rstrip()
                + "\n\n"
                + "\n\n".join(parts)
            )
        else:
            merged.append(doc)
        index = next_index

    return merged


def build_parent_child_chunks_from_documents(
    docs: list[Document],
    config: ChunkingConfig,
) -> tuple[list[Document], dict[str, Document]]:
    language_strategy = resolve_language_strategy(docs, config.language_strategy)
    separators = get_language_separators(language_strategy)

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.parent_chunk_size,
        chunk_overlap=config.parent_chunk_overlap,
        separators=separators,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.child_chunk_size,
        chunk_overlap=config.child_chunk_overlap,
        separators=separators,
    )

    parent_docs = _merge_heading_only_parents(parent_splitter.split_documents(docs))
    parent_map: dict[str, Document] = {}
    child_docs: list[Document] = []

    for parent_index, parent_doc in enumerate(parent_docs):
        parent_id = f"parent-{parent_index}"
        parent_doc.metadata["parent_id"] = parent_id
        parent_map[parent_id] = parent_doc

        children = child_splitter.split_documents([parent_doc])
        for child_index, child_doc in enumerate(children):
            child_doc.metadata["parent_id"] = parent_id
            child_doc.metadata["child_id"] = f"{parent_id}-child-{child_index}"
            child_doc.metadata["source"] = parent_doc.metadata.get("source")
            child_docs.append(child_doc)

    return child_docs, parent_map


@lru_cache(maxsize=16)
def _build_parent_child_chunks_cached(
    parsed_dir: str,
    language_strategy: str,
    parent_chunk_size: int,
    parent_chunk_overlap: int,
    child_chunk_size: int,
    child_chunk_overlap: int,
) -> tuple[list[Document], dict[str, Document]]:
    docs = load_documents(parsed_dir)
    config = ChunkingConfig(
        language_strategy=language_strategy,
        parent_chunk_size=parent_chunk_size,
        parent_chunk_overlap=parent_chunk_overlap,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
    )
    return build_parent_child_chunks_from_documents(docs, config)


def build_parent_child_chunks(
    kb_config: KnowledgeBaseConfig | None = None,
) -> tuple[list[Document], dict[str, Document]]:
    kb = kb_config or get_default_knowledge_base()
    return _build_parent_child_chunks_cached(
        str(kb.parsed_dir),
        kb.language_strategy,
        kb.parent_chunk_size,
        kb.parent_chunk_overlap,
        kb.child_chunk_size,
        kb.child_chunk_overlap,
    )


if __name__ == "__main__":
    child_docs, parent_map = build_parent_child_chunks()

    print(f"child chunks: {len(child_docs)}")
    print(f"parent chunks: {len(parent_map)}")

    first_child = child_docs[0]
    for i, (key, value) in enumerate(parent_map.items()):
        if i >= 3: break
        print(key)
        print("###########")
        print(value)
    print("*************")
    print(first_child.metadata)
    print(first_child.page_content[:500])
