from langchain_core.documents import Document

from src.parent_child_splitter import (
    ChunkingConfig,
    build_parent_child_chunks_from_documents,
    get_language_separators,
)


def test_chinese_strategy_uses_chinese_punctuation_before_spaces():
    separators = get_language_separators("zh")

    assert "。" in separators
    assert "？" in separators
    assert "！" in separators
    assert separators.index("。") < separators.index(" ")


def test_chunking_config_controls_parent_and_child_sizes():
    docs = [
        Document(
            page_content="第一段说明公司福利。第二段说明医疗保险。第三段说明年假和病假。",
            metadata={"source": "policy.md"},
        )
    ]
    config = ChunkingConfig(
        language_strategy="zh",
        parent_chunk_size=18,
        parent_chunk_overlap=0,
        child_chunk_size=8,
        child_chunk_overlap=0,
    )

    child_docs, parent_map = build_parent_child_chunks_from_documents(docs, config)

    assert len(parent_map) > 1
    assert len(child_docs) >= len(parent_map)
    assert all(len(doc.page_content) <= 18 for doc in parent_map.values())
    assert all(doc.metadata["source"] == "policy.md" for doc in child_docs)
