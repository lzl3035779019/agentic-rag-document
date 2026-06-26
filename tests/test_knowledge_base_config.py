from pathlib import Path

from src.knowledge_base import (
    DEFAULT_KB_ID,
    KnowledgeBaseConfig,
    create_knowledge_base_config,
    get_default_knowledge_base,
)


def test_default_knowledge_base_matches_existing_collection():
    kb = get_default_knowledge_base()

    assert kb.kb_id == DEFAULT_KB_ID
    assert kb.collection_name == "basecamp_handbook_visualized"
    assert kb.parsed_dir.name == "parsed_docs"


def test_create_knowledge_base_config_preserves_runtime_parameters(tmp_path: Path):
    kb = create_knowledge_base_config(
        name="中文 HR 手册",
        root_dir=tmp_path,
        embedding_model="BAAI/bge-small-zh-v1.5",
        language_strategy="zh",
        parent_chunk_size=1800,
        parent_chunk_overlap=180,
        child_chunk_size=420,
        child_chunk_overlap=80,
        file_names=["policy.md"],
    )

    assert isinstance(kb, KnowledgeBaseConfig)
    assert kb.kb_id.startswith("zhong-wen-hr-shou-ce")
    assert kb.collection_name.startswith("kb_zhong_wen_hr_shou_ce")
    assert kb.embedding_model == "BAAI/bge-small-zh-v1.5"
    assert kb.language_strategy == "zh"
    assert kb.parent_chunk_size == 1800
    assert kb.child_chunk_overlap == 80
    assert kb.raw_dir == tmp_path / kb.kb_id / "raw"
    assert kb.parsed_dir == tmp_path / kb.kb_id / "parsed"
    assert kb.file_names == ["policy.md"]
