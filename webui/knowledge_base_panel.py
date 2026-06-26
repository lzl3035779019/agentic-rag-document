from __future__ import annotations

import streamlit as st

from src.knowledge_base import (
    EMBEDDING_MODEL_OPTIONS,
    KnowledgeBaseConfig,
    create_knowledge_base_config,
    get_default_knowledge_base,
    list_knowledge_bases,
)
from src.knowledge_base_builder import SUPPORTED_SUFFIXES, build_and_register_knowledge_base


def _kb_label(kb: KnowledgeBaseConfig) -> str:
    suffix = "built-in" if kb.is_default else kb.embedding_model.rsplit("/", 1)[-1]
    return f"{kb.name} ({suffix})"


def _supported_upload_types() -> list[str]:
    return [suffix.lstrip(".") for suffix in sorted(SUPPORTED_SUFFIXES)]


def _selected_index(items: list[KnowledgeBaseConfig], kb_id: str | None) -> int:
    for index, item in enumerate(items):
        if item.kb_id == kb_id:
            return index
    return 0


def _render_current_kb(kb: KnowledgeBaseConfig) -> None:
    st.caption(f"Collection: `{kb.collection_name}`")
    st.caption(f"Embedding: `{kb.embedding_model}`")
    st.caption(
        "Chunks: "
        f"parent {kb.parent_chunk_size}/{kb.parent_chunk_overlap}, "
        f"child {kb.child_chunk_size}/{kb.child_chunk_overlap}"
    )
    st.caption(f"Language: `{kb.language_strategy}`")
    if kb.file_names:
        with st.expander(f"Files ({len(kb.file_names)})", expanded=False):
            for name in kb.file_names:
                st.markdown(f"- `{name}`")


def _render_auto_route_settings(items: list[KnowledgeBaseConfig], selected: KnowledgeBaseConfig) -> None:
    st.session_state.setdefault("auto_route_kb", False)
    st.session_state.setdefault("chinese_kb_id", selected.kb_id)
    st.session_state.setdefault("english_kb_id", selected.kb_id)

    auto_enabled = st.toggle(
        "自动按问题语言选择知识库",
        value=bool(st.session_state["auto_route_kb"]),
        help="中文问题使用中文默认知识库，英文问题使用英文默认知识库；关闭后使用当前知识库。",
    )
    st.session_state["auto_route_kb"] = auto_enabled

    if not auto_enabled:
        return

    zh_selected = st.selectbox(
        "中文问题默认知识库",
        options=items,
        index=_selected_index(items, st.session_state.get("chinese_kb_id")),
        format_func=_kb_label,
    )
    en_selected = st.selectbox(
        "英文问题默认知识库",
        options=items,
        index=_selected_index(items, st.session_state.get("english_kb_id")),
        format_func=_kb_label,
    )
    st.session_state["chinese_kb_id"] = zh_selected.kb_id
    st.session_state["english_kb_id"] = en_selected.kb_id
    st.caption(f"中文 -> `{zh_selected.name}`；英文 -> `{en_selected.name}`")


def render_knowledge_base_panel() -> KnowledgeBaseConfig:
    items = list_knowledge_bases()
    current_id = st.session_state.get("selected_kb_id")
    selected = st.selectbox(
        "当前知识库",
        options=items,
        index=_selected_index(items, current_id),
        format_func=_kb_label,
    )
    st.session_state["selected_kb_id"] = selected.kb_id
    _render_current_kb(selected)
    _render_auto_route_settings(items, selected)

    with st.expander("上传资料构建知识库", expanded=False):
        with st.form("create_knowledge_base"):
            name = st.text_input("知识库名称", value="My knowledge base")
            language = st.selectbox(
                "资料语言",
                options=["auto", "zh", "en"],
                format_func=lambda value: {
                    "auto": "自动判断",
                    "zh": "中文资料",
                    "en": "英文资料",
                }[value],
            )
            model_label = st.selectbox(
                "Embedding 模型",
                options=list(EMBEDDING_MODEL_OPTIONS.keys()),
                index=3,
            )
            col_a, col_b = st.columns(2)
            parent_size = col_a.number_input("Parent chunk size", 300, 8000, 1800, 100)
            parent_overlap = col_b.number_input("Parent overlap", 0, 2000, 180, 20)
            child_size = col_a.number_input("Child chunk size", 100, 3000, 420, 20)
            child_overlap = col_b.number_input("Child overlap", 0, 1000, 80, 10)
            files = st.file_uploader(
                "上传文档",
                type=_supported_upload_types(),
                accept_multiple_files=True,
                help="支持 md、txt、pdf、docx、pptx、xlsx、html、png、jpg、jpeg。",
            )
            submitted = st.form_submit_button("构建知识库", use_container_width=True)

        if submitted:
            if not files:
                st.warning("请先上传至少一个支持的文档文件。")
            else:
                kb = create_knowledge_base_config(
                    name=name,
                    embedding_model=EMBEDDING_MODEL_OPTIONS[model_label],
                    language_strategy=language,
                    parent_chunk_size=int(parent_size),
                    parent_chunk_overlap=int(parent_overlap),
                    child_chunk_size=int(child_size),
                    child_chunk_overlap=int(child_overlap),
                    file_names=[file.name for file in files],
                )
                payloads = [(file.name, file.getvalue()) for file in files]
                with st.spinner("正在切分、向量化并写入 Qdrant..."):
                    try:
                        built = build_and_register_knowledge_base(kb, payloads)
                    except Exception as exc:
                        st.error(f"构建失败：{type(exc).__name__}: {exc}")
                    else:
                        st.session_state["selected_kb_id"] = built.kb_id
                        if built.language_strategy == "zh":
                            st.session_state["chinese_kb_id"] = built.kb_id
                        elif built.language_strategy == "en":
                            st.session_state["english_kb_id"] = built.kb_id
                        st.success("知识库构建完成，已切换到新知识库。")
                        st.rerun()

    return selected or get_default_knowledge_base()
