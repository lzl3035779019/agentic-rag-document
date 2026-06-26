from langchain_core.documents import Document

from src.graph import _source_reference, _source_snippet, _source_snippets


def test_source_reference_uses_expanded_parent_ids_when_present():
    doc = Document(
        page_content="content",
        metadata={
            "source": "doc.md",
            "parent_id": "parent-1",
            "expanded_parent_ids": ["parent-1", "parent-2"],
        },
    )

    assert _source_reference(doc) == "doc.md#parent-1,parent-2"


def test_source_snippet_normalizes_and_truncates_text():
    doc = Document(
        page_content="第一段。\n\n第二段内容很多。" * 100,
        metadata={"source": "doc.md", "parent_id": "parent-1"},
    )

    snippet = _source_snippet(doc, max_chars=40)

    assert "\n\n" not in snippet
    assert snippet.endswith("...")
    assert len(snippet) <= 43


def test_source_snippets_deduplicates_by_reference():
    docs = [
        Document("same", metadata={"source": "doc.md", "parent_id": "parent-1"}),
        Document("same again", metadata={"source": "doc.md", "parent_id": "parent-1"}),
    ]

    snippets = _source_snippets(docs)

    assert snippets == [{"source": "doc.md#parent-1", "snippet": "same"}]
