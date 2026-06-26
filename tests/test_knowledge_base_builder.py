from pathlib import Path

from src.knowledge_base import create_knowledge_base_config
from src.knowledge_base_builder import is_supported_file, write_uploaded_files


def _test_kb(tmp_path: Path):
    return create_knowledge_base_config(
        name="Upload Test",
        root_dir=tmp_path,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        language_strategy="auto",
        parent_chunk_size=1000,
        parent_chunk_overlap=100,
        child_chunk_size=300,
        child_chunk_overlap=50,
        file_names=[],
    )


def test_write_uploaded_files_creates_raw_and_parsed_markdown(tmp_path):
    kb = _test_kb(tmp_path)

    saved = write_uploaded_files(kb, [("policy.txt", "员工福利包括医疗保险".encode("utf-8"))])

    assert saved == ["policy.txt"]
    assert (kb.raw_dir / "policy.txt").read_text(encoding="utf-8") == "员工福利包括医疗保险"
    parsed_files = list(kb.parsed_dir.glob("policy.*.md"))
    assert len(parsed_files) == 1
    parsed = parsed_files[0].read_text(encoding="utf-8")
    assert "file_name: policy.txt" in parsed
    assert "员工福利包括医疗保险" in parsed


def test_supported_upload_suffixes_include_existing_parser_formats():
    assert is_supported_file("handbook.md")
    assert is_supported_file("notes.txt")
    assert is_supported_file("policy.pdf")
    assert is_supported_file("slides.pptx")
    assert is_supported_file("sheet.xlsx")
    assert is_supported_file("page.html")
    assert is_supported_file("scan.png")
    assert not is_supported_file("archive.zip")
