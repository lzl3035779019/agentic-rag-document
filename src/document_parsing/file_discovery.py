from pathlib import Path

from src.config import PROJECT_ROOT

SUPPORTED_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".htm",
    ".png",
    ".jpg",
    ".jpeg",
}


# 只发现白名单格式，避免把 .zip/.exe/.tmp 等无关文件送进解析链路。
def discover_files(raw_docs_dir: Path) -> list[Path]:
    files = []

    for path in raw_docs_dir.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        files.append(path)

    return sorted(files)


if __name__ == "__main__":
    path = PROJECT_ROOT / "data/raw_docs"
    raw_docs_dir = Path(path)

    for file_path in discover_files(raw_docs_dir):
        print(file_path)