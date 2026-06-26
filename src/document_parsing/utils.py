import hashlib
import re
from pathlib import Path


# 用文件内容 hash 判断文件是否变化，支持后面的增量解析。
def compute_file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 把文件名清理成适合落盘的安全名字，避免空格、中文符号、特殊字符造成路径问题。
def safe_filename(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", value)
    return value.strip("_") or "document"