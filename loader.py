"""Read .txt, .md and .pdf files from a single file or a folder."""
from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def load_documents(path: str) -> list[dict]:
    """Return [{"source": name, "text": content}, ...]. Empty or unsupported files are skipped."""
    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(f for f in root.rglob("*") if f.is_file())
    else:
        return []

    docs = []
    for f in files:
        if f.suffix.lower() not in SUPPORTED:
            continue
        text = _read(f).strip()
        if text:
            source = f.name if root.is_file() else str(f.relative_to(root))
            docs.append({"source": source, "text": text})
    return docs
