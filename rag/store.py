"""Tiny local vector store: numpy matrix + cosine similarity, saved as .npy + .json."""
import json
from pathlib import Path

import numpy as np


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class VectorStore:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.vectors = np.empty((0, 0), dtype=np.float32)

    def add(self, chunks: list[dict], vectors) -> None:
        vecs = np.asarray(vectors, dtype=np.float32)
        if vecs.ndim != 2 or len(vecs) != len(chunks):
            raise ValueError("need exactly one vector per chunk")
        vecs = _normalize(vecs)
        self.vectors = vecs if not self.chunks else np.vstack([self.vectors, vecs])
        self.chunks.extend(chunks)

    def search(self, query_vector, k: int = 4) -> list[tuple[float, dict]]:
        """Top-k chunks by cosine similarity, best first, as (score, chunk)."""
        if not self.chunks:
            return []
        q = _normalize(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        scores = self.vectors @ q
        best = np.argsort(-scores)[:k]
        return [(float(scores[i]), self.chunks[i]) for i in best]

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        (d / "chunks.json").write_text(json.dumps(self.chunks, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        d = Path(directory)
        if not (d / "vectors.npy").exists() or not (d / "chunks.json").exists():
            raise SystemExit("No index found. Run first: python app.py ingest <file-or-folder>")
        store = cls()
        store.vectors = np.load(d / "vectors.npy")
        store.chunks = json.loads((d / "chunks.json").read_text(encoding="utf-8"))
        return store
