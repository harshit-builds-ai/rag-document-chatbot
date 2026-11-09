"""Offline tests: no network, no API keys. LLM calls are replaced with fakes."""
import hashlib

import numpy as np
import pytest

import app
from rag import llm
from rag.chunking import chunk_text
from rag.loader import load_documents
from rag.store import VectorStore


# ---------- chunking ----------
def test_chunk_overlap_and_coverage():
    words = [f"w{i}" for i in range(100)]
    chunks = chunk_text(" ".join(words), size=30, overlap=10)
    assert chunks[0].split()[-10:] == chunks[1].split()[:10]  # overlap is kept
    assert chunks[-1].split()[-1] == "w99"  # nothing is dropped
    assert all(len(c.split()) <= 30 for c in chunks)


def test_chunk_short_and_empty_text():
    assert chunk_text("") == []
    assert chunk_text("just a few words", size=50, overlap=5) == ["just a few words"]


def test_chunk_rejects_bad_overlap():
    with pytest.raises(ValueError):
        chunk_text("a b c", size=5, overlap=5)


# ---------- loader ----------
def test_loader_reads_txt_md_and_skips_others(tmp_path):
    (tmp_path / "a.txt").write_text("alpha text", encoding="utf-8")
    (tmp_path / "b.md").write_text("# beta", encoding="utf-8")
    (tmp_path / "c.png").write_bytes(b"not text")
    (tmp_path / "empty.txt").write_text("   ", encoding="utf-8")
    docs = load_documents(str(tmp_path))
    assert {d["source"] for d in docs} == {"a.txt", "b.md"}


def test_loader_single_file_and_missing_path(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello", encoding="utf-8")
    assert load_documents(str(f))[0]["source"] == "note.txt"
    assert load_documents(str(tmp_path / "nope")) == []


# ---------- vector store ----------
def test_store_ranks_by_cosine_similarity():
    store = VectorStore()
    chunks = [{"source": "x", "id": i, "text": t} for i, t in enumerate(["north", "east", "north-east"])]
    store.add(chunks, [[1, 0], [0, 1], [1, 1]])
    hits = store.search([1, 0.1], k=2)
    assert hits[0][1]["text"] == "north"
    assert hits[0][0] >= hits[1][0]


def test_store_is_scale_invariant():
    store = VectorStore()
    store.add([{"text": "a"}, {"text": "b"}], [[10, 0], [0, 0.1]])
    assert store.search([0, 5], k=1)[0][1]["text"] == "b"


def test_store_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add([{"text": "a"}, {"text": "b"}], [[1, 0], [0, 1]])
    store.save(str(tmp_path / "idx"))
    loaded = VectorStore.load(str(tmp_path / "idx"))
    assert loaded.chunks == store.chunks
    assert np.allclose(loaded.vectors, store.vectors)


def test_store_rejects_mismatched_lengths_and_missing_index(tmp_path):
    with pytest.raises(ValueError):
        VectorStore().add([{"text": "a"}], [[1, 0], [0, 1]])
    with pytest.raises(SystemExit):
        VectorStore.load(str(tmp_path / "missing"))


# ---------- llm client (HTTP is faked) ----------
def test_gemini_embed_and_generate_parsing(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    seen = {}

    def fake_post(url, headers, payload):
        seen["url"], seen["payload"] = url, payload
        if "batchEmbedContents" in url:
            return {"embeddings": [{"values": [1.0, 2.0]} for _ in payload["requests"]]}
        return {"candidates": [{"content": {"parts": [{"text": " hi "}, {"text": "there"}]}}]}

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm.embed(["a", "b"], query=True) == [[1.0, 2.0], [1.0, 2.0]]
    assert seen["payload"]["requests"][0]["taskType"] == "RETRIEVAL_QUERY"
    assert llm.generate("q") == "hi there"


def test_openai_embed_keeps_input_order_and_generate(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fake_post(url, headers, payload):
        if url.endswith("/embeddings"):
            return {"data": [{"index": 1, "embedding": [2.0]}, {"index": 0, "embedding": [1.0]}]}
        return {"choices": [{"message": {"content": " answer "}}]}

    monkeypatch.setattr(llm, "_post", fake_post)
    assert llm.embed(["a", "b"]) == [[1.0], [2.0]]
    assert llm.generate("q") == "answer"


def test_embed_batches_large_inputs(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []

    def fake_post(url, headers, payload):
        calls.append(len(payload["input"]))
        return {"data": [{"index": i, "embedding": [1.0]} for i in range(len(payload["input"]))]}

    monkeypatch.setattr(llm, "_post", fake_post)
    assert len(llm.embed([str(i) for i in range(150)])) == 150
    assert calls == [64, 64, 22]


def test_missing_key_and_bad_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "")
    with pytest.raises(SystemExit):
        llm.embed(["x"])
    monkeypatch.setenv("LLM_PROVIDER", "other")
    with pytest.raises(SystemExit):
        llm.generate("x")


# ---------- end to end (fake embeddings + fake LLM) ----------
def fake_embed(texts, query=False):
    """Deterministic bag-of-words hashing vectors, good enough to test retrieval."""
    out = []
    for t in texts:
        v = np.zeros(64)
        for w in t.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        out.append(v.tolist())
    return out


def test_ingest_then_ask_uses_retrieved_context_and_sources(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "leave.txt").write_text(
        "Employees get 24 days of paid leave every year.", encoding="utf-8"
    )
    (tmp_path / "docs" / "remote.txt").write_text(
        "Remote work is allowed two days per week.", encoding="utf-8"
    )
    monkeypatch.setattr(app, "INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setattr(llm, "embed", fake_embed)
    prompts = []
    monkeypatch.setattr(llm, "generate", lambda p: prompts.append(p) or "ok")

    app.ingest(str(tmp_path / "docs"))
    assert app.ask("how many days of paid leave", k=1) == "ok"
    assert "[leave.txt #0]" in prompts[0]
    assert "24 days" in prompts[0]
    assert "remote" not in prompts[0].lower()
    assert "using only the CONTEXT" in prompts[0]  # grounding instruction is sent


def test_ingest_with_no_documents_exits(tmp_path):
    with pytest.raises(SystemExit):
        app.ingest(str(tmp_path))
