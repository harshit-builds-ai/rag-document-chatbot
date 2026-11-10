"""Offline tests (no API key): irrelevant questions are refused before the LLM is called."""
import app
from rag.store import VectorStore


def _setup(tmp_path, monkeypatch, query_vector):
    store = VectorStore()
    store.add([{"source": "policy.md", "id": 0, "text": "24 days of paid leave"}], [[1.0, 0.0]])
    store.save(str(tmp_path))
    monkeypatch.setattr(app, "INDEX_DIR", str(tmp_path))
    monkeypatch.setattr(app, "MIN_SCORE", 0.5)
    monkeypatch.setattr(app.llm, "embed", lambda texts, query=False: [query_vector])


def test_relevant_question_reaches_llm(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [1.0, 0.0])
    monkeypatch.setattr(app.llm, "generate", lambda prompt: "24 days")
    assert app.ask("How much leave?") == "24 days"


def test_irrelevant_question_is_refused_without_calling_llm(tmp_path, monkeypatch):
    _setup(tmp_path, monkeypatch, [0.0, 1.0])

    def boom(prompt):
        raise AssertionError("LLM must not be called")

    monkeypatch.setattr(app.llm, "generate", boom)
    assert "could not find" in app.ask("Who won the cricket world cup?").lower()
