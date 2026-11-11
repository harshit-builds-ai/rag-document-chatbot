"""Offline tests for the evaluation script (no API key, no network)."""
import json
from pathlib import Path

import evaluate
from rag.loader import load_documents
from rag.store import VectorStore

QUESTIONS = [
    {"question": "q1", "expected": "alpha fact"},
    {"question": "q2", "expected": "beta fact"},
    {"question": "q3", "expected": None},
]


def _store():
    store = VectorStore()
    store.add(
        [{"source": "d", "id": 0, "text": "the alpha fact is here"},
         {"source": "d", "id": 1, "text": "the beta fact is here"}],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    return store


def test_metrics_with_hits_misses_and_refusals():
    # q1 finds alpha first, q2 asks for beta but its vector points at alpha (miss), q3 matches nothing
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.7, 0.7]]
    r = evaluate.evaluate(_store(), QUESTIONS, vectors, k=1, min_score=0.9)
    assert r["answerable"] == 2 and r["hit_at_1"] == 0.5 and r["hit_at_k"] == 0.5
    assert r["unanswerable"] == 1 and r["refusal_rate"] == 1.0
    assert len(r["failures"]) == 1 and r["failures"][0]["question"] == "q2"


def test_unanswerable_question_that_retrieves_something_is_a_failure():
    r = evaluate.evaluate(_store(), [QUESTIONS[2]], [[1.0, 0.0]], k=1, min_score=0.5)
    assert r["refusal_rate"] == 0.0 and r["failures"]


def test_below_min_score_counts_as_no_hit():
    r = evaluate.evaluate(_store(), [QUESTIONS[0]], [[0.7, 0.7]], k=2, min_score=0.9)
    assert r["hit_at_k"] == 0.0


def test_every_expected_text_exists_in_the_sample_docs():
    """Dataset sanity check: a question must be answerable by some chunk at the default settings."""
    docs = load_documents("data")
    chunks = evaluate.build_chunks(docs, evaluate.DEFAULT_SIZE, evaluate.DEFAULT_OVERLAP)
    texts = [evaluate._norm(c["text"]) for c in chunks]
    questions = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    for q in questions:
        if q["expected"] is not None:
            assert any(evaluate._norm(q["expected"]) in t for t in texts), q["question"]
