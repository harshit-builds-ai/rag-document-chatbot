"""Retrieval evaluation: does the right chunk come back for each question?

    python evaluate.py [docs_path] [--questions eval/questions.json]
                       [--chunk-size 40] [--overlap 10] [-k 3]

Each question has an "expected" text that must appear in a retrieved chunk, or
null for a question the documents cannot answer (it should be refused).
Only chunks scoring at least MIN_SCORE count, the same rule app.ask() uses.
Needs the same API key as the chatbot (it embeds the documents and questions).
"""
import argparse
import json
from pathlib import Path

from app import MIN_SCORE
from rag import llm
from rag.chunking import chunk_text
from rag.loader import load_documents
from rag.store import VectorStore

DEFAULT_SIZE, DEFAULT_OVERLAP = 40, 10  # small chunks so a tiny demo corpus has several of them


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def build_chunks(docs: list[dict], size: int, overlap: int) -> list[dict]:
    chunks = []
    for d in docs:
        for i, text in enumerate(chunk_text(d["text"], size, overlap)):
            chunks.append({"source": d["source"], "id": i, "text": text})
    return chunks


def evaluate(store: VectorStore, questions: list[dict], query_vectors, k: int, min_score: float) -> dict:
    answerable = hit1 = hitk = unanswerable = refused = 0
    failures = []
    for q, vec in zip(questions, query_vectors):
        hits = [h for h in store.search(vec, k=k) if h[0] >= min_score]
        if q.get("expected") is None:
            unanswerable += 1
            if hits:
                failures.append({"question": q["question"], "problem": "should have been refused"})
            else:
                refused += 1
            continue
        answerable += 1
        needle = _norm(q["expected"])
        found = [needle in _norm(c["text"]) for _, c in hits]
        hit1 += bool(found[:1] and found[0])
        hitk += any(found)
        if not any(found):
            failures.append({"question": q["question"], "problem": f"expected text not in top-{k}"})
    return {
        "answerable": answerable,
        "hit_at_1": hit1 / answerable if answerable else None,
        "hit_at_k": hitk / answerable if answerable else None,
        "unanswerable": unanswerable,
        "refusal_rate": refused / unanswerable if unanswerable else None,
        "failures": failures,
    }


def run(docs, questions, embed_fn, size=DEFAULT_SIZE, overlap=DEFAULT_OVERLAP, k=3, min_score=MIN_SCORE) -> dict:
    chunks = build_chunks(docs, size, overlap)
    store = VectorStore()
    store.add(chunks, embed_fn([c["text"] for c in chunks]))
    vectors = embed_fn([q["question"] for q in questions], query=True)
    result = evaluate(store, questions, vectors, k, min_score)
    result.update(chunks=len(chunks), chunk_size=size, overlap=overlap, k=k, min_score=min_score)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate retrieval quality")
    p.add_argument("docs", nargs="?", default="data")
    p.add_argument("--questions", default="eval/questions.json")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    p.add_argument("-k", type=int, default=3)
    a = p.parse_args()

    docs = load_documents(a.docs)
    if not docs:
        raise SystemExit(f"No documents found in '{a.docs}'.")
    questions = json.loads(Path(a.questions).read_text(encoding="utf-8"))
    r = run(docs, questions, llm.embed, a.chunk_size, a.overlap, a.k)

    pct = lambda v: "n/a" if v is None else f"{v:.0%}"
    print(f"chunks={r['chunks']} size={r['chunk_size']} overlap={r['overlap']} k={r['k']} min_score={r['min_score']}")
    print(f"hit@1 : {pct(r['hit_at_1'])}  ({r['answerable']} answerable questions)")
    print(f"hit@{r['k']} : {pct(r['hit_at_k'])}")
    print(f"refused when not answerable: {pct(r['refusal_rate'])}  ({r['unanswerable']} questions)")
    for f in r["failures"]:
        print(f"  FAIL: {f['question']} -> {f['problem']}")


if __name__ == "__main__":
    main()
