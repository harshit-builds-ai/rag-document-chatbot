"""RAG Document Chatbot: ingest documents, then ask questions answered from them only."""
import argparse
import os

from rag import llm
from rag.chunking import chunk_text
from rag.loader import load_documents
from rag.store import VectorStore

INDEX_DIR = "index"
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))

PROMPT = """Answer the QUESTION using only the CONTEXT below.
If the answer is not in the context, say "I could not find that in the documents."
Mention the source tags like [file.txt #2] that you used.

CONTEXT:
{context}

QUESTION: {question}
"""


def ingest(path: str) -> None:
    docs = load_documents(path)
    if not docs:
        raise SystemExit(f"No .txt/.md/.pdf content found in '{path}'.")
    chunks = []
    for d in docs:
        for i, text in enumerate(chunk_text(d["text"])):
            chunks.append({"source": d["source"], "id": i, "text": text})
    store = VectorStore()
    store.add(chunks, llm.embed([c["text"] for c in chunks]))
    store.save(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} document(s).")


def ask(question: str, k: int = 4) -> str:
    store = VectorStore.load(INDEX_DIR)
    hits = store.search(llm.embed([question], query=True)[0], k=k)
    hits = [h for h in hits if h[0] >= MIN_SCORE]
    if not hits:
        return "I could not find that in the documents."
    context = "\n\n".join(f"[{c['source']} #{c['id']}] {c['text']}" for _, c in hits)
    return llm.generate(PROMPT.format(context=context, question=question))


def main() -> None:
    p = argparse.ArgumentParser(description="RAG Document Chatbot")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest", help="index a file or folder").add_argument("path")
    sub.add_parser("ask", help="ask one question").add_argument("question")
    sub.add_parser("chat", help="interactive chat")
    a = p.parse_args()

    if a.cmd == "ingest":
        ingest(a.path)
    elif a.cmd == "ask":
        print(ask(a.question))
    else:
        print("Type your question (or 'exit').")
        while (q := input("you> ").strip()).lower() not in {"exit", "quit"}:
            if q:
                print(ask(q), "\n")


if __name__ == "__main__":
    main()
