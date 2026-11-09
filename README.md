# RAG Document Chatbot

Ask questions about your own documents (`.txt`, `.md`, `.pdf`) and get answers **grounded in those documents**, with source references.
Built from scratch in Python (no LangChain) to understand how Retrieval-Augmented Generation works.

## How it works
```
documents -> chunk (overlapping) -> embeddings (LLM API) -> local vector store (numpy, cosine similarity)
question  -> embedding -> top-k similar chunks -> prompt with context -> LLM answer + sources
```
| Module | Role |
|---|---|
| `rag/loader.py` | read txt / md / pdf |
| `rag/chunking.py` | overlapping word chunks |
| `rag/llm.py` | embeddings + generation for Gemini or OpenAI (REST) |
| `rag/store.py` | save/load vectors, cosine-similarity search |
| `app.py` | CLI: `ingest`, `ask`, `chat` |

## Setup
```bash
pip install -r requirements.txt
cp .env.example .env          # add your API key (never commit .env)
python app.py ingest data/    # build the index
python app.py ask "How many days of paid leave do employees get?"
python app.py chat
```
Switch provider with `LLM_PROVIDER=openai` in `.env`.

## Docker
```bash
docker build -t rag-chatbot .
docker run -it --env-file .env rag-chatbot
```
The index is created inside the container, so run `ingest` and `chat` in the same session
(or mount an `index/` volume to keep it).

## Tests
```bash
pytest
```

## Limitations / next steps
- Pure vector search; add hybrid search or re-ranking
- Simple word-based chunking; try sentence/semantic chunking
- Add a Streamlit UI and retrieval-quality evaluation
