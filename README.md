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

## Retrieval evaluation
```bash
python evaluate.py                       # hit@1, hit@3 and refusal rate on eval/questions.json
python evaluate.py --chunk-size 80 --overlap 20 -k 5   # compare chunking settings
```
Each question lists a text that must appear in a retrieved chunk (or `null` if the documents cannot answer it, so the bot should refuse). Only chunks scoring at least `MIN_SCORE` count, the same rule the chatbot uses. It needs your API key. The test `tests/test_evaluate.py` checks the metric code offline.

## Tests
```bash
pytest
```

## Limitations / next steps
- Pure vector search; add hybrid search or re-ranking
- Simple word-based chunking; try sentence/semantic chunking
- The evaluation set is tiny (11 questions on one demo file); it shows the method, not a benchmark
- Add a Streamlit UI
