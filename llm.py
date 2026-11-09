"""Embeddings + text generation for Gemini or OpenAI over plain REST (no SDK).

Pick the provider with LLM_PROVIDER=gemini|openai. Model names can be overridden
with env vars, so a deprecated model never needs a code change.
"""
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BATCH = 64
TIMEOUT = 60

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
OPENAI_BASE = "https://api.openai.com/v1"


def _provider() -> str:
    p = os.getenv("LLM_PROVIDER", "gemini").strip().lower()
    if p not in {"gemini", "openai"}:
        raise SystemExit(f"LLM_PROVIDER must be 'gemini' or 'openai', got '{p}'.")
    return p


def _key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise SystemExit(f"Missing {name}. Copy .env.example to .env and add your key.")
    return value


def _post(url: str, headers: dict, payload: dict) -> dict:
    """POST with a few retries on rate limits / server errors. Never prints the API key."""
    for attempt in range(4):
        resp = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
        if resp.status_code in {429, 500, 502, 503, 504} and attempt < 3:
            time.sleep(2**attempt)
            continue
        if not resp.ok:
            raise RuntimeError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
        return resp.json()
    raise RuntimeError("LLM API request failed after retries")


def _embed_gemini(texts: list[str], query: bool) -> list[list[float]]:
    model = os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001")
    headers = {"x-goog-api-key": _key("GEMINI_API_KEY")}
    task = "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
    payload = {
        "requests": [
            {"model": f"models/{model}", "content": {"parts": [{"text": t}]}, "taskType": task}
            for t in texts
        ]
    }
    data = _post(f"{GEMINI_BASE}/{model}:batchEmbedContents", headers, payload)
    return [e["values"] for e in data["embeddings"]]


def _embed_openai(texts: list[str]) -> list[list[float]]:
    model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"}
    data = _post(f"{OPENAI_BASE}/embeddings", headers, {"model": model, "input": texts})
    return [item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"])]


def embed(texts: list[str], query: bool = False) -> list[list[float]]:
    """Embed texts in batches. query=True marks them as search queries (used by Gemini)."""
    provider = _provider()
    vectors: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i : i + BATCH]
        vectors += _embed_gemini(batch, query) if provider == "gemini" else _embed_openai(batch)
    return vectors


def generate(prompt: str) -> str:
    """Return the model's answer text for a prompt (temperature 0 for grounded answers)."""
    if _provider() == "gemini":
        model = os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash")
        headers = {"x-goog-api-key": _key("GEMINI_API_KEY")}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0},
        }
        data = _post(f"{GEMINI_BASE}/{model}:generateContent", headers, payload)
        parts = data["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    headers = {"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"}
    payload = {"model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}
    data = _post(f"{OPENAI_BASE}/chat/completions", headers, payload)
    return data["choices"][0]["message"]["content"].strip()
