"""
FastAPI backend: /chat endpoint + serves Frontend so you open one URL (no file://).
Uses OpenAI for embeddings (1024 dims) and for answer generation. No sentence-transformers.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

TOP_K = 5
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024
OPENAI_MODEL = "gpt-4o-mini"

app = FastAPI(title="FAQ Chatbot API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_openai_client = None
_index = None


def get_openai_client():
    global _openai_client
    if _openai_client is None:
        key = os.getenv("OPENAI_API_KEY")
        if not key or not key.strip():
            raise RuntimeError("OPENAI_API_KEY not set")
        _openai_client = OpenAI(api_key=key.strip())
    return _openai_client


def get_embedding(text: str) -> list[float]:
    """Get 1024-dim embedding for one text (user query)."""
    client = get_openai_client()
    r = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return r.data[0].embedding


def get_index():
    global _index
    if _index is None:
        api_key = os.getenv("PINECONE_API_KEY")
        index_name = os.getenv("PINECONE_INDEX_NAME", "faq-chatbot")
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY not set")
        pc = Pinecone(api_key=api_key)
        _index = pc.Index(index_name)
    return _index


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


SYSTEM_PROMPT = """You are a helpful FAQ assistant for South Carolina Real Estate & Licensing (SCREC). Follow these rules:

1. If the user only says a greeting (e.g. hi, hey, hello, what's up) or something clearly off-topic, reply in one short, friendly sentence and invite them to ask about real estate or licensing in South Carolina. Do not paste FAQ content.

2. If the user asks a real question about real estate or licensing, use ONLY the "Context from FAQ" below to write one clear, coherent answer. Synthesize the information—do not list multiple separate answers or repeat the same definition. Write as a single helpful paragraph (or two if needed). Do not make up information; if the context does not contain the answer, say so briefly.

3. Keep answers focused and in plain language. Do not include question numbers or "Q:" in your reply."""


def generate_answer_with_openai(user_message: str, context: str) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return None
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Context from FAQ:\n\n{context}\n\nUser question: {user_message}"},
            ],
            max_tokens=1024,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
    except Exception:
        pass
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(answer="Please ask a question.", sources=[])

    index = get_index()
    query_embedding = get_embedding(req.message.strip())
    result = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    sources = []
    texts = []
    for match in result.get("matches", []):
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        if text:
            sources.append(text[:200] + "..." if len(text) > 200 else text)
            texts.append(text)

    context = "\n\n---\n\n".join(texts) if texts else ""

    answer = generate_answer_with_openai(req.message.strip(), context or "(No relevant FAQ context found.)")
    if answer:
        return ChatResponse(answer=answer, sources=[])
    if texts:
        answer = "\n\n".join(texts)[:8000]
    else:
        answer = "No relevant answer found in the FAQ. Try asking about real estate or licensing in South Carolina."
    return ChatResponse(answer=answer, sources=sources)


@app.get("/health")
def health():
    return {"status": "ok"}


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "Frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
