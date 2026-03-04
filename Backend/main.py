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

TOP_K = 15
MIN_SCORE = 0.4  # Use chunks with similarity above this so link-rich chunks aren't dropped
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


SYSTEM_PROMPT = """You are a helpful FAQ assistant for South Carolina Real Estate & Licensing (SCREC). Your answers must come from the "Context from FAQ" provided below.

Rules:
1. If the user only says a greeting (hi, hello, hey) or something off-topic, reply in one short friendly sentence and invite them to ask about real estate or licensing in South Carolina.

2. When the user asks about real estate or licensing: Use the "Context from FAQ" below. When that context contains real FAQ content, write a clear, direct answer in plain language. Do not say you lack information when the context clearly contains the answer.

3. IMPORTANT – Links and URLs: If the context contains any URLs, application links, or PDF links (e.g. "Online Applications Link: https://...", "PDF Exam Application Link: https://..."), you MUST include those exact links in your answer. When the user asks how to apply online or for application links, give them the links from the context. Do not reply with "contact the Commission" or "visit the website" when the context already provides the specific link.

4. Only say you don't have specific information or suggest contacting the Commission when the context is literally "(No relevant FAQ context found.)" or empty.

5. Do not make up facts. Do not include "Q:" or question numbers. Keep the tone helpful and professional."""


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


def _embedding_query(message: str) -> str:
    """Expand query for retrieval when user asks for links/apply online, so we fetch link-rich chunks."""
    msg = message.lower().strip()
    if any(w in msg for w in ("link", "apply online", "application link", "url", "website", "where to apply")):
        return message + " PDF Exam Application Link Online Applications Link instructions apply"
    return message


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(answer="Please ask a question.", sources=[])

    index = get_index()
    query_for_retrieval = _embedding_query(req.message.strip())
    query_embedding = get_embedding(query_for_retrieval)
    result = index.query(vector=query_embedding, top_k=TOP_K, include_metadata=True)

    sources = []
    texts = []
    for match in result.get("matches", []):
        score = match.get("score")
        if score is not None and score < MIN_SCORE:
            continue
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        if text:
            sources.append(text[:200] + "..." if len(text) > 200 else text)
            texts.append(text)

    context = "\n\n---\n\n".join(texts) if texts else ""
    # When no chunks retrieved, LLM must know so it can say "no info" instead of hallucinating
    context_for_llm = context if context.strip() else "(No relevant FAQ context found.)"

    answer = generate_answer_with_openai(req.message.strip(), context_for_llm)
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
