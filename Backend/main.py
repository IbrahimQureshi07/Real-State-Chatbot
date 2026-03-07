"""
FastAPI backend: /chat endpoint + serves Frontend so you open one URL (no file://).
Uses OpenAI for embeddings (1024 dims) and for answer generation. No sentence-transformers.
Logs every user question to PostgreSQL (Railway) so the admin can see what users ask.
"""
import os
import re
from datetime import datetime
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

TOP_K = 20
MIN_SCORE = 0.35  # Slightly lower so "Unit I hours" and similar queries still get relevant chunks
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


# ── PostgreSQL helpers ────────────────────────────────────────────────────────

def _db_connect():
    """Return a fresh psycopg2 connection, or None if DATABASE_URL not configured."""
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return None
    # psycopg2 needs postgresql:// not postgres://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    try:
        conn = psycopg2.connect(url, connect_timeout=5)
        conn.autocommit = True
        return conn
    except Exception as e:
        print(f"[DB] connect error: {e}")
        return None


def init_db():
    """Create the questions table if it does not exist."""
    conn = _db_connect()
    if not conn:
        print("[DB] DATABASE_URL not set – question logging disabled.")
        return
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS questions (
                    id        SERIAL PRIMARY KEY,
                    question  TEXT        NOT NULL,
                    answer    TEXT,
                    asked_at  TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        print("[DB] Table 'questions' ready.")
    except Exception as e:
        print(f"[DB] init error: {e}")
    finally:
        conn.close()


def log_question(question: str, answer: str) -> None:
    """Insert one row into questions table. Silently skips on any error."""
    conn = _db_connect()
    if not conn:
        return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO questions (question, answer) VALUES (%s, %s)",
                (question, answer[:1000] if answer else ""),
            )
    except Exception as e:
        print(f"[DB] log error: {e}")
    finally:
        conn.close()


@app.on_event("startup")
def startup():
    init_db()


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


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] | None = None  # recent conversation for memory


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]
    suggestions: list[str] = []  # follow-up question chips for the user


SYSTEM_PROMPT = """You are a warm, knowledgeable real estate assistant specializing in South Carolina Real Estate & Licensing (SCREC). Your personality is friendly, approachable, and professional — like a helpful colleague, not a robot. Always be conversational and natural.

━━━ TOPIC SCOPE ━━━

A. REAL ESTATE TOPICS (answer these):
   • Anything about real estate concepts, terms, licensing, exams, CE hours, fees, applications — even if the user types with typos, incomplete sentences, or mixes languages.
   • Math/calculations related to real estate (fractions of CE hours, fees, percentages, etc.) — use your general knowledge.
   • General real estate knowledge (what is escrow, what is a deed, etc.) — answer from general knowledge, but note it is general info.
   • If the question is about South Carolina specifically and the "Context from FAQ" has the answer — use that as the primary source.
   • If the question is about another US state's real estate/licensing — give a brief general answer and note: "Note: Our FAQ is specific to South Carolina. For [state] details, please check that state's real estate commission."

B. OFF-TOPIC (do NOT answer, be gentle):
   • Completely unrelated topics: politics, celebrities, sports, cooking, weather, general trivia, "who is X", "what is 2+2" (pure math with no real estate context).
   • For these, warmly say: "I'm here to help with real estate and licensing questions for South Carolina! Feel free to ask me anything about that."

━━━ BEHAVIOR RULES ━━━

1. TYPO TOLERANCE: If a message looks like it has a typo or is slightly incomplete but is clearly about real estate (e.g. starts with "f a real estate agent" instead of "If a real estate agent"), ALWAYS try to understand the intent and answer. Never reject a real estate question just because of a missing letter or grammar issue.

2. GREETINGS: If the user says hi, hello, hey, or similar — respond warmly and briefly, introduce yourself as the South Carolina real estate assistant, and invite them to ask their question.

3. CONVERSATION MEMORY: If the user refers to previous messages ("what did I ask?", "what were we discussing?", "mainay abhi kia phucha?") — look at the conversation history and summarize naturally.

4. SC FAQ (primary source): When the "Context from FAQ" contains relevant South Carolina FAQ content, use it as your primary answer. Do not say you lack information when the context clearly has the answer.

5. LINKS: If the context contains application links or URLs, include them in your answer. Example: "You can apply online at [Online Applications](https://llr.sc.gov/re/Instructions.aspx)"

6. TONE: Be friendly and human. Use natural language. Avoid stiff, robotic phrases like "I only answer questions about...". Instead: "That's a bit outside what I cover, but I'd love to help with real estate questions!"

7. FORMAT: Plain text only. No LaTeX. Write fractions as "2/5" not \\frac{2}{5}. No "Q:" prefixes.

8. SUGGESTED FOLLOW-UPS: At the very end of every answer, on its own line, add:
Suggested follow-ups: [Short question 1?] | [Short question 2?] | [Short question 3?]
Make them short, natural, and relevant to what the user just asked. 2–3 suggestions separated by |."""


def _parse_suggestions_from_answer(answer: str) -> tuple[str, list[str]]:
    """Extract 'Suggested follow-ups: Q1 | Q2 | Q3' from end of answer; return (clean_answer, suggestions)."""
    pattern = r"\n*Suggested follow-ups:\s*([^\n]+)$"
    match = re.search(pattern, answer, re.IGNORECASE)
    if not match:
        return answer.strip(), []
    raw = match.group(1).strip()
    clean_answer = answer[: match.start()].strip()
    parts = [p.strip() for p in raw.split("|") if p.strip()][:3]
    return clean_answer, parts


def generate_answer_with_openai(
    user_message: str,
    context: str,
    history: list[dict] | None = None,
) -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or not api_key.strip():
        return None
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        for h in history[-10:]:  # last 5 turns (10 messages)
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})
    current_content = f"Context from FAQ:\n\n{context}\n\nUser question: {user_message}"
    messages.append({"role": "user", "content": current_content})
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=1024,
        )
        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
    except Exception:
        pass
    return None


def _embedding_query(message: str) -> str:
    """Expand query for retrieval so we fetch relevant chunks (links, hours, units, etc.)."""
    msg = message.lower().strip()
    extra = []
    if any(w in msg for w in ("link", "apply online", "application link", "url", "website", "where to apply")):
        extra.append("PDF Exam Application Link Online Applications Link instructions apply")
    if any(w in msg for w in ("hour", "hours", "unit i", "unit ii", "unit 1", "unit 2", "required", "pre-licensing", "course")):
        extra.append("pre-licensing course hours Unit I Sales Unit II Advanced Real Estate 60 hours 30 hours")
    if extra:
        return message + " " + " ".join(extra)
    return message


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        return ChatResponse(answer="Please ask a question.", sources=[], suggestions=[])

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

    history_dicts = None
    if req.history:
        history_dicts = [{"role": h.role, "content": h.content} for h in req.history]

    answer = generate_answer_with_openai(req.message.strip(), context_for_llm, history_dicts)
    if answer:
        clean_answer, suggestions = _parse_suggestions_from_answer(answer)
        log_question(req.message.strip(), clean_answer)
        return ChatResponse(answer=clean_answer, sources=[], suggestions=suggestions)
    if texts:
        answer = "\n\n".join(texts)[:8000]
    else:
        answer = "No relevant answer found in the FAQ. Try asking about real estate or licensing in South Carolina."
    log_question(req.message.strip(), answer)
    return ChatResponse(answer=answer, sources=sources, suggestions=[])


@app.get("/health")
def health():
    return {"status": "ok"}


# ── Admin endpoints ───────────────────────────────────────────────────────────

@app.get("/admin/questions")
def admin_questions_json():
    """Return all logged questions as JSON (newest first)."""
    conn = _db_connect()
    if not conn:
        return {"error": "DATABASE_URL not configured", "questions": []}
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT id, question, answer, asked_at FROM questions ORDER BY asked_at DESC")
            rows = cur.fetchall()
        return {"total": len(rows), "questions": [dict(r) for r in rows]}
    except Exception as e:
        return {"error": str(e), "questions": []}
    finally:
        conn.close()


@app.get("/admin", response_class=HTMLResponse)
def admin_page():
    """Simple HTML dashboard showing all user questions."""
    conn = _db_connect()
    rows = []
    error = ""
    if not conn:
        error = "DATABASE_URL not configured – logging is disabled."
    else:
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT id, question, answer, asked_at FROM questions ORDER BY asked_at DESC"
                )
                rows = cur.fetchall()
        except Exception as e:
            error = str(e)
        finally:
            conn.close()

    rows_html = ""
    for r in rows:
        asked = str(r["asked_at"])[:19].replace("T", " ")
        q = str(r["question"]).replace("<", "&lt;").replace(">", "&gt;")
        a = str(r["answer"] or "")[:200].replace("<", "&lt;").replace(">", "&gt;")
        rows_html += f"""
        <tr>
          <td>{r['id']}</td>
          <td>{asked}</td>
          <td>{q}</td>
          <td class="ans">{a}{"…" if len(str(r["answer"] or "")) > 200 else ""}</td>
        </tr>"""

    error_html = f'<p class="err">{error}</p>' if error else ""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>FAQ Chatbot – Admin</title>
<style>
  body{{font-family:system-ui,sans-serif;background:#1a1b26;color:#c0caf5;padding:2rem;}}
  h1{{color:#7aa2f7;margin-bottom:0.25rem;}}
  p.sub{{color:#a9b1d6;margin-top:0;}}
  .err{{color:#f7768e;}}
  table{{width:100%;border-collapse:collapse;margin-top:1.5rem;}}
  th{{background:#414868;color:#e0af68;padding:0.6rem 0.8rem;text-align:left;}}
  td{{padding:0.55rem 0.8rem;border-bottom:1px solid #414868;vertical-align:top;}}
  td.ans{{color:#a9b1d6;font-size:0.85rem;}}
  tr:hover td{{background:#24283b;}}
  .badge{{background:#7aa2f7;color:#1a1b26;border-radius:999px;padding:0.15rem 0.6rem;font-size:0.8rem;}}
</style>
</head>
<body>
<h1>FAQ Chatbot – Questions Log</h1>
<p class="sub">Total questions: <span class="badge">{len(rows)}</span></p>
{error_html}
<table>
  <thead><tr><th>#</th><th>Time</th><th>Question</th><th>Answer (preview)</th></tr></thead>
  <tbody>{rows_html if rows_html else '<tr><td colspan="4" style="text-align:center;color:#a9b1d6;">No questions yet.</td></tr>'}</tbody>
</table>
</body>
</html>"""


FRONTEND_DIR = Path(__file__).resolve().parent.parent / "Frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
