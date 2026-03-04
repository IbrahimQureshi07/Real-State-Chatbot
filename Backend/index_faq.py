"""
One-time script: Load FAQ file, chunk by Q&A, embed with OpenAI (1024 dims),
and upsert to Pinecone index. No sentence-transformers — small deploy image.
Run: python index_faq.py
"""
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

FAQ_PATH = Path(__file__).parent / "Frequently Asked Questions.md.txt"
BATCH_SIZE = 50
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1024


def chunk_faq(text: str) -> list[str]:
    """Split FAQ into chunks by numbered questions (e.g. '1. ' or '2.\t')."""
    parts = re.split(r"\n(?=\d+[.\t]\s)", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < 20:
            continue
        if len(p) > 2000:
            for i in range(0, len(p), 1500):
                chunk = p[i : i + 1500].strip()
                if len(chunk) >= 50:
                    chunks.append(chunk)
        else:
            chunks.append(p)
    return chunks


def get_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    """Get OpenAI embeddings for a batch of texts (1024 dims)."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [item.embedding for item in response.data]


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "faq-chatbot")
    openai_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing PINECONE_API_KEY in .env")
    if not openai_key:
        raise SystemExit("Missing OPENAI_API_KEY in .env")

    if not FAQ_PATH.exists():
        raise SystemExit(f"FAQ file not found: {FAQ_PATH}")

    print("Loading FAQ...")
    text = FAQ_PATH.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_faq(text)
    print(f"Got {len(chunks)} chunks.")

    print("Connecting to OpenAI and Pinecone...")
    openai_client = OpenAI(api_key=openai_key.strip())
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print("Embedding and upserting in batches...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [str(uuid.uuid4()) for _ in batch]
        emb = get_embeddings(openai_client, batch)
        vectors = [
            {"id": id_, "values": vec, "metadata": {"text": txt[:30000]}}
            for id_, vec, txt in zip(ids, emb, batch)
        ]
        index.upsert(vectors=vectors)
        print(f"  Upserted {i + len(batch)} / {len(chunks)}")

    print("Done. Run the API with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
