"""
One-time script: Load FAQ file, chunk by Q&A, embed with sentence-transformers (384),
and upsert to Pinecone index.
Run: python index_faq.py
"""
import os
import re
import uuid
from pathlib import Path

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

load_dotenv()

FAQ_PATH = Path(__file__).parent / "Frequently Asked Questions.md.txt"
BATCH_SIZE = 100
MODEL_NAME = "all-MiniLM-L6-v2"  # 384 dimensions


def chunk_faq(text: str) -> list[str]:
    """Split FAQ into chunks by numbered questions (e.g. '1. ' or '2.\t')."""
    # Split when we see a new line that starts with digits and a period
    parts = re.split(r"\n(?=\d+[.\t]\s)", text)
    chunks = []
    for p in parts:
        p = p.strip()
        if not p or len(p) < 20:
            continue
        # Optional: limit chunk size (e.g. 1500 chars) and split if needed
        if len(p) > 2000:
            for i in range(0, len(p), 1500):
                chunk = p[i : i + 1500].strip()
                if len(chunk) >= 50:
                    chunks.append(chunk)
        else:
            chunks.append(p)
    return chunks


def main():
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "faq-chatbot")
    if not api_key:
        raise SystemExit("Missing PINECONE_API_KEY in .env")

    if not FAQ_PATH.exists():
        raise SystemExit(f"FAQ file not found: {FAQ_PATH}")

    print("Loading FAQ...")
    text = FAQ_PATH.read_text(encoding="utf-8", errors="replace")
    chunks = chunk_faq(text)
    print(f"Got {len(chunks)} chunks.")

    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)
    index = pc.Index(index_name)

    print("Embedding and upserting in batches...")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        ids = [str(uuid.uuid4()) for _ in batch]
        emb = model.encode(batch).tolist()
        # Store full text in metadata (Pinecone allows ~40KB per value)
        vectors = [
            {"id": id_, "values": vec, "metadata": {"text": txt[:30000]}}
            for id_, vec, txt in zip(ids, emb, batch)
        ]
        index.upsert(vectors=vectors)
        print(f"  Upserted {i + len(batch)} / {len(chunks)}")

    print("Done. Run the API with: uvicorn main:app --reload")


if __name__ == "__main__":
    main()
