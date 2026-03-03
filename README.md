# FAQ Chatbot (Pinecone + RAG)

Backend: Python (FastAPI, sentence-transformers, Pinecone).  
Frontend: HTML/CSS/JS chat UI.

## What you need to do once

1. **Backend `.env`**  
   Open `Backend/.env` and set:
   - `PINECONE_API_KEY` = your Pinecone API key  
   - `PINECONE_INDEX_NAME` = `faq-chatbot` (already set)

2. **Index FAQ (one-time)**  
   From project root:
   ```bash
   cd Backend
   pip install -r requirements.txt
   python index_faq.py
   ```
   This reads `Backend/Frequently Asked Questions.md.txt`, chunks it, embeds with sentence-transformers (384), and upserts to your Pinecone index.

3. **Run backend (pick one)**
   - **Easy:** Double-click `Backend/run_backend.bat`  
   - **Or in terminal:** Run **two separate commands** (don’t use `→`):
     ```bash
     cd Backend
     pip install -r requirements.txt
     uvicorn main:app --reload --host 0.0.0.0 --port 8000
     ```
   - `requirements.txt` is inside `Backend/`, so always run `pip install` and `uvicorn` from the `Backend` folder.

4. **Open in browser**  
   Go to **http://localhost:8000** — the frontend is served by the backend, so one URL. Don’t open `Frontend/index.html` as a file (`file://`) or you’ll get “Failed to fetch”.

## Structure

- `Backend/` — FastAPI, indexing script, FAQ file, `.env`
- `Frontend/` — Chat UI (HTML, CSS, JS)
