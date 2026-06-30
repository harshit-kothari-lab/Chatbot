"""
FastAPI backend wiring together ingestion.py and rag_chat.py.

Endpoints:
  POST /upload  -> ingest a single PDF incrementally into the existing Chroma DB
  POST /chat    -> answer a query, either RAG-grounded or general LLM mode
  GET  /health  -> simple healthcheck

Run with: uvicorn app:app --host 0.0.0.0 --port 8000
"""

import os
import shutil
import logging
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- import your existing pipelines -----------------------------------------
# ingestion.py and rag_chat.py both run heavy setup at import time
# (embedding model, OCR reader, cross encoder, Chroma DB, Groq LLM client).
# That happens once, here, at server startup.
import ingestion
import rag_chat

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="RAG Chatbot API")

# Tighten allow_origins to your actual Lovable domain once deployed,
# e.g. ["https://your-app.lovable.app"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("data/pdfs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# /upload  — incremental ingestion (does NOT wipe existing DB)
# =============================================================================

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        # 1. extract -> 2. chunk (reuses your existing functions as-is)
        docs = ingestion.extract_pdf_pages(str(save_path))
        if not docs:
            raise HTTPException(422, "No usable content extracted from this PDF.")

        chunks = ingestion.split_documents(docs)
        if not chunks:
            raise HTTPException(422, "No valid chunks produced from this PDF.")

        # 3. add to the SAME Chroma instance rag_chat.py already has loaded,
        #    instead of build_vectorstore()'s Chroma.from_documents(), which
        #    would overwrite the whole collection.
        ids = [
            f"{c.metadata['source']}_{c.metadata['page']}_{c.metadata['chunk_id']}"
            for c in chunks
        ]
        rag_chat.db.add_documents(chunks, ids=ids)

        logger.info(f"Ingested {file.filename}: {len(chunks)} chunks added.")

        return {
            "status": "success",
            "filename": file.filename,
            "pages_processed": len({d.metadata["page"] for d in docs}),
            "chunks_added": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {e}")


# =============================================================================
# /chat — RAG mode or General mode
# =============================================================================

class ChatRequest(BaseModel):
    message: str
    mode: str = "rag"  # "rag" or "general"


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "Message cannot be empty.")

    if req.mode == "rag":
        result = rag_chat.answer_query(req.message)
        return {
            "answer": result["answer"],
            "citations": [
                {"source": s["source"], "page": s["page"]}
                for s in result["sources"]
            ],
            "mode": "rag",
        }

    elif req.mode == "general":
        # Bypass retrieval entirely — plain Groq call, no document grounding.
        response = rag_chat.llm.invoke(req.message)
        return {
            "answer": response.content.strip(),
            "citations": [],
            "mode": "general",
        }

    else:
        raise HTTPException(400, "mode must be 'rag' or 'general'.")


@app.get("/health")
async def health():
    return {"status": "ok"}
