"""
FastAPI service to ingest uploaded PDFs into OpenSearch.

Uploads are saved temporarily, processed via the existing chunking + ingestion
pipeline, and ingested into the configured index. Designed to be consumed by a
future frontend that can monitor job status.
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Callable
from uuid import uuid4
from datetime import datetime
from collections import defaultdict

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from chunker import (
    create_semantic_chunks,
    process_images_with_caption,
    process_tables_with_description,
)
from ingestion import ingest_all_content_into_opensearch
from workflow_2 import AgenticRagState, run_workflow
from unstructured.partition.pdf import partition_pdf

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="PDF Ingestion Service", version="0.1.0")

# Enable permissive CORS so a future frontend can call this API locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path(os.getenv("PDF_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

INDEX_NAME = os.getenv("PDF_INDEX_NAME", "pdf_content_index")

# In-memory conversation storage
# Format: {conversation_id: [{"role": "user"|"assistant", "content": str, "timestamp": datetime}, ...]}
conversations: dict[str, list[dict]] = defaultdict(list)


def _partition_pdf_for_ingestion(pdf_path: Path):
    """
    Run both media and semantic partition flows for a given PDF.
    """
    logger.info("Partitioning media chunks for %s", pdf_path.name)
    raw_chunks_media = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Figure", "Table"],
        extract_image_block_to_payload=True,
        chunking_strategy=None,
    )
    processed_images = process_images_with_caption(raw_chunks_media, use_gemini=True)
    processed_tables = process_tables_with_description(raw_chunks_media, use_gemini=True)

    logger.info("Partitioning semantic text chunks for %s", pdf_path.name)
    raw_chunks_text = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        chunking_strategy="by_title",
        max_characters=2000,
        combine_text_under_n_chars=500,
        new_after_n_chars=1500,
    )
    semantic_chunks = create_semantic_chunks(raw_chunks_text)
    return processed_images, processed_tables, semantic_chunks


def run_ingestion_job(pdf_path: Path, index_name: str = INDEX_NAME):
    """
    Full ingestion pipeline for a PDF file.
    """
    try:
        images, tables, semantics = _partition_pdf_for_ingestion(pdf_path)
        ingest_all_content_into_opensearch(images, tables, semantics, index_name)
        logger.info("Ingestion complete for %s", pdf_path.name)
    finally:
        try:
            pdf_path.unlink()
        except OSError:
            logger.warning("Failed to delete temp file %s", pdf_path)


def _save_upload_to_disk(upload: UploadFile, destination: Path):
    """
    Persist UploadFile contents to disk.
    """
    with destination.open("wb") as buffer:
        shutil.copyfileobj(upload.file, buffer)


def _validate_pdf(upload: UploadFile):
    """
    Basic validation for PDF uploads.
    """
    filename = upload.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    return filename


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_pdf_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Accept a PDF upload, save it locally, and trigger ingestion in the background.
    """
    original_name = _validate_pdf(file)
    unique_name = f"{uuid4().hex}_{Path(original_name).name}"
    target_path = UPLOAD_DIR / unique_name

    try:
        _save_upload_to_disk(file, target_path)
    except Exception as exc:  # pragma: no cover - file system failure
        logger.exception("Failed to store uploaded file")
        raise HTTPException(status_code=500, detail=f"Failed to store file: {exc}") from exc

    logger.info("Stored upload at %s", target_path)
    background_tasks.add_task(run_ingestion_job, target_path, INDEX_NAME)

    return {
        "message": "Ingestion started",
        "original_filename": original_name,
        "stored_filename": unique_name,
        "index": INDEX_NAME,
    }


@app.post("/query")
async def query_rag(payload: dict):
    """
    Run the Agentic RAG workflow (from workflow_2) for a given query string.
    Supports conversation memory by including previous messages in context.

    Request body:
    {
      "query": "your question here",
      "conversation_id": "optional-conversation-id"
    }

    Response includes the full workflow state, including the final combined
    summary (`r_g_summary`) and intermediate fields, plus conversation_id.
    """
    query = (payload or {}).get("query")
    if not query or not isinstance(query, str):
        raise HTTPException(status_code=400, detail="Field 'query' (string) is required.")

    conversation_id = (payload or {}).get("conversation_id")
    if not conversation_id:
        conversation_id = str(uuid4())

    # Get conversation history
    history = conversations.get(conversation_id, [])
    
    # Build context from previous messages (last 5 exchanges for context window)
    context_messages = history[-10:] if len(history) > 10 else history
    context_text = ""
    if context_messages:
        context_parts = []
        for msg in context_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                context_parts.append(f"User: {content}")
            else:
                context_parts.append(f"Assistant: {content}")
        context_text = "\n".join(context_parts)
        # Enhance query with context
        if context_text:
            query_with_context = f"Previous conversation:\n{context_text}\n\nCurrent question: {query}"
        else:
            query_with_context = query
    else:
        query_with_context = query

    # Store user message
    conversations[conversation_id].append({
        "role": "user",
        "content": query,
        "timestamp": datetime.now().isoformat()
    })

    initial_state: AgenticRagState = {"query": query_with_context}
    try:
        final_state = run_workflow(initial_state)
        
        # Store assistant response
        assistant_response = final_state.get("r_g_summary", "")
        conversations[conversation_id].append({
            "role": "assistant",
            "content": assistant_response,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as exc:  # pragma: no cover - LLM or network failure
        logger.exception("Workflow execution failed")
        raise HTTPException(status_code=500, detail=f"Workflow execution failed: {exc}") from exc

    # Return the state with conversation_id added
    response = dict(final_state)
    response["conversation_id"] = conversation_id
    return response


@app.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    Retrieve conversation history for a given conversation_id.
    """
    if conversation_id not in conversations:
        return {"conversation_id": conversation_id, "messages": []}
    
    return {
        "conversation_id": conversation_id,
        "messages": conversations[conversation_id]
    }


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    Delete a conversation by conversation_id.
    """
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"status": "deleted", "conversation_id": conversation_id}


