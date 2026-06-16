"""Upload pipeline: file -> OCR -> MongoDB -> clinical entities -> ChromaDB.

Ties together ``llm_ocr``, ``clinical_extraction``, ``chunking`` and the storage
layers so the Streamlit Upload page can ingest a document with one call.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import settings
from ingestion import chunking, clinical_extraction, llm_ocr
from tools import chromadb_tools, mongodb_tools


def ensure_patient(patient_id: str | None, name: str | None = None) -> str:
    """Return a patient id, creating a minimal patient record if needed."""
    if not patient_id:
        patient_id = f"patient-{uuid.uuid4().hex[:8]}"
    existing = mongodb_tools.get_patient(patient_id)
    if existing is None:
        mongodb_tools.upsert_patient(
            patient_id,
            {"name": name or "Redacted", "metadata": {}},
        )
    return patient_id


def save_upload(file_bytes: bytes, file_name: str) -> Path:
    settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    dest = settings.UPLOADS_DIR / file_name
    dest.write_bytes(file_bytes)
    return dest


def ingest_path(
    path: Path, patient_id: str, force_vision: bool = False
) -> dict:
    """Run the full ingestion pipeline on a saved file. Returns a summary."""
    path = Path(path)

    ocr = llm_ocr.run_ocr(path, force_vision=force_vision)
    raw_text = ocr["extracted_text"]

    extraction, extract_usage = clinical_extraction.extract_entities(raw_text)

    document_id = mongodb_tools.insert_document(
        {
            "patientId": patient_id,
            "fileName": path.name,
            "documentType": extraction.document_type,
            "rawText": raw_text,
            "ocrConfidence": ocr["confidence"],
            "ocrMethod": ocr["method"],
            "pageCount": ocr["page_count"],
            "source": "upload",
        }
    )

    resources = clinical_extraction.entities_to_resources(
        extraction.clinical_entities, patient_id, document_id
    )
    resources_inserted = mongodb_tools.insert_clinical_resources(resources)

    chunks = chunking.chunk_text(raw_text)
    chunks_indexed = chromadb_tools.add_chunks(
        chromadb_tools.PATIENT_DOCUMENTS,
        source=f"{patient_id}::{document_id}::{path.name}",
        chunks=chunks,
        base_metadata={
            "patientId": patient_id,
            "documentId": document_id,
            "fileName": path.name,
            "documentType": extraction.document_type,
        },
    )

    mongodb_tools.insert_audit_event(
        {
            "patientId": patient_id,
            "event": "document_ingested",
            "documentId": document_id,
            "fileName": path.name,
            "method": ocr["method"],
            "chunksIndexed": chunks_indexed,
            "resourcesExtracted": resources_inserted,
        }
    )

    return {
        "document_id": document_id,
        "file_name": path.name,
        "document_type": extraction.document_type,
        "ocr_method": ocr["method"],
        "ocr_confidence": ocr["confidence"],
        "page_count": ocr["page_count"],
        "resources_extracted": resources_inserted,
        "chunks_indexed": chunks_indexed,
        "text_preview": raw_text[:600],
    }


def ingest_upload(
    file_bytes: bytes,
    file_name: str,
    patient_id: str | None = None,
    force_vision: bool = False,
) -> dict:
    patient_id = ensure_patient(patient_id)
    saved = save_upload(file_bytes, file_name)
    summary = ingest_path(saved, patient_id, force_vision=force_vision)
    summary["patient_id"] = patient_id
    return summary
