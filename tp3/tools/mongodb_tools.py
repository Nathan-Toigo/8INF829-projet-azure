"""MongoDB access layer for the ThreeTokens care agent.

A single ``pymongo`` client backs the seven collections from spec section 7:
``patients``, ``documents``, ``clinical_resources``, ``care_plans``,
``agent_runs``, ``audit_events``, ``memory_snapshots``. Helpers return plain
dicts with stringified ``_id`` so they are JSON/Streamlit friendly.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection

from config import settings

_client: MongoClient | None = None

COLLECTIONS = [
    "patients",
    "documents",
    "clinical_resources",
    "care_plans",
    "agent_runs",
    "audit_events",
    "memory_snapshots",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_client() -> MongoClient:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URI, serverSelectionTimeoutMS=5000)
    return _client


def get_db():
    return get_client()[settings.MONGODB_DB]


def collection(name: str) -> Collection:
    return get_db()[name]


def ping() -> bool:
    """Return True if MongoDB is reachable."""
    try:
        get_client().admin.command("ping")
        return True
    except Exception:
        return False


def ensure_indexes() -> None:
    """Create the indexes the app relies on (idempotent)."""
    try:
        collection("documents").create_index([("patientId", ASCENDING)])
        collection("clinical_resources").create_index([("patientId", ASCENDING)])
        collection("care_plans").create_index([("patientId", ASCENDING)])
        collection("agent_runs").create_index([("patientId", ASCENDING)])
        collection("memory_snapshots").create_index([("runId", ASCENDING)])
    except Exception:
        pass


def _clean(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc


# --- patients ---------------------------------------------------------------

def create_patient(data: dict) -> str:
    data = {**data, "createdAt": data.get("createdAt") or _now()}
    return str(collection("patients").insert_one(data).inserted_id)


def get_patient(patient_id: str) -> dict | None:
    return _clean(collection("patients").find_one({"_id": patient_id}))


def list_patients() -> list[dict]:
    return [_clean(d) for d in collection("patients").find().sort("createdAt", -1)]


def upsert_patient(patient_id: str, data: dict) -> None:
    collection("patients").update_one(
        {"_id": patient_id},
        {"$set": {**data, "updatedAt": _now()}, "$setOnInsert": {"createdAt": _now()}},
        upsert=True,
    )


# --- documents --------------------------------------------------------------

def insert_document(doc: dict) -> str:
    doc = {**doc, "createdAt": doc.get("createdAt") or _now()}
    return str(collection("documents").insert_one(doc).inserted_id)


def list_documents(patient_id: str) -> list[dict]:
    cur = collection("documents").find({"patientId": patient_id}).sort("createdAt", 1)
    return [_clean(d) for d in cur]


# --- clinical_resources -----------------------------------------------------

def insert_clinical_resources(resources: list[dict]) -> int:
    if not resources:
        return 0
    stamped = [{**r, "createdAt": r.get("createdAt") or _now()} for r in resources]
    res = collection("clinical_resources").insert_many(stamped)
    return len(res.inserted_ids)


def list_clinical_resources(
    patient_id: str, resource_type: str | None = None
) -> list[dict]:
    query: dict[str, Any] = {"patientId": patient_id}
    if resource_type:
        query["resourceType"] = resource_type
    cur = collection("clinical_resources").find(query)
    return [_clean(d) for d in cur]


# --- care_plans -------------------------------------------------------------

def insert_care_plan(plan: dict) -> str:
    plan = {**plan, "createdAt": plan.get("createdAt") or _now()}
    return str(collection("care_plans").insert_one(plan).inserted_id)


def latest_care_plan(patient_id: str) -> dict | None:
    return _clean(
        collection("care_plans")
        .find_one({"patientId": patient_id}, sort=[("createdAt", -1)])
    )


# --- agent_runs -------------------------------------------------------------

def insert_agent_run(run: dict) -> str:
    run = {**run, "createdAt": run.get("createdAt") or _now()}
    return str(collection("agent_runs").insert_one(run).inserted_id)


def latest_agent_run(patient_id: str) -> dict | None:
    return _clean(
        collection("agent_runs")
        .find_one({"patientId": patient_id}, sort=[("createdAt", -1)])
    )


# --- audit_events -----------------------------------------------------------

def insert_audit_event(event: dict) -> str:
    event = {**event, "createdAt": event.get("createdAt") or _now()}
    return str(collection("audit_events").insert_one(event).inserted_id)


# --- memory_snapshots -------------------------------------------------------

def insert_memory_snapshot(snapshot: dict) -> str:
    snapshot = {**snapshot, "createdAt": snapshot.get("createdAt") or _now()}
    return str(collection("memory_snapshots").insert_one(snapshot).inserted_id)
