"""
Step 3: Embed extracted opportunities and upsert to Qdrant.

Reads opportunities_en.json, generates Jina embeddings,
and upserts points with structured payloads to Qdrant.
"""

import json
import time
import logging
from typing import List

import numpy as np
import requests as http_requests
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from countries import normalize_countries

from config import (
    JINA_API_KEY,
    JINA_ENDPOINT,
    JINA_MODEL,
    EMBED_BATCH_SIZE,
    QDRANT_ENDPOINT,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    UPSERT_BATCH_SIZE,
    OPPORTUNITIES_JSON,
)

logger = logging.getLogger(__name__)


# ── Embedding helper ───────────────────────────────────────────────────


def get_jina_embedding(texts: List[str]) -> np.ndarray:
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": JINA_MODEL, "input": texts}
    response = http_requests.post(JINA_ENDPOINT, json=payload, headers=headers, timeout=60)
    response.raise_for_status()
    return np.array(
        [d["embedding"] for d in response.json()["data"]],
        dtype=np.float32,
    )


# ── Data helpers ───────────────────────────────────────────────────────


def ensure_list(val):
    if val is None:
        return []
    return [val] if isinstance(val, str) else list(val)


def build_exam_scores(lang_reqs):
    """Convert language_requirements dict → nested exam_scores list."""
    if not lang_reqs or not isinstance(lang_reqs, dict):
        return []
    scores = []
    for exam_name, score_val in lang_reqs.items():
        try:
            scores.append({"name": exam_name.strip().lower(), "score": float(score_val)})
        except (ValueError, TypeError):
            continue
    return scores


# ── Main ───────────────────────────────────────────────────────────────


def run() -> bool:
    """
    Execute the embedding step.
    Returns True on success.
    """
    if not OPPORTUNITIES_JSON.exists():
        logger.error("opportunities_en.json not found at %s — run extract step first", OPPORTUNITIES_JSON)
        return False

    opportunities = json.loads(OPPORTUNITIES_JSON.read_text(encoding="utf-8"))
    logger.info("Loaded %d opportunities from %s", len(opportunities), OPPORTUNITIES_JSON)

    if not opportunities:
        logger.warning("No opportunities to embed")
        return False

    # ── Prepare texts + metadata ───────────────────────────────────────
    rich_texts = []
    opportunity_data = []

    for s in opportunities:
        # Normalize country names
        if s.get("country"):
            raw = s["country"]
            s["country"] = normalize_countries(raw if isinstance(raw, list) else [raw])
        if s.get("eligible_nationalities") and s["eligible_nationalities"] != "all":
            raw = s["eligible_nationalities"]
            if isinstance(raw, list):
                s["eligible_nationalities"] = normalize_countries(raw)

        rich_text = (
            f"{s.get('title', '')}\n"
            f"{s.get('description', '')}\n"
            f"Eligibility: {s.get('eligibility', '')}\n"
        )
        rich_texts.append(rich_text)
        opportunity_data.append(s)

    # ── Batch embed ────────────────────────────────────────────────────
    all_embeddings = []
    for i in range(0, len(rich_texts), EMBED_BATCH_SIZE):
        batch = rich_texts[i : i + EMBED_BATCH_SIZE]
        logger.info("Embedding batch %d–%d / %d", i, i + len(batch), len(rich_texts))
        embeddings = get_jina_embedding(batch)
        all_embeddings.extend(embeddings)
        time.sleep(1)

    logger.info("Generated %d embeddings", len(all_embeddings))

    # ── Build Qdrant points ────────────────────────────────────────────
    points = []

    for s, embedding in zip(opportunity_data, all_embeddings):
        countries = ensure_list(s.get("country"))
        fund_types = ensure_list(s.get("fund_type"))
        subtypes = ensure_list(s.get("type", {}).get("subtype"))
        target_segments = ensure_list(s.get("target_segment"))
        documents_required = ensure_list(s.get("documents_required"))
        eligible_nationalities = ensure_list(s.get("eligible_nationalities", "all"))
        application_fee = s.get("application_fee")

        exam_scores = build_exam_scores(s.get("language_requirements"))

        payload = {
            "program_id": s["id"],
            "title": s.get("title"),
            "country": countries,
            "fund_type": fund_types,
            "category": s.get("type", {}).get("category"),
            "subtype": subtypes,
            "documents_required": documents_required,
            "exam_scores": exam_scores,
            "is_remote": s.get("is_remote", False),
            "eligible_nationalities": eligible_nationalities,
            "target_segment": target_segments,
            "deadline": s.get("deadline"),
            "min_age": int(s["min_age"]) if s.get("min_age") is not None else None,
            "max_age": int(s["max_age"]) if s.get("max_age") is not None else None,
            "gpa": float(s["gpa"]) if s.get("gpa") is not None else None,
            "has_language_requirements": len(exam_scores) > 0,
            "has_fee": bool(application_fee),
            "has_document_requirements": bool(documents_required),
        }

        points.append(PointStruct(id=s["id"], vector=embedding, payload=payload))

    # ── Upsert to Qdrant ──────────────────────────────────────────────
    client = QdrantClient(url=QDRANT_ENDPOINT, api_key=QDRANT_API_KEY)
    for i in range(0, len(points), UPSERT_BATCH_SIZE):
        batch = points[i : i + UPSERT_BATCH_SIZE]
        client.upsert(collection_name=QDRANT_COLLECTION, points=batch)

    logger.info("Upserted %d points to Qdrant collection '%s'", len(points), QDRANT_COLLECTION)
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
