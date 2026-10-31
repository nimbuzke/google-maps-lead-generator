"""
pgvector_store — Persist scraped Google Maps leads into pgvector for semantic retrieval.

Stores each lead as a vector embedding in the `lead_embeddings` table so leads can be
searched later by meaning (e.g. "dentists in Toronto", "plumbers with websites").
Works with or without OpenAI — falls back to deterministic pseudo-embeddings.

Usage:
    from src.pgvector_store import store_leads
    store_leads(places_list, source="serper-maps")
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add the valgist factory-controller to path so we can import vector-service
FACTORY_CTRL = Path(
    os.environ.get("VALGIST_REPO_PATH",
                   "/home/sage/Documents/Automation/marketing/valgist-marketing-engine")
) / "tools" / "factory-controller"
sys.path.insert(0, str(FACTORY_CTRL))

import importlib.util
_VS_PATH = FACTORY_CTRL / "vector-service.py"
_VS_SPEC = importlib.util.spec_from_file_location("vector_service", _VS_PATH)
_VS_MODULE = importlib.util.module_from_spec(_VS_SPEC)
sys.modules["vector_service"] = _VS_MODULE
_VS_SPEC.loader.exec_module(_VS_MODULE)

VectorService = _VS_MODULE.VectorService


def _normalize_lead(lead: Dict[str, Any]) -> Dict[str, Any]:
    """Map Serper API fields to canonical lead fields."""
    return {
        "name": lead.get("title") or lead.get("name", ""),
        "address": lead.get("address", ""),
        "phone": lead.get("phoneNumber") or lead.get("phone", ""),
        "website": lead.get("website", ""),
        "category": lead.get("type", "") or lead.get("category", ""),
        "keywords": " || ".join(lead.get("types", [])) if lead.get("types") else lead.get("keywords", ""),
        "rating": lead.get("rating", ""),
        "reviews": lead.get("ratingCount", "") or lead.get("reviews", ""),
        "email": lead.get("email", ""),
        "facebook": lead.get("facebook", ""),
        "instagram": lead.get("instagram", ""),
        "twitter": lead.get("twitter", ""),
        "opening_hours": lead.get("openingHours", ""),
        "place_id": lead.get("placeId", "") or lead.get("place_id", ""),
    }


def _lead_to_text(lead: Dict[str, Any]) -> str:
    """Flatten a lead dict into searchable text."""
    parts = [
        f"name: {lead.get('name', '')}",
        f"address: {lead.get('address', '')}",
        f"phone: {lead.get('phone', '')}",
        f"website: {lead.get('website', '')}",
        f"category: {lead.get('category', '')}",
        f"keywords: {lead.get('keywords', '')}",
        f"rating: {lead.get('rating', '')}",
        f"reviews: {lead.get('reviews', '')}",
        f"email: {lead.get('email', '')}",
        f"facebook: {lead.get('facebook', '')}",
        f"instagram: {lead.get('instagram', '')}",
        f"twitter: {lead.get('twitter', '')}",
    ]
    return " | ".join(p for p in parts if p)


def store_leads(
    places: List[Dict[str, Any]],
    source: str = "serper-maps",
    location: str = "",
    query: str = "",
) -> Dict[str, Any]:
    """
    Store a list of scraped leads into pgvector lead_embeddings.

    Each lead is embedded and stored with its metadata. Returns a summary dict.

    Args:
        places: List of lead dicts (name, address, phone, website, etc.)
        source: Source label (e.g. "serper-maps")
        location: Search location (for metadata)
        query: Search query (for metadata)

    Returns:
        Dict with stored count, ids, and any errors.
    """
    vs = VectorService()
    if not vs.available:
        return {
            "stored": 0,
            "errors": ["pgvector not available — leads not persisted"],
            "ids": [],
        }

    stored = 0
    ids: List[int] = []
    errors: List[str] = []

    for lead in places:
        try:
            # Normalize Serper fields to canonical names
            norm = _normalize_lead(lead)
            lead_id = f"{source}:{location}:{query}:{norm.get('name','')}:{norm.get('address','')}"
            lead_id = lead_id.replace(" ", "_")[:120]

            text = _lead_to_text(norm)
            lead_data = {
                **norm,
                "source": source,
                "location": location,
                "query": query,
            }

            row_id = vs.store_lead(lead_id, lead_data, source=source)
            if row_id is not None:
                stored += 1
                ids.append(row_id)
            else:
                errors.append(f"Failed to store: {lead.get('name','unknown')}")
        except Exception as e:
            errors.append(f"Error storing {lead.get('name','unknown')}: {e}")

    return {
        "stored": stored,
        "total": len(places),
        "ids": ids,
        "errors": errors,
    }


def search_leads_semantic(
    query: str,
    limit: int = 10,
    min_similarity: float = 0.3,
) -> List[Dict[str, Any]]:
    """
    Semantic search over stored leads.

    Args:
        query: Natural language query (e.g. "dentists in Toronto")
        limit: Max results
        min_similarity: Minimum cosine similarity (0-1)

    Returns:
        List of lead dicts with similarity score.
    """
    vs = VectorService()
    if not vs.available:
        return []

    results = vs.search_leads(query, limit=limit, min_similarity=min_similarity)
    leads: List[Dict[str, Any]] = []
    for r in results:
        try:
            data = json.loads(r.content)
            data["_similarity"] = r.similarity
            data["_lead_id"] = r.metadata.get("lead_id")
            leads.append(data)
        except Exception:
            continue

    # Keyword fallback when deterministic embeddings return no semantic matches
    if not leads:
        leads = _keyword_search_leads(query, limit)

    return leads


def _keyword_search_leads(query: str, limit: int) -> List[Dict[str, Any]]:
    """Fallback keyword search using ILIKE on lead_data JSONB when pgvector
    semantic search returns empty (e.g. deterministic embeddings)."""
    import psycopg2

    conn = _pg_connect()
    if conn is None:
        return []

    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, lead_id, lead_data, source, created_at
                FROM lead_embeddings
                WHERE lead_data::text ILIKE %s
                LIMIT %s
                """,
                (f"%{query}%", limit),
            )
            rows = cur.fetchall()
    except Exception:
        rows = []
    finally:
        conn.close()

    leads: List[Dict[str, Any]] = []
    for row in rows:
        try:
            data = row[2]
            if isinstance(data, str):
                data = json.loads(data)
            data["_similarity"] = 0.5  # keyword match baseline
            data["_lead_id"] = row[1]
            leads.append(data)
        except Exception:
            continue
    return leads


def _pg_connect():
    """Direct psycopg2 connection for keyword fallback."""
    try:
        import psycopg2
    except ImportError:
        return None
    try:
        return psycopg2.connect(
            host=os.environ.get("HERMES_PGVECTOR_HOST", "127.0.0.1"),
            port=int(os.environ.get("HERMES_PGVECTOR_PORT", "5433")),
            dbname=os.environ.get("HERMES_PGVECTOR_DB", "valgist"),
            user=os.environ.get("HERMES_PGVECTOR_USER", "valgist"),
            password=os.environ.get("HERMES_PGVECTOR_PASSWORD", "valgist"),
            connect_timeout=5,
        )
    except Exception:
        return None



def match_lead_to_icp(
    lead: Dict[str, Any],
    icp_id: str = "default",
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """
    Score how well a lead matches stored ICP profiles.

    Args:
        lead: Lead dict to score
        icp_id: ICP profile id to match against
        limit: Max results

    Returns:
        List of ICP match dicts with similarity.
    """
    vs = VectorService()
    if not vs.available:
        return []

    # First ensure an ICP profile exists
    vs.store_icp_profile(icp_id, {
        "industry": lead.get("category", ""),
        "location": lead.get("address", ""),
        "has_website": bool(lead.get("website")),
        "has_phone": bool(lead.get("phone")),
        "source": "google-maps-leads",
    })

    results = vs.match_icp(lead, limit=limit)
    matches: List[Dict[str, Any]] = []
    for r in results:
        try:
            data = json.loads(r.content)
            data["_similarity"] = r.similarity
            data["_icp_id"] = r.metadata.get("icp_id")
            matches.append(data)
        except Exception:
            continue
    return matches
