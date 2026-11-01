"""
Google Maps Lead Generator — main entry point.

Scrapes local business leads from Google Maps via Serper API, optionally enriches
them with emails/social media via the configured LLM provider (default: Groq), saves to Excel, and
persists to pgvector for semantic retrieval by the Valgist marketing factory.

Usage:
    python3 main.py --location "Toronto" --query "dentists" --pages 1 [--store-pgvector]
    python3 main.py --location "Toronto" --query "dentists" --pages 1 --no-enrich
    python3 main.py --store-only --query "plumbers in Toronto"  # pgvector search only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

from src.places_api import search_places, get_coordinates
from src.business_info import process_businesses
from src.data_export import save_places_to_excel
from src.utils import get_current_date


# ────────────────────────── Helpers ──────────────────────────

def _extract_places(raw_response: Any) -> List[Dict[str, Any]]:
    """Extract a flat list of place dicts from Serper's response."""
    places: List[Dict[str, Any]] = []
    if isinstance(raw_response, list):
        for page in raw_response:
            if isinstance(page, dict) and "places" in page:
                places.extend(page["places"])
    elif isinstance(raw_response, dict) and "places" in raw_response:
        places = raw_response["places"]
    return places


def _fallback_coords(location: str) -> Optional[Dict[str, float]]:
    """Hardcoded coords for common cities when Nominatim is blocked."""
    fallback = {
        "toronto": {"lat": 43.6532, "lon": -79.3832},
        "new york": {"lat": 40.7128, "lon": -74.0060},
        "london": {"lat": 51.5074, "lon": -0.1278},
        "vancouver": {"lat": 49.2827, "lon": -123.1207},
        "calgary": {"lat": 51.0447, "lon": -114.0719},
        "montreal": {"lat": 45.5017, "lon": -73.5673},
        "sydney": {"lat": -33.8688, "lon": 151.2093},
        "berlin": {"lat": 52.5200, "lon": 13.4050},
        "paris": {"lat": 48.8566, "lon": 2.3522},
    }
    return fallback.get(location.strip().lower())


# ────────────────────────── Main ──────────────────────────

async def main(
    location: str,
    search_query: str,
    num_pages: int = 1,
    enrich: bool = True,
    store_pgvector: bool = False,
) -> Dict[str, Any]:
    """
    Orchestrate lead generation: scrape → (enrich) → Excel → pgvector.

    Returns a result dict with counts and any errors.
    """
    print(f"\n🔍 Starting lead generation for '{search_query}' in '{location}'")
    result: Dict[str, Any] = {
        "location": location,
        "query": search_query,
        "pages": num_pages,
        "places_scraped": 0,
        "enriched": 0,
        "pgvector_stored": 0,
        "errors": [],
    }

    # ── Step 1: Coordinates ──────────────────────────────────
    coords = get_coordinates(location)
    if not coords:
        print(f"⚠️  Nominatim geocoding failed for '{location}' — using fallback coords.")
        coords = _fallback_coords(location)
        if not coords:
            err = f"Could not get coordinates for '{location}' and no fallback available."
            print(f"❌ {err}")
            result["errors"].append(err)
            return result
        print(f"✓ Using fallback coordinates: {coords}")

    # ── Step 2: Serper Maps search ───────────────────────────
    places_data = search_places(search_query, coords, num_pages)
    if not places_data:
        err = "No places found from Serper API."
        print(f"❌ {err}")
        result["errors"].append(err)
        return result

    places = _extract_places(places_data)
    result["places_scraped"] = len(places)
    print(f"✓ Scraped {len(places)} places from Serper Maps")

    # Show first few
    for p in places[:3]:
        print(f"   • {p.get('title','')} — {p.get('address','')} — {p.get('phoneNumber','')}")

    # ── Step 3: Save to Excel ────────────────────────────────
    excel_filename = f"data_{search_query}_{location}_{get_current_date()}.xlsx"
    try:
        save_places_to_excel(places_data, excel_filename)
        print(f"✓ Saved to Excel: {excel_filename}")
    except Exception as e:
        err = f"Excel save failed: {e}"
        print(f"❌ {err}")
        result["errors"].append(err)

    # ── Step 4: Enrich with emails/social (needs OpenAI key) ─
    if enrich:
        provider = os.getenv("LLM_PROVIDER", "groq")
        model = os.getenv("LLM_MODEL", "qwen/qwen3.8-27b")
        print(f"\n🤖 Enriching businesses with emails & social media via {provider} ({model})...")
        try:
            await process_businesses(excel_filename)
            result["enriched"] = len(places)
            print("✓ Enrichment complete")
        except Exception as e:
            err = f"Enrichment failed (OpenAI key may be missing): {e}"
            print(f"⚠️  {err}")
            result["errors"].append(err)
    else:
        print("⏭️  Skipping enrichment (--no-enrich flag set)")

    # ── Step 5: Persist to pgvector ──────────────────────────
    if store_pgvector:
        print("\n🗄️  Storing leads to pgvector for semantic retrieval...")
        try:
            from src.pgvector_store import store_leads
            pg_result = store_leads(
                places,
                source="serper-maps",
                location=location,
                query=search_query,
            )
            result["pgvector_stored"] = pg_result.get("stored", 0)
            result["pgvector_errors"] = pg_result.get("errors", [])
            if pg_result.get("stored", 0) > 0:
                print(f"✓ Stored {pg_result['stored']} leads to pgvector")
            if pg_result.get("errors"):
                for e in pg_result["errors"]:
                    print(f"   ⚠️  {e}")
        except ImportError as e:
            err = f"pgvector store import failed: {e}"
            print(f"❌ {err}")
            result["errors"].append(err)

    print(f"\n✅ Done: {result['places_scraped']} scraped, "
          f"{result['pgvector_stored']} in pgvector")
    return result


# ────────────────────────── CLI ──────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Google Maps Lead Generator")
    p.add_argument("--location", help="City/area to search (e.g. Toronto)")
    p.add_argument("--query", help="Business type (e.g. dentists, plumbers)")
    p.add_argument("--pages", type=int, default=1, help="Pages of results (20 per page)")
    p.add_argument("--no-enrich", action="store_true",
                   help="Skip AI email/social enrichment (no OpenAI key needed)")
    p.add_argument("--store-pgvector", action="store_true",
                   help="Also persist leads to pgvector for semantic search")
    p.add_argument("--search-pgvector", metavar="QUERY",
                   help="Semantic-search stored leads in pgvector instead of scraping")
    p.add_argument("--limit", type=int, default=10, help="Max pgvector search results")
    return p


def _run_pgvector_search(query: str, limit: int) -> None:
    from src.pgvector_store import search_leads_semantic
    results = search_leads_semantic(query, limit=limit)
    print(f"\n🔎 pgvector semantic search: '{query}' ({len(results)} results)\n")
    for i, lead in enumerate(results, 1):
        print(f"{i}. {lead.get('name','')} (sim={lead.get('_similarity',0):.3f})")
        print(f"   {lead.get('address','')} | {lead.get('phone','')} | {lead.get('website','')}")
        print(f"   category={lead.get('category','')} rating={lead.get('rating','')}")
        print()


def main_cli() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.search_pgvector:
        _run_pgvector_search(args.search_pgvector, args.limit)
        return

    if not args.location or not args.query:
        parser.error("--location and --query are required when not using --search-pgvector")

    asyncio.run(main(
        location=args.location,
        search_query=args.query,
        num_pages=args.pages,
        enrich=not args.no_enrich,
        store_pgvector=args.store_pgvector,
    ))


if __name__ == "__main__":
    main_cli()
