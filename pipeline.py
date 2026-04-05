"""
Pipeline orchestrator for the Agentic Search system.

Wires together: web search → scraping → LLM extraction.

Key design note: the OpenAI search_summary is passed directly to
extract_entities() so it acts as a high-quality source even when
individual website scrapes fail (JS-rendered pages, bot blocks, etc.).
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Optional

from schema import SearchTableResponse
from search import web_search
from scraper import scrape_urls
from extractor import extract_entities

logger = logging.getLogger(__name__)


def run_pipeline(
    query: str,
    schema_hint=None,   # Optional[EntitySchemaHint] — overrides dynamic column inference
    num_results: int = 10,
    max_chars_per_page: int = 5000,
    llm_model: str = "gpt-4o-mini",
) -> dict:
    """
    Run the full agentic search pipeline.

    Args:
        query: Topic query (e.g., "AI startups in healthcare").
        schema_hint: Optional EntitySchemaHint to constrain which columns to extract.
        num_results: Number of search results to retrieve.
        max_chars_per_page: Max characters to extract per scraped page.
        llm_model: OpenAI model for extraction.

    Returns:
        Challenge-compliant dict: query, entity_type, columns, entities, metadata.
    """
    t_total = time.perf_counter()
    logger.info(f"Starting pipeline for query: '{query}'")

    # --- Step 1: Web Search ---
    logger.info("Step 1/3: Searching the web...")
    t0 = time.perf_counter()
    search_results, search_summary = web_search(query, num_results=num_results)
    search_elapsed = time.perf_counter() - t0
    logger.info(
        f"  [TIMING] search: {search_elapsed:.1f}s  "
        f"urls={len(search_results)}  summary={len(search_summary)} chars"
    )

    # --- Step 2: Scrape Pages ---
    scraped_pages = []
    successful = 0
    if search_results:
        logger.info("Step 2/3: Scraping web pages...")
        t0 = time.perf_counter()
        urls = [r.url for r in search_results]
        scraped_pages = scrape_urls(urls, max_chars=max_chars_per_page)
        scrape_elapsed = time.perf_counter() - t0
        successful = sum(1 for p in scraped_pages if p.success)
        logger.info(
            f"  [TIMING] scrape: {scrape_elapsed:.1f}s  "
            f"ok={successful}/{len(urls)}"
        )
    else:
        logger.warning("No search results; will rely on search_summary only.")

    # --- Step 3: Extract Entities ---
    logger.info("Step 3/3: Extracting entities via LLM...")
    t0 = time.perf_counter()
    result = extract_entities(
        query=query,
        pages=scraped_pages,
        search_summary=search_summary,   # always pass: primary source when pages fail
        schema_hint=schema_hint,
        model=llm_model,
    )
    llm_elapsed = time.perf_counter() - t0
    logger.info(
        f"  [TIMING] llm_total (schema+extraction+consolidation): {llm_elapsed:.1f}s  "
        f"entities={len(result.entities)}"
    )

    total_elapsed = time.perf_counter() - t_total
    logger.info(f"  [TIMING] pipeline total: {total_elapsed:.1f}s")

    # --- Wrap Response ---
    return _wrap_response(result, sources_consulted=successful, timing={
        "search_s": round(search_elapsed, 2),
        "scrape_s": round(scrape_elapsed if search_results else 0.0, 2),
        "llm_s":    round(llm_elapsed, 2),
        "total_s":  round(total_elapsed, 2),
    })


def _wrap_response(
    result: SearchTableResponse,
    sources_consulted: int,
    timing: dict | None = None,
) -> dict:
    """Add metadata and convert to final dict output."""
    output = result.model_dump()
    output["metadata"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sources_consulted": sources_consulted,
        "total_entities": len(result.entities),
        "timing": timing or {},
    }
    return output
