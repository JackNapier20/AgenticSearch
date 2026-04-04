"""
LLM-based entity extraction module.

Strategy:
  1. Infer entity type and relevant columns from the query (fast, cheap call).
  2. Extract candidates per page/passage (keeps provenance tight, avoids giant prompts).
  3. Consolidate + deduplicate across all candidates into final EntityRows.
  4. Fallback: if strict parsing fails, use a lenient re-parse before giving up.
"""

from __future__ import annotations

import json
import logging
from typing import Type

from openai import OpenAI
from pydantic import BaseModel

from schema import (
    SearchTableResponse, EntityRow, CellValue,
    schema_to_prompt_description, build_entity_row_prompt_spec,
)
from scraper import ScrapedPage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1: Infer entity type + columns from query
# ---------------------------------------------------------------------------

def infer_schema(query: str, client: OpenAI, model: str) -> tuple[str, list[str]]:
    """
    Ask the LLM what kind of entities the query is about and what columns
    would make a useful table. Returns (entity_type, [col1, col2, ...]).
    """
    prompt = (
        f'Given the search query: "{query}"\n\n'
        "Return a JSON object with:\n"
        '  "entity_type": a short label for what kind of entities will be found '
        '(e.g. "Restaurant", "AI Startup", "Software Tool", "Person")\n'
        '  "columns": a list of 5-8 useful column names for a table of these entities. '
        "Column names should be short, title-case strings like "
        '"Name", "Location", "Founded", "Rating", "Description", "Price Range".\n'
        "The first column MUST be the entity name.\n"
        "Return ONLY the JSON object, no other text."
    )
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    try:
        data = json.loads(response.choices[0].message.content)
        entity_type = data.get("entity_type", "Entity")
        columns = data.get("columns", ["Name", "Description"])
        if not columns:
            columns = ["Name", "Description"]
        return entity_type, columns
    except Exception:
        return "Entity", ["Name", "Description"]


# ---------------------------------------------------------------------------
# Step 2: Extract candidate entities from a single source
# ---------------------------------------------------------------------------

def _extract_from_source(
    query: str,
    entity_type: str,
    columns: list[str],
    source_url: str,
    source_text: str,
    client: OpenAI,
    model: str,
) -> list[dict]:
    """
    Extract candidate entity rows from a single source page.
    Returns a list of raw dicts (not yet validated) for leniency.
    """
    col_list = ", ".join(f'"{c}"' for c in columns)
    example = build_entity_row_prompt_spec(columns)

    prompt = (
        f"You are extracting structured data about {entity_type}s from a web page.\n\n"
        f"QUERY: {query}\n"
        f"SOURCE URL: {source_url}\n\n"
        f"COLUMNS TO EXTRACT: {col_list}\n\n"
        "For EACH entity you find, return a JSON object with:\n"
        '  "entity_type": "<entity type string>"\n'
        '  "fields": a dict mapping each column name to a cell object:\n'
        '    {"value": "<extracted text>", "source_url": "<URL above>", '
        '"excerpt": "<verbatim quote from the text below>", "confidence": 0.9}\n'
        '  "summary": "<one sentence why relevant to query>"\n'
        '  "relevance": <float 0-1>\n\n'
        f"EXAMPLE of one entity:\n{example}\n\n"
        "Return a JSON object: {\"entities\": [<entity1>, <entity2>, ...]}\n"
        "Only include entities clearly relevant to the query. "
        "If no relevant entities are found, return {\"entities\": []}.\n\n"
        f"SOURCE TEXT:\n{source_text}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        data = json.loads(response.choices[0].message.content)
        candidates = data.get("entities", [])
        logger.debug(f"  Source {source_url[:60]}: got {len(candidates)} candidate(s)")
        return candidates
    except Exception as e:
        logger.warning(f"  Extraction failed for {source_url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 3: Consolidate candidates across sources
# ---------------------------------------------------------------------------

def _consolidate(
    query: str,
    entity_type: str,
    columns: list[str],
    all_candidates: list[dict],
    client: OpenAI,
    model: str,
) -> list[dict]:
    """
    Merge and deduplicate candidate entities from multiple sources.
    Returns consolidated raw dicts.
    """
    if not all_candidates:
        return []

    # If only a few, skip the consolidation LLM call
    if len(all_candidates) <= 3:
        return all_candidates

    candidates_json = json.dumps(all_candidates, indent=1)
    col_list = ", ".join(f'"{c}"' for c in columns)

    prompt = (
        f"You are consolidating extracted {entity_type} data from multiple web sources.\n\n"
        f"QUERY: {query}\n"
        f"COLUMNS: {col_list}\n\n"
        "Below are candidate entity rows extracted from different pages. "
        "Some may be duplicates or near-duplicates (same entity, different sources).\n\n"
        "Your job:\n"
        "1. Merge duplicates into single rows, preserving the best source evidence.\n"
        "2. Remove irrelevant entries.\n"
        "3. Sort by relevance (highest first).\n"
        "4. For merged entities, keep the source_url and excerpt from the best evidence.\n\n"
        "Return a JSON object: {\"entities\": [<consolidated entity rows>]}\n"
        "Each entity must have the same structure: entity_type, fields, summary, relevance.\n\n"
        f"CANDIDATES:\n{candidates_json}"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4000,
        )
        data = json.loads(response.choices[0].message.content)
        return data.get("entities", all_candidates)
    except Exception as e:
        logger.warning(f"Consolidation failed ({e}), using unmerged candidates")
        return all_candidates


# ---------------------------------------------------------------------------
# Step 4: Parse + validate with lenient fallback
# ---------------------------------------------------------------------------

def _parse_entities_lenient(raw_entities: list[dict]) -> list[EntityRow]:
    """
    Parse raw entity dicts into EntityRow objects.
    Accepts partial data and logs individual field failures rather than
    discarding the whole entity.
    """
    parsed: list[EntityRow] = []
    for i, raw in enumerate(raw_entities):
        try:
            row = EntityRow.model_validate(raw)
            parsed.append(row)
        except Exception as e:
            logger.warning(f"Entity {i} failed strict validate ({e}), trying field-by-field repair")
            row = _repair_entity(raw, i)
            if row:
                parsed.append(row)
    return parsed


def _repair_entity(raw: dict, idx: int) -> EntityRow | None:
    """
    Try to salvage an entity that failed strict validation by
    building the fields dict manually from whatever is present.
    """
    try:
        # Determine fields
        reserved = {"entity_type", "summary", "relevance", "source_urls",
                    "relevant_snippets", "attributes", "category", "fields"}

        fields_raw = raw.get("fields", {})
        if not fields_raw:
            # Try extracting from top-level keys
            for k, v in raw.items():
                if k not in reserved and isinstance(v, dict):
                    fields_raw[k] = v

        fields: dict[str, CellValue] = {}
        for col, cell in fields_raw.items():
            if not isinstance(cell, dict):
                cell = {"value": str(cell), "source_url": "", "excerpt": ""}
            try:
                fields[col] = CellValue.model_validate(cell)
            except Exception:
                # Last resort: build a minimal CellValue
                fields[col] = CellValue(
                    value=str(cell.get("value", cell)),
                    source_url=cell.get("source_url", cell.get("url", "")),
                    excerpt=cell.get("excerpt", cell.get("snippet", cell.get("text", ""))),
                )

        if not fields:
            logger.warning(f"  Entity {idx}: no usable fields found, skipping")
            return None

        return EntityRow(
            entity_type=str(raw.get("entity_type", raw.get("category", "Entity"))),
            fields=fields,
            summary=str(raw.get("summary", "")),
            relevance=float(raw.get("relevance", 0.5)),
        )
    except Exception as e:
        logger.error(f"  Entity {idx}: repair also failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Public extract function
# ---------------------------------------------------------------------------

def extract_entities(
    query: str,
    pages: list[ScrapedPage],
    search_summary: str = "",
    schema_hint=None,   # Optional[EntitySchemaHint] — overrides dynamic inference
    entity_model=None,  # kept for API compatibility, ignored
    model: str = "gpt-4o-mini",
) -> SearchTableResponse:
    """
    Extract structured entities from scraped pages (and search summary) using an LLM.

    Strategy:
      1. Infer entity_type + columns from query (or use schema_hint if provided).
      2. Extract candidates per source (page text and/or search_summary).
      3. Consolidate across sources.
      4. Parse with lenient fallback.

    Args:
        query: The original topic query.
        pages: Scraped pages to extract from.
        search_summary: Raw OpenAI search summary text (used as a source if pages are thin).
        schema_hint: Optional EntitySchemaHint to override dynamic column inference.
        model: OpenAI model for extraction.

    Returns:
        A SearchTableResponse with populated entities.
    """
    client = OpenAI()

    # --- 1. Infer schema (or use hint) ---
    if schema_hint is not None:
        entity_type = schema_hint.entity_type
        columns = schema_hint.columns
        logger.info(f"Using schema hint: entity_type={entity_type!r}, columns={columns}")
    else:
        logger.info("Inferring entity type and columns from query...")
        entity_type, columns = infer_schema(query, client, model)
        logger.info(f"  entity_type={entity_type!r}, columns={columns}")

    # --- 2. Collect sources ---
    # Use scraped pages that have content; always include search_summary as a source
    sources: list[tuple[str, str]] = []  # (url, text)

    successful_pages = [p for p in pages if p.success and p.text.strip()]
    for page in successful_pages:
        sources.append((page.url, page.text))

    if search_summary and search_summary.strip():
        sources.append(("openai_search_summary", search_summary))

    if not sources:
        logger.warning("No sources available (all scrapes failed and no search_summary).")
        return SearchTableResponse(query=query, entity_type=entity_type, columns=columns)

    logger.info(f"Extracting from {len(sources)} source(s) (pages + summary)...")

    # --- 3. Per-source extraction ---
    all_candidates: list[dict] = []
    for url, text in sources:
        candidates = _extract_from_source(
            query=query,
            entity_type=entity_type,
            columns=columns,
            source_url=url,
            source_text=text[:6000],   # cap per-source to avoid token limits
            client=client,
            model=model,
        )
        all_candidates.extend(candidates)

    logger.info(f"Total raw candidates: {len(all_candidates)}")

    # --- 4. Consolidate ---
    if len(sources) > 1:
        logger.info("Consolidating and deduplicating...")
        all_candidates = _consolidate(query, entity_type, columns, all_candidates, client, model)
        logger.info(f"After consolidation: {len(all_candidates)} entities")

    # --- 5. Parse with lenient fallback ---
    entity_rows = _parse_entities_lenient(all_candidates)
    logger.info(f"Successfully parsed {len(entity_rows)} entities")

    return SearchTableResponse(
        query=query,
        entity_type=entity_type,
        columns=columns,
        entities=entity_rows,
    )
