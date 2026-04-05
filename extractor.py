"""
LLM-based entity extraction module.

Strategy:
  1. Infer entity type and relevant columns from the query (fast, cheap call).
  2. Extract candidates per page/passage in parallel (keeps provenance tight,
     avoids giant prompts, and runs all sources concurrently).
  3. Consolidate + deduplicate across all candidates into final EntityRows.
  4. Fallback: if strict parsing fails, use a lenient re-parse before giving up.
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from pydantic import BaseModel

from schema import (
    SearchTableResponse, EntityRow, CellValue,
    schema_to_prompt_description, build_entity_row_prompt_spec,
)
from schemas import get_few_shot_example
from scraper import ScrapedPage

logger = logging.getLogger(__name__)

# Maximum number of scraped pages to send to LLM (search_summary always added on top)
MAX_PAGE_SOURCES = 4
# Maximum parallel LLM calls for per-source extraction
MAX_EXTRACTION_WORKERS = 5
# Character budget per source text
MAX_CHARS_PER_SOURCE = 4000


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
    example = get_few_shot_example(entity_type, columns)

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
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=2000,
        )
        elapsed = time.perf_counter() - t0
        data = json.loads(response.choices[0].message.content)
        candidates = data.get("entities", [])
        logger.info(
            f"  [TIMING] extract {source_url[:55]!r}: {len(candidates)} candidate(s) in {elapsed:.1f}s"
        )
        return candidates
    except Exception as e:
        logger.warning(f"  Extraction failed for {source_url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 3: Consolidate candidates across sources
# ---------------------------------------------------------------------------

def _slim_candidates(candidates: list[dict]) -> list[dict]:
    """
    Strip verbose excerpt strings from each candidate's fields before sending
    to the consolidation LLM, reducing token usage without losing entity values.
    """
    slimmed = []
    for c in candidates:
        sc = {k: v for k, v in c.items() if k != "excerpt"}
        if isinstance(sc.get("fields"), dict):
            sc["fields"] = {
                col: {fk: fv for fk, fv in cell.items() if fk != "excerpt"}
                for col, cell in sc["fields"].items()
                if isinstance(cell, dict)
            }
        slimmed.append(sc)
    return slimmed


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

    # Skip the consolidation LLM call when there are only a few candidates
    if len(all_candidates) <= 3:
        return all_candidates

    slim = _slim_candidates(all_candidates)
    candidates_json = json.dumps(slim, indent=1)
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
        "4. For merged entities, keep the source_url from the best evidence.\n\n"
        "Return a JSON object: {\"entities\": [<consolidated entity rows>]}\n"
        "Each entity must have the same structure: entity_type, fields, summary, relevance.\n\n"
        f"CANDIDATES:\n{candidates_json}"
    )

    try:
        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,
        )
        elapsed = time.perf_counter() - t0
        data = json.loads(response.choices[0].message.content)
        result = data.get("entities", all_candidates)
        logger.info(f"  [TIMING] consolidation: {elapsed:.1f}s → {len(result)} entities")
        return result
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
        reserved = {"entity_type", "summary", "relevance", "source_urls",
                    "relevant_snippets", "attributes", "category", "fields"}

        fields_raw = raw.get("fields", {})
        if not fields_raw:
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
      2. Collect up to MAX_PAGE_SOURCES scraped pages + always the search_summary.
      3. Extract candidates from all sources in parallel (ThreadPoolExecutor).
      4. Consolidate across sources (cheaper: excerpts stripped before sending).
      5. Parse with lenient fallback.

    Args:
        query: The original topic query.
        pages: Scraped pages to extract from.
        search_summary: Raw OpenAI search summary text (always used as a source).
        schema_hint: Optional EntitySchemaHint to override dynamic column inference.
        model: OpenAI model for extraction.

    Returns:
        A SearchTableResponse with populated entities.
    """
    client = OpenAI()

    # --- 1. Infer schema (or use hint) ---
    t_schema = time.perf_counter()
    if schema_hint is not None:
        entity_type = schema_hint.entity_type
        columns = schema_hint.columns
        logger.info(f"Using schema hint: entity_type={entity_type!r}, columns={columns}")
    else:
        logger.info("Inferring entity type and columns from query...")
        entity_type, columns = infer_schema(query, client, model)
        schema_elapsed = time.perf_counter() - t_schema
        logger.info(
            f"  [TIMING] schema_inference: {schema_elapsed:.1f}s  "
            f"entity_type={entity_type!r}, columns={columns}"
        )

    # --- 2. Collect sources (cap scraped pages; always include search_summary) ---
    sources: list[tuple[str, str]] = []

    successful_pages = [p for p in pages if p.success and p.text.strip()]
    for page in successful_pages[:MAX_PAGE_SOURCES]:
        sources.append((page.url, page.text))

    if search_summary and search_summary.strip():
        sources.append(("openai_search_summary", search_summary))

    if not sources:
        logger.warning("No sources available (all scrapes failed and no search_summary).")
        return SearchTableResponse(query=query, entity_type=entity_type, columns=columns)

    logger.info(f"Extracting from {len(sources)} source(s) (up to {MAX_PAGE_SOURCES} pages + summary)...")

    # --- 3. Per-source extraction — parallel ---
    t_extract = time.perf_counter()
    all_candidates: list[dict] = []
    n_workers = min(len(sources), MAX_EXTRACTION_WORKERS)

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(
                _extract_from_source,
                query, entity_type, columns,
                url, text[:MAX_CHARS_PER_SOURCE],
                client, model,
            ): url
            for url, text in sources
        }
        for future in as_completed(futures):
            try:
                candidates = future.result()
                all_candidates.extend(candidates)
            except Exception as e:
                logger.warning(f"  Future failed for {futures[future]}: {e}")

    extraction_elapsed = time.perf_counter() - t_extract
    logger.info(
        f"  [TIMING] extraction ({len(sources)} sources, parallel): "
        f"{extraction_elapsed:.1f}s  raw candidates={len(all_candidates)}"
    )

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
