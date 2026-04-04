# Agentic Search Pipeline

> **Take a topic query → get a structured, source-traceable entity table from the web.**

---

## Overview

Agentic Search is a multi-stage pipeline that accepts a free-form topic query, searches the web using OpenAI's native web search tool, scrapes and processes result pages, uses an LLM to extract structured entity data, and returns a challenge-compliant JSON table where every cell carries a source URL and verbatim evidence.

Runs as both a **CLI tool** and a **FastAPI web app** (single service, Cloud Run ready).

**Example queries:**
```
"AI startups in healthcare"
"top pizza places in Brooklyn"
"open source vector databases"
"quantum computing companies raising Series B"
```

---

## Running the Web App (Recommended)

### 1. Install dependencies

```bash
conda activate AgenticSearch
pip install -r requirements.txt
```

### 2. Set your API key

```bash
export OPENAI_API_KEY=sk-...
# or add it to a .env file
```

### 3. Start the server

```bash
python app.py
# or: uvicorn app:app --reload --port 8080
```

### 4. Open in browser

```
http://localhost:8080
```

The UI lets you enter a query, pick an optional schema, and see a live entity table with clickable per-cell provenance (source URL + excerpt + confidence).

---

---

## Architecture

```
Query + Schema Hint (optional)
       │
       ▼
┌─────────────┐
│  search.py  │  OpenAI Responses API + web_search_preview tool
│             │  → list of (URL, title, snippet) + rich search_summary
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  scraper.py │  Concurrent HTTP fetch → BeautifulSoup text extraction
│             │  → ScrapedPage(url, title, text, success)
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  extractor.py                                    │
│                                                  │
│  Step 1: Infer entity_type + columns from query  │  ← gpt-4o-mini (cheap)
│  Step 2: Extract candidates per source           │  ← one call per page/summary
│  Step 3: Consolidate + deduplicate               │  ← merge across sources
│  Step 4: Lenient validate + repair               │  ← never silent-discard
└──────┬───────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  pipeline.py│  Orchestration + metadata wrapping
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  main.py    │  CLI + auto-save to outputs/<query>/<utc>.json
└─────────────┘
```

---

## Output Format

Every run produces a challenge-compliant JSON table:

```json
{
  "query": "top pizza places in Brooklyn",
  "entity_type": "Restaurant",
  "columns": ["Name", "Neighborhood", "Cuisine", "Price Range", "Rating", "Signature Dish"],
  "entities": [
    {
      "entity_type": "Restaurant",
      "fields": {
        "Name": {
          "value": "Di Fara Pizza",
          "source_url": "https://www.difarapizzany.com/",
          "excerpt": "Dom built Di Fara on premium ingredients and recipes perfected over decades.",
          "confidence": 0.95
        },
        "Neighborhood": {
          "value": "Midwood",
          "source_url": "https://www.difarapizzany.com/",
          "excerpt": "1424 Avenue J, Brooklyn, NY 11230",
          "confidence": 1.0
        }
      },
      "summary": "A legendary Brooklyn pizzeria established in 1965, widely considered one of NYC's best.",
      "relevance": 0.97
    }
  ],
  "metadata": {
    "timestamp": "2026-04-04T19:00:00Z",
    "sources_consulted": 6,
    "total_entities": 8
  }
}
```

**Key traceability guarantee:** Every `fields[col]` value comes with:
- `source_url` — the exact page the fact was found on
- `excerpt` — verbatim text from that page supporting the value
- `confidence` — per-cell confidence score

---

## Setup

### 1. Prerequisites

- Python 3.11+
- Conda (recommended) or any virtualenv

### 2. Create and activate the environment

```bash
# Create the conda environment
conda create -n AgenticSearch python=3.11

# Activate it
conda activate AgenticSearch

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure your OpenAI API key

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Or export it directly:
```bash
export OPENAI_API_KEY=sk-...
```

---

## Usage

### Basic — fully automatic

The system infers the entity type and columns from the query:

```bash
python main.py "top pizza places in Brooklyn"
python main.py "AI startups in healthcare"
python main.py "open source relational databases"
```

### With a schema hint — constrain columns

Use a pre-defined schema to control exactly which attributes are extracted:

```bash
python main.py "top pizza places in Brooklyn" --schema Restaurant
python main.py "AI startups in healthcare" --schema AIStartup
python main.py "open source database tools" --schema SoftwareTool
```

### All options

```
python main.py QUERY [options]

Positional:
  query                 Topic query string

Options:
  --schema, -s NAME     Constrain columns (AIStartup|Restaurant|SoftwareTool|ResearchPaper)
  --num-results, -n N   Number of web results to retrieve [default: 10]
  --output, -o PATH     Custom output path [default: outputs/<query>/<timestamp>.json]
  --model, -m MODEL     OpenAI model [default: gpt-4o-mini]
  --verbose, -v         Enable debug logging
  --print, -p           Print the JSON to stdout as well as saving
```

### Output location

All results are auto-saved to:
```
outputs/<query_slug>/<UTC_timestamp>.json
```

Example: `outputs/top_pizza_places_in_Brooklyn/20260404T190000Z.json`

---

## Adding Custom Schemas

Add a new `EntitySchemaHint` to `schemas.py` and register it:

```python
from schemas import EntitySchemaHint, SCHEMA_REGISTRY

SCHEMA_REGISTRY["Hospital"] = EntitySchemaHint(
    entity_type="Hospital",
    columns=["Name", "Location", "Specialty", "Beds", "Rating", "System"],
    description="A hospital or healthcare facility.",
)
```

Then use it:
```bash
python main.py "top hospitals in Boston" --schema Hospital
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **OpenAI Responses API for search** | Native web search returns rich, pre-summarised results including structured snippets and citations — far better than raw scraping alone |
| **search_summary as a primary source** | Restaurant/company websites are often JS-rendered or bot-blocked; the OpenAI search summary already contains clean, factual data about each entity |
| **Per-source candidate extraction** | Avoids one giant context window. Each page/source is extracted separately, keeping provenance tight and reducing hallucination |
| **Consolidation pass** | Merges duplicate entities from multiple sources, selects best evidence, sorts by relevance |
| **Lenient validation with field repair** | Instead of silently returning `entities=[]` on any schema mismatch, the parser attempts field-by-field repair — preserving partial data rather than discarding everything |
| **Dynamic schema inference** | System infers entity type and columns from the query automatically. Schema hints are optional overrides, not requirements |
| **CellValue per cell** | Every cell carries `source_url`, `excerpt`, and `confidence`. This makes the table directly auditable without any post-processing |
| **`confidence` field** | Signals extraction certainty per-cell, enabling downstream filtering |

---

## Tradeoffs

| Tradeoff | Notes |
|---|---|
| **Multiple LLM calls per run** | Per-source extraction + consolidation = N+2 calls. More reliable than one giant prompt but costs more tokens. Mitigation: use `gpt-4o-mini` |
| **Website scraping limited by JS rendering** | Many modern sites (Yelp, Roberta's, etc.) are React-rendered and return thin HTML. The search_summary fallback mitigates this significantly |
| **Dynamic columns are non-deterministic** | Two runs of the same query may produce slightly different column names. Use `--schema` for reproducible structure |
| **Consolidation can over-merge** | Similar-named entities can get merged incorrectly. Acceptable for MVP; a more sophisticated dedup strategy would use embedding similarity |

---

## Known Limitations

- JS-rendered pages are not scraped (uses static HTTP fetch). Add Playwright for full JS support.
- OpenAI search summary is English-only and US-biased for local entity queries.
- Consolidation LLM call can be slow for large candidate sets (>20 entities).
- No caching — re-running the same query re-calls all APIs.

---

## Future Improvements

- [ ] Playwright-based scraping for JS-heavy sites
- [ ] Embedding-based entity deduplication (cosine similarity on name + description)
- [ ] Result caching by query hash
- [ ] REST API wrapper (FastAPI) with streaming SSE output
- [ ] Interactive HTML table output with clickable source links
- [ ] Support for multiple output formats (CSV, Markdown table)
- [ ] Async pipeline for parallel LLM calls across sources

---

## Project Structure

```
AgenticSearch/
├── main.py          # CLI entry point
├── pipeline.py      # Orchestration: search → scrape → extract → output
├── search.py        # Web search via OpenAI Responses API
├── scraper.py       # Concurrent HTTP scraping + HTML → text
├── extractor.py     # 4-step LLM extraction strategy
├── schema.py        # Pydantic models: CellValue, EntityRow, SearchTableResponse
├── schemas.py       # Named schema hints (Restaurant, AIStartup, etc.)
├── requirements.txt # Python dependencies
├── .env.example     # API key template
└── outputs/         # Auto-saved results (query/timestamp.json)
```

---

## Requirements

```
openai>=1.0.0
pydantic>=2.0.0
requests>=2.31.0
beautifulsoup4>=4.12.0
lxml>=5.0.0
python-dotenv>=1.0.0
```
