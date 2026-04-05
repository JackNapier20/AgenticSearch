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

The UI lets you enter a query, set the number of results, and see a live entity table with clickable per-cell provenance (source URL + excerpt + confidence). Entity type and columns are inferred automatically from the query — no schema selection needed.

**Browser history:** Previous searches are saved automatically in `localStorage` and shown in a history panel below the results. Clicking any item re-renders the saved table instantly with no network call. History is private to your browser and never stored on the server or in the repository.

---

## Architecture

```
Topic Query
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
┌──────────────────────────────────────────────────────────┐
│  extractor.py                                            │
│                                                          │
│  Step 1: Infer entity_type + columns from query          │  ← gpt-4o-mini
│  Step 2: Extract candidates per source  ─── PARALLEL ──▶ │  ← up to 5 sources concurrently
│          (top 4 scraped pages + search_summary)          │
│  Step 3: Consolidate + deduplicate                       │  ← merge across sources
│  Step 4: Lenient validate + repair                       │  ← never silent-discard
└──────┬───────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  pipeline.py│  Orchestration + per-stage timing + metadata wrapping
└──────┬──────┘
       │
       ├── Web API (app.py) → JSON response to browser
       │
       └── CLI (main.py) → auto-save to outputs/<query>/<utc>.json
```

**Few-shot prompting:** Each extraction call injects a concrete example entity from `schemas.py` that matches the inferred entity type (e.g. "Restaurant" → Lucali example, "AI Startup" → Hugging Face example). This is resolved automatically — no user action required.

**Parallelism:** Per-source LLM extraction calls run concurrently via `ThreadPoolExecutor` (up to 5 workers). This reduces extraction latency from ~N×3s serial to ~1 parallel round (~5–7s for a typical query).

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
    "sources_consulted": 4,
    "total_entities": 8,
    "timing": {
      "search_s": 2.1,
      "scrape_s": 3.4,
      "llm_s": 6.8,
      "total_s": 12.3
    }
  }
}
```

**Key traceability guarantee:** Every `fields[col]` value comes with:
- `source_url` — the exact page the fact was found on
- `excerpt` — verbatim text from that page supporting the value
- `confidence` — per-cell confidence score

**`metadata.timing`** breaks down wall-clock seconds for each pipeline stage — useful for profiling and Cloud Run monitoring.

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

### Web app

```bash
python app.py
# open http://localhost:8080
```

Type any query and hit Search. Entity type and columns are inferred automatically. Previous searches are saved in your browser's `localStorage` and shown in a history panel below the results.

### CLI — fully automatic

The system infers the entity type and columns from the query:

```bash
python main.py "top pizza places in Brooklyn"
python main.py "AI startups in healthcare"
python main.py "open source relational databases"
```

### CLI — with a schema hint (constrain columns)

Use a pre-defined schema to control exactly which attributes are extracted:

```bash
python main.py "top pizza places in Brooklyn" --schema Restaurant
python main.py "AI startups in healthcare" --schema AIStartup
python main.py "open source database tools" --schema SoftwareTool
```

### All CLI options

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

### CLI output location

All CLI results are auto-saved to:
```
outputs/<query_slug>/<UTC_timestamp>.json
```

Example: `outputs/top_pizza_places_in_Brooklyn/20260404T190000Z.json`

The `outputs/` directory is listed in `.gitignore` and will not be committed.

---

## Adding Custom Schemas

Add a new `EntitySchemaHint` to `schemas.py` and register it. Optionally include a concrete example entity to improve few-shot extraction quality:

```python
from schemas import EntitySchemaHint, SCHEMA_REGISTRY

SCHEMA_REGISTRY["Hospital"] = EntitySchemaHint(
    entity_type="Hospital",
    columns=["Name", "Location", "Specialty", "Beds", "Rating", "System"],
    description="A hospital or healthcare facility.",
    examples=[
        {
            "entity_type": "Hospital",
            "fields": {
                "Name":      {"value": "Mass General", "source_url": "https://www.massgeneral.org", "excerpt": "Massachusetts General Hospital", "confidence": 0.99},
                "Location":  {"value": "Boston, MA",   "source_url": "https://www.massgeneral.org", "excerpt": "55 Fruit Street, Boston", "confidence": 0.99},
                "Specialty": {"value": "Academic Medical Center", "source_url": "https://www.massgeneral.org", "excerpt": "world-renowned academic medical center", "confidence": 0.95},
            },
            "summary": "Top-ranked academic hospital in Boston.",
            "relevance": 0.98,
        }
    ],
)
```

Then use it from the CLI:
```bash
python main.py "top hospitals in Boston" --schema Hospital
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **OpenAI Responses API for search** | Native web search returns rich, pre-summarised results including structured snippets and citations — far better than raw scraping alone |
| **search_summary as a primary source** | Restaurant/company websites are often JS-rendered or bot-blocked; the OpenAI search summary already contains clean, factual data about each entity |
| **Parallel per-source extraction** | All sources (up to 4 scraped pages + search_summary) are extracted concurrently via `ThreadPoolExecutor`, reducing latency from N×serial to ~1 parallel round |
| **Source cap (4 pages + summary)** | Processing all 10–25 scraped pages serially was the dominant latency. Capping at 4 preserves coverage while eliminating tail latency. The search_summary always provides broad entity coverage. |
| **Few-shot examples in prompts** | Each extraction call injects a domain-relevant concrete entity example from `schemas.py`. This improves format adherence and output consistency without user intervention. |
| **Consolidation pass** | Merges duplicate entities from multiple sources, selects best evidence, sorts by relevance. Excerpts are stripped before sending to reduce consolidation token cost. |
| **Lenient validation with field repair** | Instead of silently returning `entities=[]` on any schema mismatch, the parser attempts field-by-field repair — preserving partial data rather than discarding everything |
| **Dynamic schema inference** | System infers entity type and columns from the query automatically. Schema hints are optional CLI overrides, not requirements — and are not exposed in the web UI. |
| **CellValue per cell** | Every cell carries `source_url`, `excerpt`, and `confidence`. This makes the table directly auditable without any post-processing |
| **Browser-local history** | Previous search results are stored in `localStorage` only — never sent to the server or written to disk. Private by default. |

---

## Tradeoffs

| Tradeoff | Notes |
|---|---|
| **Multiple LLM calls per run** | Schema inference + per-source extraction (parallel) + consolidation = ~3 LLM roundtrips. More reliable than one giant prompt; parallelism keeps latency acceptable. Uses `gpt-4o-mini` throughout. |
| **Source cap may miss niche data** | Capping at 4 scraped pages can miss a relevant source on long-tail queries. Mitigated by always including the OpenAI search summary which covers the full result set. |
| **Website scraping limited by JS rendering** | Many modern sites are React-rendered and return thin HTML. The search_summary fallback mitigates this significantly. |
| **Dynamic columns are non-deterministic** | Two runs of the same query may produce slightly different column names. Use `--schema` for reproducible structure. |
| **Consolidation can over-merge** | Similar-named entities can get merged incorrectly. Acceptable for MVP; a more sophisticated dedup strategy would use embedding similarity. |

---

## Known Limitations

- JS-rendered pages are not scraped (uses static HTTP fetch). Add Playwright for full JS support.
- OpenAI search summary is English-only and US-biased for local entity queries.
- No caching — re-running the same query re-calls all APIs.

---

## Future Improvements

- [ ] Playwright-based scraping for JS-heavy sites
- [ ] Embedding-based entity deduplication (cosine similarity on name + description)
- [ ] Result caching by query hash
- [ ] Support for multiple output formats (CSV, Markdown table)
- [x] Parallel LLM extraction across sources (done — `ThreadPoolExecutor`)
- [x] FastAPI web app with live entity table (done)
- [x] Browser-local search history (done — `localStorage`)
- [x] Per-stage timing logs (done — `[TIMING]` in logs + `metadata.timing` in output)

---

## Project Structure

```
AgenticSearch/
├── app.py           # FastAPI web app: serves UI + /api/search + /health
├── main.py          # CLI entry point (saves outputs to disk)
├── pipeline.py      # Orchestration: search → scrape → extract → wrap
├── search.py        # Web search via OpenAI Responses API
├── scraper.py       # Concurrent HTTP scraping + HTML → text
├── extractor.py     # 4-step LLM extraction (parallel per-source)
├── schema.py        # Pydantic models: CellValue, EntityRow, SearchTableResponse
├── schemas.py       # Named schema hints + few-shot examples + get_few_shot_example()
├── static/
│   └── index.html   # Single-page frontend (vanilla JS, localStorage history)
├── requirements.txt # Python dependencies
├── .env.example     # API key template
├── .gitignore       # Excludes outputs/, __pycache__/, .env, etc.
└── outputs/         # CLI auto-saved results — gitignored, not committed
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
fastapi>=0.100.0
uvicorn>=0.23.0
```
