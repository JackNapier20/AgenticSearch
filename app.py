"""
FastAPI application — serves both the API and the static frontend.

Structure:
    GET  /              → static/index.html  (the UI)
    GET  /health        → liveness check
    GET  /api/schemas   → available schema hints
    POST /api/search    → run the extraction pipeline

Cloud Run deployment:
    uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import run_pipeline
from schemas import SCHEMA_REGISTRY

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agentic Search",
    description="Extract structured entity tables from the web using LLMs.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Topic query string.")
    schema_name: Optional[str] = Field(
        default=None, description="Optional schema hint name (e.g. 'Restaurant')."
    )
    num_results: int = Field(
        default=10, ge=1, le=25, description="Number of web results to retrieve."
    )
    model: str = Field(
        default="gpt-4o-mini", description="OpenAI model to use."
    )


class HealthResponse(BaseModel):
    status: str
    version: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness check — used by Cloud Run."""
    return {"status": "ok", "version": app.version}


@app.get("/api/schemas", tags=["Search"])
async def list_schemas():
    """Return available schema hints and their column definitions."""
    return {
        name: {
            "entity_type": hint.entity_type,
            "columns": hint.columns,
            "description": hint.description,
        }
        for name, hint in SCHEMA_REGISTRY.items()
    }


@app.post("/api/search", tags=["Search"])
async def search(req: SearchRequest):
    """
    Run the agentic search pipeline.

    Returns a challenge-compliant JSON table:
    { query, entity_type, columns, entities, metadata }
    """
    # Validate schema name if provided
    schema_hint = None
    if req.schema_name:
        if req.schema_name not in SCHEMA_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown schema '{req.schema_name}'. "
                       f"Available: {list(SCHEMA_REGISTRY.keys())}",
            )
        schema_hint = SCHEMA_REGISTRY[req.schema_name]

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server.",
        )

    logger.info(f"Search request: query={req.query!r} schema={req.schema_name} n={req.num_results}")

    try:
        result = run_pipeline(
            query=req.query,
            schema_hint=schema_hint,
            num_results=req.num_results,
            llm_model=req.model,
        )
    except Exception as e:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(e))

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Static frontend — served last so API routes always take priority
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the single-page frontend."""
    index = STATIC_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(str(index), media_type="text/html")


# Serve any other static assets (CSS, JS files if added later)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)
