"""
CLI entry point for the Agentic Search system.

Usage:
    python main.py "AI startups in healthcare"
    python main.py "top pizza places in Brooklyn" --schema Restaurant
    python main.py "open source database tools" --schema SoftwareTool --output results.json
    python main.py "quantum computing companies" -v
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

from pipeline import run_pipeline
from schemas import SCHEMA_REGISTRY


def _make_output_path(query: str) -> str:
    """Generate outputs/<query_slug>/<utc_timestamp>.json"""
    safe_query = re.sub(r"[^a-zA-Z0-9_\-]", "_", query.strip())
    safe_query = re.sub(r"_+", "_", safe_query).strip("_")[:60]
    utc_time = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    folder = os.path.join("outputs", safe_query)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{utc_time}.json")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Agentic Search: extract structured entity tables from the web.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available schema hints (constrain which columns to extract):\n  "
            + "\n  ".join(f"{k}: {v.description}" for k, v in SCHEMA_REGISTRY.items())
            + "\n\nOmit --schema for fully dynamic column inference."
        ),
    )
    parser.add_argument(
        "query",
        help='Topic query, e.g. "AI startups in healthcare"',
    )
    parser.add_argument(
        "--schema", "-s",
        default=None,
        choices=list(SCHEMA_REGISTRY.keys()),
        help="Optional: constrain the columns extracted (default: inferred from query).",
    )
    parser.add_argument(
        "--num-results", "-n",
        type=int,
        default=10,
        help="Number of web search results to retrieve (default: 10).",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Path to write JSON output. If omitted, auto-saved to outputs/<query>/<timestamp>.json",
    )
    parser.add_argument(
        "--model", "-m",
        default="gpt-4o-mini",
        help="OpenAI model for extraction (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )
    parser.add_argument(
        "--print", "-p",
        action="store_true",
        help="Also print the JSON result to stdout (in addition to saving).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Resolve schema hint (if provided)
    schema_hint = SCHEMA_REGISTRY.get(args.schema) if args.schema else None

    # Run pipeline
    try:
        result = run_pipeline(
            query=args.query,
            schema_hint=schema_hint,
            num_results=args.num_results,
            llm_model=args.model,
        )
    except Exception as e:
        logging.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)

    # Determine save path
    save_path = args.output or _make_output_path(args.query)
    output_json = json.dumps(result, indent=2, ensure_ascii=False)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write(output_json)

    if args.print:
        print(output_json)

    # Summary to stderr
    meta = result.get("metadata", {})
    num_entities = meta.get("total_entities", 0)
    sources = meta.get("sources_consulted", 0)
    entity_type = result.get("entity_type", "entities")
    columns = result.get("columns", [])

    print(f"\nResults saved to: {save_path}", file=sys.stderr)
    print(
        f"Found {num_entities} {entity_type}(s) from {sources} source(s). "
        f"Columns: {', '.join(columns)}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
