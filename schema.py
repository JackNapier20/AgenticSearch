"""
Pydantic response schemas for the Agentic Search pipeline.

Design:
- CellValue: one traceable extracted fact (value + source url + evidence)
- EntityRow: a single discovered entity as a dict of field_name -> CellValue
- SearchTableResponse: the full challenge-compliant table output

The schema is deliberately lenient on parsing (aliases for common LLM mistakes)
so that validation fixes rather than discards partial data.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Building-block: one traceable cell
# ---------------------------------------------------------------------------

class CellValue(BaseModel):
    """
    A single extracted value with full provenance.

    Accepts 'snippet' or 'text' as aliases for 'excerpt' since LLMs
    sometimes use those field names despite being instructed otherwise.
    """

    value: str = Field(..., description="The extracted value as a plain string.")
    source_url: str = Field(..., description="URL of the page this value was taken from.")
    excerpt: str = Field(
        default="",
        description="Verbatim text from the source page that supports this value.",
        alias="excerpt",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (0–1). Use 1.0 if not uncertain.",
    )

    model_config = {"populate_by_name": True}

    @model_validator(mode="before")
    @classmethod
    def remap_aliases(cls, data: Any) -> Any:
        """Accept 'snippet' or 'text' in place of 'excerpt'."""
        if isinstance(data, dict):
            if "excerpt" not in data or not data.get("excerpt"):
                data["excerpt"] = data.get("snippet") or data.get("text") or ""
            # Normalise source_url <- url if needed
            if "source_url" not in data and "url" in data:
                data["source_url"] = data.pop("url")
        return data

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value_to_str(cls, v: Any) -> str:
        """Allow int/float/list values and coerce to string."""
        if isinstance(v, list):
            return ", ".join(str(i) for i in v)
        return str(v) if v is not None else ""


# ---------------------------------------------------------------------------
# Entity row: one discovered entity
# ---------------------------------------------------------------------------

class EntityRow(BaseModel):
    """
    A single discovered entity, structured as a table row.

    `fields` maps column names to CellValue objects, making every
    cell independently traceable.
    """

    entity_type: str = Field(
        default="",
        description="Type/category of this entity (e.g. 'Restaurant', 'Startup').",
    )
    fields: Dict[str, CellValue] = Field(
        ...,
        description=(
            "Mapping of column name to cell value. Each column must be a CellValue "
            "with value, source_url, and excerpt."
        ),
    )
    summary: str = Field(
        default="",
        description="One-sentence summary of why this entity is relevant to the query.",
    )
    relevance: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "How relevant this entity is to the query. "
            "0.0–0.25: marginal, 0.25–0.5: partial, 0.5–0.75: moderate, 0.75–1: high."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def flatten_legacy_format(cls, data: Any) -> Any:
        """
        Accept the legacy flat format where top-level keys are field names
        (e.g. {"name": {...}, "description": {...}, ...}) and convert to
        {"fields": {"name": {...}, "description": {...}}}.
        """
        if isinstance(data, dict) and "fields" not in data:
            reserved = {"entity_type", "summary", "relevance", "source_urls",
                        "relevant_snippets", "attributes", "category"}
            fields: Dict[str, Any] = {}
            top_level: Dict[str, Any] = {}
            for k, v in data.items():
                if k in reserved:
                    top_level[k] = v
                elif isinstance(v, dict) and ("value" in v or "source_url" in v or "url" in v):
                    fields[k] = v
            if fields:
                top_level["fields"] = fields
                # Carry category -> entity_type
                if "category" in top_level and not top_level.get("entity_type"):
                    cat = top_level.pop("category")
                    if isinstance(cat, list):
                        cat = cat[0].get("type", "") if cat and isinstance(cat[0], dict) else str(cat[0])
                    top_level["entity_type"] = str(cat) if cat else ""
                return top_level
        return data


# ---------------------------------------------------------------------------
# Top-level challenge-compliant table response
# ---------------------------------------------------------------------------

class SearchTableResponse(BaseModel):
    """
    Challenge-compliant structured output.

    Renders as a table: `columns` defines headers, `entities` are rows.
    Every cell is a CellValue with full provenance.
    """

    query: str = Field(..., description="The original topic query.")
    entity_type: str = Field(
        default="",
        description="Inferred type of entities discovered (e.g. 'Restaurant', 'AI Startup').",
    )
    columns: List[str] = Field(
        default_factory=list,
        description="Ordered list of column names present across all entity rows.",
    )
    entities: List[EntityRow] = Field(
        default_factory=list,
        description="Discovered entities ordered by relevance (highest first).",
    )

    @model_validator(mode="after")
    def derive_columns(self) -> "SearchTableResponse":
        """Auto-populate columns from the union of all entity field keys."""
        if not self.columns and self.entities:
            seen: list[str] = []
            for ent in self.entities:
                for k in ent.fields:
                    if k not in seen:
                        seen.append(k)
            self.columns = seen
        return self


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_ref(ref: str, defs: dict) -> dict:
    """Resolve a JSON Schema $ref like '#/$defs/CellValue'."""
    name = ref.split("/")[-1]
    return defs.get(name, {})


def schema_to_prompt_description(model: type[BaseModel], indent: int = 0) -> str:
    """
    Generate a human-readable field description from a Pydantic model,
    recursively expanding nested $defs so the LLM sees the full structure.
    """
    full_schema = model.model_json_schema()
    defs = full_schema.get("$defs", {})

    def describe(schema_fragment: dict, depth: int) -> list[str]:
        lines = []
        prefix = "  " * depth
        properties = schema_fragment.get("properties", {})
        required = set(schema_fragment.get("required", []))

        for field_name, field_info in properties.items():
            desc = field_info.get("description", "")
            req_tag = "required" if field_name in required else "optional"

            # Resolve $ref
            if "$ref" in field_info:
                ref_schema = _resolve_ref(field_info["$ref"], defs)
                ftype = ref_schema.get("title", field_info["$ref"].split("/")[-1])
                lines.append(f"{prefix}- {field_name} ({ftype}, {req_tag}): {desc}")
                lines.extend(describe(ref_schema, depth + 1))
            # Handle anyOf / allOf (e.g. Optional fields)
            elif "anyOf" in field_info:
                inner = next((x for x in field_info["anyOf"] if "$ref" in x), None)
                if inner:
                    ref_schema = _resolve_ref(inner["$ref"], defs)
                    ftype = ref_schema.get("title", inner["$ref"].split("/")[-1])
                    lines.append(f"{prefix}- {field_name} ({ftype}, {req_tag}): {desc}")
                    lines.extend(describe(ref_schema, depth + 1))
                else:
                    ftype = " | ".join(x.get("type", "?") for x in field_info["anyOf"] if "type" in x)
                    lines.append(f"{prefix}- {field_name} ({ftype}, {req_tag}): {desc}")
            else:
                ftype = field_info.get("type", "object")
                lines.append(f"{prefix}- {field_name} ({ftype}, {req_tag}): {desc}")

        return lines

    header = [f"Schema: {model.__name__}"]
    if model.__doc__:
        header.append(model.__doc__.strip().split("\n")[0])
    header.append("")
    return "\n".join(header + describe(full_schema, 0))


def build_entity_row_prompt_spec(columns: list[str]) -> str:
    """
    Generate a concrete JSON example showing exactly what one entity row
    should look like given a specific column list.
    """
    example_fields = {
        col: {
            "value": f"<{col} value>",
            "source_url": "https://example.com/page",
            "excerpt": "Verbatim text from that page supporting this value.",
            "confidence": 0.9
        }
        for col in columns
    }
    example = {
        "entity_type": "<type>",
        "fields": example_fields,
        "summary": "<one sentence why this entity is relevant>",
        "relevance": 0.9
    }
    return json.dumps(example, indent=2)
