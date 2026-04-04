"""
Example entity schemas.

NOTE: With the new dynamic extraction strategy, you no longer need to define
schemas to get useful output — entity_type and columns are inferred from the
query automatically.

These classes now serve a different purpose: they let you CONSTRAIN extraction
by passing --schema on the CLI. If you specify a schema, the pipeline will
use its column definitions as the extraction target instead of inferring them.

To add a new schema, subclass EntitySchemaHint and register it in SCHEMA_REGISTRY.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EntitySchemaHint:
    """
    Describes the columns to extract for a specific entity type.
    Passed to the pipeline to override dynamic column inference.
    """
    entity_type: str
    columns: list[str]
    description: str = ""


# ---------------------------------------------------------------------------
# Built-in schema hints
# ---------------------------------------------------------------------------

AIStartup = EntitySchemaHint(
    entity_type="AI Startup",
    columns=["Name", "Description", "Founded", "Funding", "Headquarters", "Focus Area", "Key Product"],
    description="An AI startup company operating in a specific domain.",
)

Restaurant = EntitySchemaHint(
    entity_type="Restaurant",
    columns=["Name", "Neighborhood", "Cuisine", "Price Range", "Rating", "Signature Dish", "Address"],
    description="A restaurant or food establishment.",
)

SoftwareTool = EntitySchemaHint(
    entity_type="Software Tool",
    columns=["Name", "Description", "Category", "License", "Language", "GitHub Stars", "Website"],
    description="An open-source or commercial software tool.",
)

ResearchPaper = EntitySchemaHint(
    entity_type="Research Paper",
    columns=["Title", "Authors", "Year", "Venue", "Abstract", "Key Contribution", "URL"],
    description="An academic or technical research paper.",
)

# ---------------------------------------------------------------------------
# Registry for CLI lookup by name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, EntitySchemaHint] = {
    "AIStartup": AIStartup,
    "Restaurant": Restaurant,
    "SoftwareTool": SoftwareTool,
    "ResearchPaper": ResearchPaper,
}
