"""
Example entity schemas.

NOTE: With the new dynamic extraction strategy, you no longer need to define
schemas to get useful output — entity_type and columns are inferred from the
query automatically.

These classes now serve two purposes:
  1. Constrain extraction by passing --schema on the CLI.
  2. Provide few-shot entity examples that are injected into extraction prompts
     to improve output consistency and format adherence.

To add a new schema, subclass EntitySchemaHint and register it in SCHEMA_REGISTRY.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class EntitySchemaHint:
    """
    Describes the columns to extract for a specific entity type,
    plus a concrete example entity row used as a few-shot prompt.
    """
    entity_type: str
    columns: list[str]
    description: str = ""
    examples: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Built-in schema hints with concrete few-shot examples
# ---------------------------------------------------------------------------

AIStartup = EntitySchemaHint(
    entity_type="AI Startup",
    columns=["Name", "Description", "Founded", "Funding", "Headquarters", "Focus Area", "Key Product"],
    description="An AI startup company operating in a specific domain.",
    examples=[
        {
            "entity_type": "AI Startup",
            "fields": {
                "Name":          {"value": "Hugging Face",    "source_url": "https://huggingface.co/about",    "excerpt": "Hugging Face is an AI company...",      "confidence": 0.97},
                "Description":   {"value": "Open-source AI model hub and NLP platform", "source_url": "https://huggingface.co/about", "excerpt": "We are on a mission to democratize good machine learning.", "confidence": 0.95},
                "Founded":       {"value": "2016",            "source_url": "https://huggingface.co/about",    "excerpt": "founded in 2016",                       "confidence": 0.95},
                "Funding":       {"value": "$395M Series D",  "source_url": "https://techcrunch.com/2023/08/24/hugging-face-raises-235m", "excerpt": "raised $235 million in a Series D round", "confidence": 0.90},
                "Headquarters":  {"value": "New York, NY",    "source_url": "https://huggingface.co/about",    "excerpt": "headquartered in New York",             "confidence": 0.92},
                "Focus Area":    {"value": "NLP, Open-Source ML", "source_url": "https://huggingface.co/about", "excerpt": "natural language processing and open-source AI", "confidence": 0.95},
                "Key Product":   {"value": "Transformers library, Model Hub", "source_url": "https://huggingface.co/docs", "excerpt": "The Transformers library provides thousands of pretrained models", "confidence": 0.95},
            },
            "summary": "Hugging Face is a leading open-source AI platform focused on NLP and model sharing.",
            "relevance": 0.97,
        }
    ],
)

Restaurant = EntitySchemaHint(
    entity_type="Restaurant",
    columns=["Name", "Neighborhood", "Cuisine", "Price Range", "Rating", "Signature Dish", "Address"],
    description="A restaurant or food establishment.",
    examples=[
        {
            "entity_type": "Restaurant",
            "fields": {
                "Name":           {"value": "Lucali",              "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "Lucali is a pizza restaurant in Carroll Gardens", "confidence": 0.98},
                "Neighborhood":   {"value": "Carroll Gardens, Brooklyn", "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "located in Carroll Gardens",               "confidence": 0.97},
                "Cuisine":        {"value": "Neapolitan Pizza",    "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "wood-fired Neapolitan-style pizza",              "confidence": 0.96},
                "Price Range":    {"value": "$$",                  "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "moderately priced, cash only",                  "confidence": 0.90},
                "Rating":         {"value": "4.7 / 5",             "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "4.7 stars based on 2,300 reviews",              "confidence": 0.95},
                "Signature Dish": {"value": "Margherita Pizza",    "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "famous for its classic margherita pizza",        "confidence": 0.93},
                "Address":        {"value": "575 Henry St, Brooklyn, NY 11231", "source_url": "https://www.yelp.com/biz/lucali-brooklyn", "excerpt": "575 Henry Street",                 "confidence": 0.98},
            },
            "summary": "Lucali is a beloved Brooklyn pizzeria known for its simple, high-quality Neapolitan pies and long waits.",
            "relevance": 0.97,
        }
    ],
)

SoftwareTool = EntitySchemaHint(
    entity_type="Software Tool",
    columns=["Name", "Description", "Category", "License", "Language", "GitHub Stars", "Website"],
    description="An open-source or commercial software tool.",
    examples=[
        {
            "entity_type": "Software Tool",
            "fields": {
                "Name":         {"value": "FastAPI",              "source_url": "https://fastapi.tiangolo.com",        "excerpt": "FastAPI is a modern, fast web framework",      "confidence": 0.98},
                "Description":  {"value": "High-performance Python web framework for building APIs", "source_url": "https://fastapi.tiangolo.com", "excerpt": "FastAPI framework, high performance, easy to learn", "confidence": 0.97},
                "Category":     {"value": "Web Framework",        "source_url": "https://fastapi.tiangolo.com",        "excerpt": "web framework for building APIs",              "confidence": 0.96},
                "License":      {"value": "MIT",                  "source_url": "https://github.com/tiangolo/fastapi", "excerpt": "License: MIT",                                "confidence": 0.99},
                "Language":     {"value": "Python",               "source_url": "https://github.com/tiangolo/fastapi", "excerpt": "Python 3.7+",                                 "confidence": 0.99},
                "GitHub Stars": {"value": "73k+",                 "source_url": "https://github.com/tiangolo/fastapi", "excerpt": "73,000 stars on GitHub",                      "confidence": 0.92},
                "Website":      {"value": "https://fastapi.tiangolo.com", "source_url": "https://fastapi.tiangolo.com", "excerpt": "Official documentation at fastapi.tiangolo.com", "confidence": 0.99},
            },
            "summary": "FastAPI is a modern Python web framework for building high-performance REST APIs with automatic OpenAPI docs.",
            "relevance": 0.96,
        }
    ],
)

ResearchPaper = EntitySchemaHint(
    entity_type="Research Paper",
    columns=["Title", "Authors", "Year", "Venue", "Abstract", "Key Contribution", "URL"],
    description="An academic or technical research paper.",
    examples=[
        {
            "entity_type": "Research Paper",
            "fields": {
                "Title":            {"value": "Attention Is All You Need", "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "Attention Is All You Need",                          "confidence": 0.99},
                "Authors":          {"value": "Vaswani et al.",            "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "Ashish Vaswani, Noam Shazeer, Niki Parmar, ...",   "confidence": 0.98},
                "Year":             {"value": "2017",                      "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "Submitted on 12 Jun 2017",                         "confidence": 0.99},
                "Venue":            {"value": "NeurIPS 2017",              "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "Advances in Neural Information Processing Systems", "confidence": 0.97},
                "Abstract":         {"value": "Proposes the Transformer architecture based solely on attention mechanisms, dispensing with recurrence.", "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks", "confidence": 0.97},
                "Key Contribution": {"value": "Transformer architecture with multi-head self-attention", "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "we propose a new simple network architecture, the Transformer, based solely on attention mechanisms", "confidence": 0.98},
                "URL":              {"value": "https://arxiv.org/abs/1706.03762", "source_url": "https://arxiv.org/abs/1706.03762", "excerpt": "arXiv:1706.03762",                        "confidence": 0.99},
            },
            "summary": "Foundational paper introducing the Transformer architecture, now the basis of most modern LLMs.",
            "relevance": 0.99,
        }
    ],
)

# ---------------------------------------------------------------------------
# Registry for CLI lookup by name
# ---------------------------------------------------------------------------

SCHEMA_REGISTRY: dict[str, EntitySchemaHint] = {
    "AIStartup":     AIStartup,
    "Restaurant":    Restaurant,
    "SoftwareTool":  SoftwareTool,
    "ResearchPaper": ResearchPaper,
}


# ---------------------------------------------------------------------------
# Few-shot helper — used by extractor.py
# ---------------------------------------------------------------------------

def get_few_shot_example(entity_type: str, columns: list[str]) -> str:
    """
    Return a JSON string of one concrete example entity row for use in
    extraction prompts.

    Matching strategy:
      1. Exact match on entity_type (case-insensitive).
      2. Substring match (e.g. "pizza restaurant" matches "Restaurant").
      3. Fallback: build a synthetic template from the given columns.

    The returned string is ready to paste directly into a prompt.
    """
    et_lower = entity_type.lower()

    # Try exact then substring match
    best: EntitySchemaHint | None = None
    for hint in SCHEMA_REGISTRY.values():
        hint_lower = hint.entity_type.lower()
        if hint_lower == et_lower:
            best = hint
            break
        if hint_lower in et_lower or et_lower in hint_lower:
            best = hint

    if best and best.examples:
        return json.dumps(best.examples[0], indent=2)

    # Synthetic fallback (same as before)
    from schema import build_entity_row_prompt_spec
    return build_entity_row_prompt_spec(columns)
