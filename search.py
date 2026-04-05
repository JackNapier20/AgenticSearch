"""
Web search module using OpenAI Responses API with web_search_preview tool.

Sends a query to GPT with web search enabled, then parses response annotations
to extract source URLs, titles, and text snippets.
"""

import logging
from dataclasses import dataclass
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with URL, title, and text snippet."""
    url: str
    title: str
    snippet: str


def web_search(query: str, num_results: int = 10) -> tuple[list[SearchResult], str]:
    """
    Search the web for a topic query using OpenAI's web_search_preview tool.

    Args:
        query: The search query string.
        num_results: Desired number of unique URL results (best-effort).

    Returns:
        A tuple of (list of SearchResult, raw response text).
        On any failure, returns ([], "") so the pipeline can continue with
        whatever data is available.
    """
    client = OpenAI()

    search_prompt = (
        f"Search the web for: {query}\n\n"
        f"Find at least {num_results} distinct, relevant results. "
        "For each result, provide a brief summary of what makes it relevant. "
        "Include a diverse range of sources (articles, lists, reviews, official sites)."
    )

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type": "web_search_preview"}],
            input=search_prompt,
        )
    except Exception as e:
        logger.error(f"web_search API call failed: {e}")
        return [], ""

    # Extract URLs from annotations in the response output
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    try:
        for item in (response.output or []):
            if not hasattr(item, "type") or item.type != "message":
                continue
            for content_block in (getattr(item, "content", None) or []):
                if not hasattr(content_block, "type") or content_block.type != "output_text":
                    continue
                annotations = getattr(content_block, "annotations", None) or []
                for annotation in annotations:
                    try:
                        if getattr(annotation, "type", None) != "url_citation":
                            continue
                        url = getattr(annotation, "url", None)
                        if not url or url in seen_urls:
                            continue
                        seen_urls.add(url)
                        title = getattr(annotation, "title", "") or ""
                        snippet = _extract_snippet(
                            getattr(content_block, "text", "") or "",
                            getattr(annotation, "start_index", 0) or 0,
                            getattr(annotation, "end_index", 0) or 0,
                        )
                        results.append(SearchResult(url=url, title=title, snippet=snippet))
                    except Exception as ann_err:
                        logger.debug(f"  Skipping annotation: {ann_err}")
    except Exception as parse_err:
        logger.warning(f"web_search response parsing error: {parse_err}")

    # Extract the summary text from the response
    response_text = ""
    try:
        response_text = response.output_text or ""
    except Exception:
        # output_text may not exist on all response shapes
        try:
            for item in (response.output or []):
                if hasattr(item, "type") and item.type == "message":
                    for cb in (getattr(item, "content", None) or []):
                        if getattr(cb, "type", None) == "output_text":
                            response_text = getattr(cb, "text", "") or ""
                            break
                    if response_text:
                        break
        except Exception as text_err:
            logger.debug(f"  Could not extract response text: {text_err}")

    logger.debug(f"web_search: {len(results)} URLs, summary {len(response_text)} chars")
    return results[:num_results], response_text


def _extract_snippet(full_text: str, start: int, end: int, context_chars: int = 200) -> str:
    """Extract a text snippet around an annotation's position."""
    if not full_text:
        return ""
    snippet_start = max(0, start - context_chars)
    snippet_end = min(len(full_text), end + context_chars)

    text = full_text[snippet_start:snippet_end]

    if snippet_start > 0:
        first_period = text.find(". ")
        if first_period != -1 and first_period < context_chars:
            text = text[first_period + 2:]

    if snippet_end < len(full_text):
        last_period = text.rfind(". ")
        if last_period != -1:
            text = text[:last_period + 1]

    return text.strip()
