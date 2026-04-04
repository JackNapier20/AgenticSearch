"""
Web search module using OpenAI Responses API with web_search_preview tool.

Sends a query to GPT with web search enabled, then parses response annotations
to extract source URLs, titles, and text snippets.
"""

from dataclasses import dataclass, field
from openai import OpenAI


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
    """
    client = OpenAI()

    search_prompt = (
        f"Search the web for: {query}\n\n"
        f"Find at least {num_results} distinct, relevant results. "
        "For each result, provide a brief summary of what makes it relevant. "
        "Include a diverse range of sources (articles, lists, reviews, official sites)."
    )

    response = client.responses.create(
        model="gpt-4o-mini",
        tools=[{"type": "web_search_preview"}],
        input=search_prompt,
    )

    # Extract URLs from annotations in the response output
    results: list[SearchResult] = []
    seen_urls: set[str] = set()

    for item in response.output:
        if item.type == "message":
            for content_block in item.content:
                if content_block.type == "output_text":
                    # Parse annotations for source URLs
                    if hasattr(content_block, "annotations") and content_block.annotations:
                        for annotation in content_block.annotations:
                            if annotation.type == "url_citation":
                                url = annotation.url
                                title = annotation.title or ""
                                if url not in seen_urls:
                                    seen_urls.add(url)
                                    # Extract snippet from surrounding text
                                    snippet = _extract_snippet(
                                        content_block.text,
                                        annotation.start_index,
                                        annotation.end_index,
                                    )
                                    results.append(SearchResult(
                                        url=url,
                                        title=title,
                                        snippet=snippet,
                                    ))

    response_text = response.output_text if hasattr(response, "output_text") else ""
    return results[:num_results], response_text


def _extract_snippet(full_text: str, start: int, end: int, context_chars: int = 200) -> str:
    """Extract a text snippet around an annotation's position."""
    # Walk backwards to find the start of the sentence/paragraph
    snippet_start = max(0, start - context_chars)
    snippet_end = min(len(full_text), end + context_chars)

    # Try to align to sentence boundaries
    text = full_text[snippet_start:snippet_end]

    # Clean up partial sentences at boundaries
    if snippet_start > 0:
        first_period = text.find(". ")
        if first_period != -1 and first_period < context_chars:
            text = text[first_period + 2:]

    if snippet_end < len(full_text):
        last_period = text.rfind(". ")
        if last_period != -1:
            text = text[:last_period + 1]

    return text.strip()
