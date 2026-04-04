"""
Web scraping module that fetches and extracts visible text from URLs.

Uses requests + BeautifulSoup to download HTML pages and extract
clean, readable text content.
"""

import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Tags whose content we skip entirely
SKIP_TAGS = {"script", "style", "noscript", "iframe", "svg", "head", "meta", "link"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


@dataclass
class ScrapedPage:
    """A scraped web page with extracted text."""
    url: str
    title: str
    text: str
    success: bool
    error: str = ""


def _extract_visible_text(html: str, max_chars: int) -> tuple[str, str]:
    """
    Extract visible text and title from HTML content.

    Returns:
        (title, visible_text) tuple.
    """
    soup = BeautifulSoup(html, "lxml")

    # Extract title
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Remove unwanted elements
    for tag in soup.find_all(SKIP_TAGS):
        tag.decompose()

    # Also remove common non-content elements
    for selector in ["nav", "footer", "header", ".sidebar", ".menu", ".ad", ".cookie"]:
        for el in soup.select(selector):
            el.decompose()

    # Extract text with some structure preservation
    lines = []
    for element in soup.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td", "th", "span", "div"]):
        text = element.get_text(separator=" ", strip=True)
        if text and len(text) > 20:  # Skip very short fragments
            lines.append(text)

    full_text = "\n".join(lines)

    # Deduplicate repeated content (common with nested divs)
    seen = set()
    deduped_lines = []
    for line in full_text.split("\n"):
        line_stripped = line.strip()
        if line_stripped and line_stripped not in seen:
            seen.add(line_stripped)
            deduped_lines.append(line_stripped)

    clean_text = "\n".join(deduped_lines)

    # Truncate to max_chars
    if len(clean_text) > max_chars:
        # Try to truncate at a sentence boundary
        truncated = clean_text[:max_chars]
        last_period = truncated.rfind(". ")
        if last_period > max_chars * 0.7:
            truncated = truncated[:last_period + 1]
        clean_text = truncated + "\n[... truncated]"

    return title, clean_text


def _scrape_single_url(url: str, max_chars: int, timeout: int) -> ScrapedPage:
    """Scrape a single URL and return a ScrapedPage."""
    try:
        response = requests.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type and "application/xhtml" not in content_type:
            return ScrapedPage(
                url=url, title="", text="",
                success=False, error=f"Non-HTML content type: {content_type}"
            )

        title, text = _extract_visible_text(response.text, max_chars)

        if not text.strip():
            return ScrapedPage(
                url=url, title=title, text="",
                success=False, error="No visible text extracted"
            )

        return ScrapedPage(url=url, title=title, text=text, success=True)

    except requests.Timeout:
        return ScrapedPage(url=url, title="", text="", success=False, error="Request timed out")
    except requests.RequestException as e:
        return ScrapedPage(url=url, title="", text="", success=False, error=str(e))
    except Exception as e:
        return ScrapedPage(url=url, title="", text="", success=False, error=f"Unexpected error: {e}")


def scrape_urls(
    urls: list[str],
    max_chars: int = 5000,
    timeout: int = 15,
    max_workers: int = 5,
) -> list[ScrapedPage]:
    """
    Scrape multiple URLs concurrently and extract visible text.

    Args:
        urls: List of URLs to scrape.
        max_chars: Maximum characters to extract per page.
        timeout: Request timeout in seconds.
        max_workers: Number of concurrent scraping threads.

    Returns:
        List of ScrapedPage objects (same order as input URLs).
    """
    results: dict[str, ScrapedPage] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {
            executor.submit(_scrape_single_url, url, max_chars, timeout): url
            for url in urls
        }

        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                page = future.result()
                results[url] = page
                if page.success:
                    logger.info(f"Scraped {url} — {len(page.text)} chars")
                else:
                    logger.warning(f"Failed {url}: {page.error}")
            except Exception as e:
                results[url] = ScrapedPage(
                    url=url, title="", text="",
                    success=False, error=str(e)
                )

    # Return in original URL order
    return [results[url] for url in urls]
