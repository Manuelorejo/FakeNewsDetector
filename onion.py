import requests
from bs4 import BeautifulSoup
from typing import Tuple


def scrape_onion_article(url: str) -> Tuple[str, str]:
    """
    Scrapes a The Onion article and returns the title and main text.

    Args:
        url (str): URL of the The Onion article

    Returns:
        Tuple[str, str]: (title, article_text)

    Raises:
        ValueError: If the page structure is not as expected
        requests.RequestException: If the request fails
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch page: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # ---- Extract title ----
    title_tag = soup.find("h1")
    if not title_tag:
        raise ValueError("Article title not found")

    title = title_tag.get_text(strip=True)

    # ---- Extract article body ----
    paragraphs = soup.select("div.entry-content p")
    if not paragraphs:
        raise ValueError("Article paragraphs not found")

    article_text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs
    )

    return {
            "title": title,
            "text": article_text,
        }

scrape_onion_article("https://theonion.com/gop-adds-ice-kills-everyone-pillar-to-2026-platform/")