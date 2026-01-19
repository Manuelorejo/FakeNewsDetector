import requests
from bs4 import BeautifulSoup
from typing import Tuple


def scrape_pulse_article(url: str) -> Tuple[str, str]:
    """
    Scrapes a Pulse.ng article and returns the title and main text.

    Args:
        url (str): URL of the Pulse article

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
        response.raise_for_status()  # HTTP errors (4xx, 5xx)
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch page: {e}")

    soup = BeautifulSoup(response.text, "html.parser")

    # ---- Extract title ----
    title_tag = soup.find("h1")
    if not title_tag:
        raise ValueError("Article title not found")

    title = title_tag.get_text(strip=True)

    # ---- Extract article body ----
    container = soup.find("section", class_="space-y-5 sm:space-y-7")
    if not container:
        raise ValueError("Article content container not found")

    paragraphs = container.find_all("p")
    if not paragraphs:
        raise ValueError("No article paragraphs found")

    article_text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs
    )

    return {
            "title": title,
            "text": article_text,
        }



