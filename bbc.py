import requests
from bs4 import BeautifulSoup
from typing import Tuple


def scrape_bbc_article(url: str) -> Tuple[str, str]:
    """
    Scrapes a BBC News article and returns the title and main text.

    Args:
        url (str): URL of the BBC News article

    Returns:
        Tuple[str, str]: (title, article_text)

    Raises:
        ValueError: If expected article structure isn't found
        requests.RequestException: If the HTTP request fails
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
        raise ValueError("Title not found on BBC article")

    title = title_tag.get_text(strip=True)

    # ---- Extract article body ----
    # BBC articles typically put paragraphs inside <article> or a div with "ssrcss-*" classes
    article_body = soup.find("article")
    if not article_body:
        # fallback: look for divs that contain paragraphs
        article_body = soup.find("div", {"data-component": "text-block"})

    if not article_body:
        raise ValueError("Could not locate the article content")

    paragraphs = article_body.find_all("p")
    if not paragraphs:
        raise ValueError("No article paragraphs found")

    article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

    return {
            "title": title,
            "text": article_text,
        }

scrape_bbc_article("https://www.bbc.com/news/articles/cqj2qgw8w08o")