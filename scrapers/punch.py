# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 12:27:40 2026

@author: Oreoluwa
"""

import requests
from bs4 import BeautifulSoup
from typing import Tuple

def scrape_punch_article(url: str) -> Tuple[str, str]:
    """
    Scrapes a PunchNG article and returns the title and main text.

    Args:
        url (str): URL of the Punch article

    Returns:
        Tuple[str, str]: (title, article_text)

    Raises:
        ValueError: If article structure isn't found
        requests.RequestException: If network request fails
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

    # ---- Extract the title ----
    title_tag = soup.find("h1")
    if not title_tag:
        raise ValueError("Article title not found on PunchNG")

    title = title_tag.get_text(strip=True)

    # ---- Extract article body paragraphs ----
    # From the page HTML the content follows a structure of multiple <p> tags
    # directly under the article container
    article_paragraphs = soup.select("article p") or soup.find_all("p")

    if not article_paragraphs:
        raise ValueError("Article body paragraphs not found on PunchNG")

    article_text = "\n\n".join(
        p.get_text(strip=True) for p in article_paragraphs
    )

    return {
            "title": title,
            "text": article_text,
        }
