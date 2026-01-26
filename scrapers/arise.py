# -*- coding: utf-8 -*-
"""
Created on Mon Jan 26 09:07:09 2026

@author: Oreoluwa
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict

def scrape_arise_tv_article(url: str) -> Dict[str, str]:
    """
    Scrapes an Arise.tv news article and returns the title and main text.

    Args:
        url (str): URL of the Arise.tv article

    Returns:
        Dict[str, str]: {"title": ..., "text": ...}

    Raises:
        ValueError: If expected content isn’t found
        requests.RequestException: If HTTP request fails
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
    article_container = (
        soup.find("div", {"class": "story__body"}) or
        soup.find("div", {"data-component": "article-body"}) or
        soup.find("article")
    )

    if not article_container:
        raise ValueError("Article content container not found")

    paragraphs = article_container.find_all("p")
    if not paragraphs:
        raise ValueError("No article paragraphs found")

    article_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs)

    return {"title": title, "text": article_text}


