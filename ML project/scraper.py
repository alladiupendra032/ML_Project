"""
Web Scraping Module
Scrapes text from public websites using BeautifulSoup.
Supports: Books to Scrape (ratings → labels), Quotes to Scrape,
          and Custom URL (auto-labelled with TextBlob polarity).
"""

import requests
import time
import random
import re
import pandas as pd
from typing import List, Dict, Optional
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY = (1.0, 2.5)   # seconds between page requests

# TextBlob polarity thresholds for 3-class labelling
POS_THRESHOLD  =  0.08   # polarity >  0.08 → Positive
NEG_THRESHOLD  = -0.08   # polarity < -0.08 → Negative
                          # otherwise        → Neutral


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _get_page(url: str, timeout: int = 12) -> Optional[BeautifulSoup]:
    """Fetch a URL and return a BeautifulSoup object, or None on error."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        return BeautifulSoup(resp.text, "lxml")
    except Exception:
        return None


def _textblob_label(text: str) -> str:
    """
    Return Positive / Negative / Neutral based on TextBlob polarity.
    TextBlob is a rule-based NLP library — no training required.
    Polarity ranges from -1.0 (very negative) to +1.0 (very positive).
    """
    try:
        from textblob import TextBlob
        polarity = TextBlob(text).sentiment.polarity
        if polarity > POS_THRESHOLD:
            return "Positive"
        elif polarity < NEG_THRESHOLD:
            return "Negative"
        else:
            return "Neutral"
    except Exception:
        return "Neutral"   # safe fallback


# ─── Source: Books to Scrape ──────────────────────────────────────────────────

def scrape_books_to_scrape(max_pages: int = 3) -> List[Dict]:
    """
    Scrape book titles + star ratings from books.toscrape.com.
    Labels are assigned from the numeric rating (1-2 → Neg, 3 → Neu, 4-5 → Pos).
    """
    base_url   = "https://books.toscrape.com/catalogue/page-{}.html"
    rating_map = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
    results    = []

    for page in range(1, max_pages + 1):
        soup = _get_page(base_url.format(page))
        if soup is None:
            break

        for art in soup.select("article.product_pod"):
            title_tag  = art.select_one("h3 a")
            rating_tag = art.select_one("p.star-rating")
            price_tag  = art.select_one("p.price_color")
            if not title_tag:
                continue

            title  = title_tag.get("title", title_tag.get_text(strip=True))
            rating = rating_map.get(rating_tag["class"][1] if rating_tag else "Three", 3)
            price  = price_tag.get_text(strip=True) if price_tag else "N/A"
            text   = f"{title}. Rating: {rating} out of 5 stars. Price: {price}."

            results.append({
                "text":       text,
                "raw_rating": rating,
                "source":     "books.toscrape.com",
            })

        time.sleep(random.uniform(*REQUEST_DELAY))

    return results


# ─── Source: Quotes to Scrape ─────────────────────────────────────────────────

def scrape_quotes_to_scrape(max_pages: int = 5) -> List[Dict]:
    """
    Scrape quotes from quotes.toscrape.com.
    No star ratings → TextBlob polarity used for auto-labelling.
    """
    base_url = "https://quotes.toscrape.com/page/{}/"
    results  = []

    for page in range(1, max_pages + 1):
        soup = _get_page(base_url.format(page))
        if soup is None:
            break

        for q in soup.select("div.quote"):
            text_tag   = q.select_one("span.text")
            author_tag = q.select_one("small.author")
            tags       = [t.get_text(strip=True) for t in q.select("a.tag")]
            if not text_tag:
                continue

            text   = text_tag.get_text(strip=True).strip('"\u201c\u201d')
            author = author_tag.get_text(strip=True) if author_tag else "Unknown"
            full   = f"{text} — {author}." + (f" Tags: {', '.join(tags)}." if tags else "")

            results.append({
                "text":       full,
                "raw_rating": None,
                "source":     "quotes.toscrape.com",
            })

        if not soup.select_one("li.next a"):
            break
        time.sleep(random.uniform(*REQUEST_DELAY))

    return results


# ─── Source: Custom URL ───────────────────────────────────────────────────────

def scrape_custom_url(url: str, max_items: int = 300) -> List[Dict]:
    """
    General-purpose scraper: extracts paragraph/list-item text from any URL.
    Each extracted text segment is AUTO-LABELLED using TextBlob polarity,
    so the data is immediately usable for model training.
    """
    results = []
    soup    = _get_page(url)
    if soup is None:
        return results

    # Strip non-content tags
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript"]):
        tag.decompose()

    seen = set()
    # Prefer <p> and <li> — richer semantic content than <div>/<span>
    for tag in soup.find_all(["p", "li", "blockquote", "td", "h2", "h3"]):
        text = tag.get_text(separator=" ", strip=True)
        # Keep only segments long enough to be meaningful
        if len(text) < 40 or text in seen:
            continue
        seen.add(text)

        label = _textblob_label(text)
        results.append({
            "text":       text,
            "raw_rating": None,
            "sentiment":  label,   # already labelled via TextBlob
            "source":     url,
        })

        if len(results) >= max_items:
            break

    return results


# ─── Label assignment ─────────────────────────────────────────────────────────

def assign_labels_from_rating(records: List[Dict]) -> List[Dict]:
    """
    For records that have a raw_rating, map it to a sentiment label.
    For records already labelled (custom URL), leave the label untouched.
    Records with no rating and no label get TextBlob auto-labelling.

    Rating mapping:
        1–2  → Negative
        3    → Neutral
        4–5  → Positive
    """
    for r in records:
        # Already has a label (e.g. custom URL scrape) — no overwrite
        if r.get("sentiment") is not None:
            continue

        rating = r.get("raw_rating")
        if rating is not None:
            if rating <= 2:
                r["sentiment"] = "Negative"
            elif rating == 3:
                r["sentiment"] = "Neutral"
            else:
                r["sentiment"] = "Positive"
        else:
            # Fallback: TextBlob auto-label for sources without ratings
            r["sentiment"] = _textblob_label(r.get("text", ""))
    return records


# ─── DataFrame helper ─────────────────────────────────────────────────────────

def records_to_dataframe(records: List[Dict]) -> pd.DataFrame:
    """Convert scraped records list to a clean, deduplicated DataFrame."""
    if not records:
        return pd.DataFrame(columns=["text", "sentiment", "source"])

    df = pd.DataFrame(records)

    # Ensure required columns exist
    for col in ["text", "sentiment", "source"]:
        if col not in df.columns:
            df[col] = None

    df = df[["text", "sentiment", "source"]].copy()
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip() != ""].reset_index(drop=True)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


# ─── UI metadata ──────────────────────────────────────────────────────────────

AVAILABLE_SOURCES = {
    "📚 Books to Scrape (Recommended)": "books",
    "💬 Quotes to Scrape":              "quotes",
    "🌐 Custom URL":                    "custom",
}
