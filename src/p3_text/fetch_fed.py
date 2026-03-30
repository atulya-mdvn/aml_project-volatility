from pathlib import Path
import hashlib
import re
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.shared.schemas import P3_DOCUMENT_COLUMNS


OUTPUT_PATH = Path("data/interim/p3_documents.parquet")


def make_doc_id(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


def fetch_url_html(url: str) -> str:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def parse_date_text(date_text: str) -> tuple[str | None, str | None]:
    """
    Convert a raw date/time string into:
    - date: YYYY-MM-DD
    - timestamp: YYYY-MM-DD HH:MM:SS if time is present, else None
    """
    if not date_text:
        return None, None

    cleaned = " ".join(date_text.split())
    cleaned = cleaned.replace("Sept.", "Sep.").replace("Sept ", "Sep ")

    datetime_formats = [
        "%B %d, %Y %I:%M %p %Z",
        "%b %d, %Y %I:%M %p %Z",
        "%B %d, %Y %I:%M %p",
        "%b %d, %Y %I:%M %p",
    ]

    date_formats = [
        "%B %d, %Y",
        "%b %d, %Y",
        "%B %d %Y",
        "%b %d %Y",
    ]

    for fmt in datetime_formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.date().isoformat(), dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    for fmt in date_formats:
        try:
            dt = datetime.strptime(cleaned, fmt)
            return dt.date().isoformat(), None
        except ValueError:
            pass

    return None, None


def extract_date_from_url(url: str) -> str | None:
    """
    Extract a publication date from Federal Reserve URLs when the URL
    contains an 8-digit YYYYMMDD date token.
    """
    match = re.search(r"(\d{8})[a-z]?\.(?:htm|html)$", url.lower())
    if not match:
        return None

    raw = match.group(1)

    try:
        dt = datetime.strptime(raw, "%Y%m%d")
        return dt.date().isoformat()
    except ValueError:
        return None

def extract_fed_date_timestamp(
    soup: BeautifulSoup, url: str
) -> tuple[str | None, str | None]:
    """
    Try several common Federal Reserve patterns to extract a publication
    date/timestamp. Prefer URL-encoded publication dates when available.
    """
    url_date = extract_date_from_url(url)
    if url_date:
        return url_date, None

    candidate_selectors = [
        ("p", {"class": "article__time"}),
        ("p", {"class": "datetime"}),
        ("time", {}),
        ("div", {"class": "lastUpdate"}),
    ]

    for tag_name, attrs in candidate_selectors:
        tag = soup.find(tag_name, attrs)
        if tag:
            raw_text = tag.get_text(" ", strip=True)
            date_value, timestamp_value = parse_date_text(raw_text)
            if date_value:
                return date_value, timestamp_value

    page_text = soup.get_text("\n", strip=True)

    patterns = [
        r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}(?:\s+\d{1,2}:\d{2}\s+[AP]M(?:\s+[A-Z]{2,4})?)?",
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)\.?\s+\d{1,2},\s+\d{4}(?:\s+\d{1,2}:\d{2}\s+[AP]M(?:\s+[A-Z]{2,4})?)?",
    ]

    for pattern in patterns:
        match = re.search(pattern, page_text)
        if match:
            return parse_date_text(match.group(0))

    return None, None

def classify_fed_source_type(url: str, title: str | None) -> str:
    url_lower = url.lower()
    title_lower = (title or "").lower()

    if "fomcminutes" in url_lower or "minutes" in title_lower:
        return "fed_minutes"
    if "testimony" in url_lower or "testimony" in title_lower:
        return "fed_testimony"
    if "speech" in url_lower or "speech" in title_lower:
        return "fed_speech"
    if "pressreleases" in url_lower or "statement" in title_lower:
        return "fed_press_release"

    return "fed"


def extract_title(soup: BeautifulSoup) -> str | None:
    title_tag = soup.find("h3", class_="title")
    if title_tag:
        return title_tag.get_text(" ", strip=True)

    if soup.title:
        return soup.title.get_text(" ", strip=True)

    return None


def extract_main_text(soup: BeautifulSoup) -> str:
    content = (
        soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        or soup.find("div", id="article")
        or soup.find("main")
        or soup
    )

    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]
    paragraphs = [p for p in paragraphs if p]

    return "\n".join(paragraphs)


def parse_fed_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    title = extract_title(soup)
    date_value, timestamp_value = extract_fed_date_timestamp(soup, url)
    source_type = classify_fed_source_type(url, title)
    text = extract_main_text(soup)

    paragraph_count = len([p for p in text.split("\n") if p])

    print(f"Found {paragraph_count} paragraphs for {url}")
    if text:
        print("First paragraph preview:", text.split("\n")[0][:120])
    print(
        f"Extracted date: {date_value}, "
        f"timestamp: {timestamp_value}, "
        f"source_type: {source_type}"
    )

    return {
        "doc_id": make_doc_id(url),
        "date": date_value,
        "timestamp": timestamp_value,
        "source": "federalreserve",
        "source_type": source_type,
        "title": title,
        "text": text,
        "ticker": None,
        "url": url,
    }


def build_fed_documents(urls: list[str]) -> pd.DataFrame:
    rows = []

    for url in urls:
        print(f"Fetching: {url}")
        try:
            html = fetch_url_html(url)
            row = parse_fed_page(html, url)
            print(f"Parsed title: {row['title']}")
            print(f"Text length: {len(row['text']) if row['text'] else 0}")
            rows.append(row)
        except Exception as e:
            print(f"Failed for {url}: {e}")

    return pd.DataFrame(rows, columns=P3_DOCUMENT_COLUMNS)


def validate_fed_documents(df: pd.DataFrame) -> None:
    if list(df.columns) != P3_DOCUMENT_COLUMNS:
        raise ValueError("Fed document schema does not match P3_DOCUMENT_COLUMNS.")

    if "doc_id" in df.columns and df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values found.")

    if df["text"].isna().any():
        raise ValueError("Missing text found in Fed documents.")


def main() -> None:
    urls = [
        "https://www.federalreserve.gov/newsevents/pressreleases/monetary20240320a.htm",
        "https://www.federalreserve.gov/monetarypolicy/fomcminutes20240131.htm",
        "https://www.federalreserve.gov/newsevents/testimony/powell20240306a.htm",
    ]

    df = build_fed_documents(urls)
    validate_fed_documents(df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()