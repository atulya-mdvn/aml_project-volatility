from pathlib import Path
import hashlib
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


def parse_fed_page(html: str, url: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    # Better title extraction
    title_tag = soup.find("h3", class_="title")
    if title_tag:
        title = title_tag.get_text(" ", strip=True)
    elif soup.title:
        title = soup.title.get_text(" ", strip=True)
    else:
        title = None

    # Try to isolate the main content area
    content = (
        soup.find("div", class_="col-xs-12 col-sm-8 col-md-8")
        or soup.find("div", id="article")
        or soup.find("main")
        or soup
    )

    paragraphs = [p.get_text(" ", strip=True) for p in content.find_all("p")]

    print(f"Found {len(paragraphs)} paragraphs for {url}")
    if paragraphs:
        print("First paragraph preview:", paragraphs[0][:120])

    text = "\n".join([p for p in paragraphs if p])

    return {
        "doc_id": make_doc_id(url),
        "date": None,
        "timestamp": None,
        "source": "federalreserve",
        "source_type": "fed",
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

    df = pd.DataFrame(rows, columns=P3_DOCUMENT_COLUMNS)
    return df


def validate_fed_documents(df: pd.DataFrame) -> None:
    if list(df.columns) != P3_DOCUMENT_COLUMNS:
        raise ValueError("Fed document schema does not match P3_DOCUMENT_COLUMNS.")
    if "doc_id" in df.columns and df["doc_id"].duplicated().any():
        raise ValueError("Duplicate doc_id values found.")


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