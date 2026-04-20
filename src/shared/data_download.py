"""
Data download: IBKR for intraday + options, Yahoo Finance as fallback.
"""
import json
import os
import time

import feedparser
import numpy as np
import pandas as pd
import yfinance as yf

from src.shared.config import (
    CROSS_ASSETS,
    END_DATE,
    IBKR_BAR_SIZE,
    IBKR_CLIENT_ID,
    IBKR_DURATION,
    IBKR_HOST,
    IBKR_PORT,
    IBKR_TIMEOUT,
    RAW_DIR,
    START_DATE,
)


def connect_ibkr():
    """Try to connect to IBKR TWS/Gateway. Returns ib object or None."""
    try:
        from ib_insync import IB, Stock, Index, util
        ib = IB()
        ib.connect(IBKR_HOST, IBKR_PORT, clientId=IBKR_CLIENT_ID, timeout=IBKR_TIMEOUT)
        print(f"  IBKR connected: {ib.isConnected()}")
        return ib
    except Exception as e:
        print(f"  IBKR connection failed: {e}")
        print("  Falling back to Yahoo Finance.")
        return None


# CHANGE: Added a small helper so Yahoo fallback is more robust.
# Why: yfinance can get rate-limited, and the old code immediately accepted
# empty results. This preserves the fallback design, but makes it retry first.
def safe_yf_download(ticker, max_retries=3, sleep_seconds=3, **kwargs):
    """Retry Yahoo Finance download a few times before giving up."""
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            df = yf.download(ticker, progress=False, threads=False, **kwargs)
            if df is not None and len(df) > 0:
                return df

            print(f"    {ticker}: empty download on attempt {attempt}/{max_retries}")
        except Exception as e:
            last_error = e
            print(f"    {ticker}: attempt {attempt}/{max_retries} failed ({e})")

        if attempt < max_retries:
            time.sleep(sleep_seconds * attempt)

    if last_error is not None:
        print(f"    {ticker}: all Yahoo attempts failed; returning empty DataFrame")
    else:
        print(f"    {ticker}: Yahoo returned no rows after {max_retries} attempts")

    return pd.DataFrame()


def download_spx_daily(ib=None):
    """Download SPX daily OHLCV. Uses IBKR if available, else Yahoo."""
    path = RAW_DIR / "spx_daily.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            print(f"  SPX daily (cached): {len(df)} rows")
            return df
        except Exception as e:
            print(f"  SPX daily cache unreadable: {e}")
            print("  Deleting bad cache and re-downloading...")
            os.remove(path)

    if ib is not None:
        try:
            from ib_insync import Index
            contract = Index("SPX", "CBOE")
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime="",
                durationStr="20 Y",
                barSizeSetting="1 day",
                whatToShow="TRADES",
                useRTH=True
            )
            df = pd.DataFrame([{
                "Date": b.date,
                "Open": b.open,
                "High": b.high,
                "Low": b.low,
                "Close": b.close,
                "Volume": b.volume
            } for b in bars]).set_index("Date")
            df.index = pd.to_datetime(df.index)
            df.to_csv(path)
            print(f"  SPX daily (IBKR): {len(df)} rows")
            return df
        except Exception as e:
            print(f"  IBKR SPX daily failed: {e}, falling back to Yahoo")

    # CHANGE: Use safe_yf_download instead of raw yf.download.
    # Why: keeps Yahoo as fallback, but gives it a few chances before failing.
    df = safe_yf_download("^GSPC", start=START_DATE, end=END_DATE, auto_adjust=True)

    # CHANGE: Do not save empty files.
    # Why: empty cached CSVs would make later runs misleading and harder to debug.
    if len(df) == 0:
        print("  SPX daily (Yahoo) failed: 0 rows")
        return df

    df.to_csv(path)
    print(f"  SPX daily (Yahoo): {len(df)} rows")
    return df


def download_spx_intraday(ib=None):
    """Download SPX 5-minute bars from IBKR for more accurate RV.
    Returns None if IBKR unavailable (daily RV will be used instead)."""
    path = RAW_DIR / "spx_intraday.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col="DateTime", parse_dates=True)
        print(f"  SPX intraday (cached): {len(df)} rows")
        return df

    if ib is None:
        print("  SPX intraday: IBKR required, skipping (will use daily RV)")
        return None

    try:
        from ib_insync import Index
        contract = Index("SPX", "CBOE")
        ib.qualifyContracts(contract)

        # IBKR limits intraday to ~1 year per request, so we loop
        all_bars = []
        end = ""
        for _ in range(5):  # up to 5 years back
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end,
                durationStr=IBKR_DURATION,
                barSizeSetting=IBKR_BAR_SIZE,
                whatToShow="TRADES",
                useRTH=True
            )
            if not bars:
                break
            all_bars = bars + all_bars
            end = bars[0].date.strftime("%Y%m%d %H:%M:%S")
            time.sleep(2)  # IBKR pacing

        df = pd.DataFrame([{
            "DateTime": b.date,
            "Open": b.open,
            "High": b.high,
            "Low": b.low,
            "Close": b.close,
            "Volume": b.volume
        } for b in all_bars]).set_index("DateTime")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index().drop_duplicates()
        df.to_csv(path)
        print(f"  SPX intraday (IBKR): {len(df)} rows")
        return df
    except Exception as e:
        print(f"  IBKR intraday failed: {e}")
        return None


def download_options_chain(ib=None):
    """Download SPX options chain for real IV surface.
    Returns None if IBKR unavailable (VIX proxies used instead)."""
    path = RAW_DIR / "spx_options.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        print(f"  SPX options (cached): {len(df)} rows")
        return df

    if ib is None:
        print("  SPX options: IBKR required, skipping (will use VIX proxies)")
        return None

    try:
        from ib_insync import Index, Option
        contract = Index("SPX", "CBOE")
        ib.qualifyContracts(contract)

        chains = ib.reqSecDefOptParams(contract.symbol, "", contract.secType, contract.conId)
        if not chains:
            print("  No options chains found")
            return None

        chain = [c for c in chains if c.exchange == "SMART"]
        if not chain:
            chain = chains
        chain = chain[0]

        expirations = sorted(chain.expirations)[:3]
        strikes = sorted(chain.strikes)

        ticker = ib.reqMktData(contract)
        ib.sleep(2)
        spot = ticker.marketPrice()
        if np.isnan(spot):
            spot = ticker.close
        atm_idx = np.argmin(np.abs(np.array(strikes) - spot))
        nearby_strikes = strikes[max(0, atm_idx - 10):atm_idx + 11]

        records = []
        for exp in expirations:
            for strike in nearby_strikes:
                for right in ["C", "P"]:
                    opt = Option("SPX", exp, strike, right, "SMART")
                    try:
                        ib.qualifyContracts(opt)
                        ticker = ib.reqMktData(opt, genericTickList="106")
                        ib.sleep(0.5)
                        records.append({
                            "expiration": exp,
                            "strike": strike,
                            "right": right,
                            "bid": ticker.bid,
                            "ask": ticker.ask,
                            "impliedVol": ticker.modelGreeks.impliedVol if ticker.modelGreeks else np.nan,
                            "delta": ticker.modelGreeks.delta if ticker.modelGreeks else np.nan,
                        })
                    except Exception:
                        pass
            time.sleep(1)

        df = pd.DataFrame(records)
        df.to_csv(path)
        print(f"  SPX options (IBKR): {len(df)} rows")
        return df
    except Exception as e:
        print(f"  IBKR options failed: {e}")
        return None


def download_vix(ib=None):
    """Download VIX term structure."""
    path = RAW_DIR / "vix_term_structure.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            print(f"  VIX (cached): {len(df)} rows")
            return df
        except Exception as e:
            print(f"  VIX cache unreadable: {e}")
            print("  Deleting bad cache and re-downloading...")
            os.remove(path)

    frames = {}
    for label, ticker in {"VIX": "^VIX", "VIX3M": "^VIX3M", "VIX6M": "^VIX6M"}.items():
        try:
            # CHANGE: use safe_yf_download instead of raw yf.download
            df = safe_yf_download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
            if len(df) > 0:
                frames[label] = df["Close"].squeeze()

            # CHANGE: small pacing between Yahoo calls
            # Why: reduces the chance of immediate back-to-back rate limits.
            time.sleep(1)
        except Exception as e:
            print(f"    {ticker}: FAILED ({e})")

    vix = pd.DataFrame(frames)
    vix.index.name = "Date"

    # CHANGE: only cache if we actually got something
    if len(vix) > 0:
        vix.to_csv(path)

    print(f"  VIX (Yahoo): {len(vix)} rows")
    return vix


def download_cross_assets():
    """Download cross-asset ETFs for P4."""
    path = RAW_DIR / "cross_assets.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col="Date", parse_dates=True)
            print(f"  Cross-assets (cached): {df.shape}")
            return df
        except Exception as e:
            print(f"  Cross-assets cache unreadable: {e}")
            print("  Deleting bad cache and re-downloading...")
            os.remove(path)

    frames = {}
    for ticker, name in CROSS_ASSETS.items():
        try:
            # CHANGE: use safe_yf_download instead of raw yf.download
            df = safe_yf_download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
            if len(df) > 0:
                frames[ticker] = df["Close"].squeeze()
                print(f"    {ticker} ({name}): {len(df)}")
            else:
                print(f"    {ticker} ({name}): 0 rows")

            # CHANGE: small pacing between requests
            time.sleep(1)
        except Exception as e:
            print(f"    {ticker} ({name}): FAILED ({e})")

    cross = pd.DataFrame(frames)
    cross.index.name = "Date"

    # CHANGE: only cache non-empty cross-asset data
    if cross.shape[0] > 0 and cross.shape[1] > 0:
        cross.to_csv(path)

    print(f"  Cross-assets: {cross.shape}")
    return cross


def download_news():
    """Download financial news headlines from RSS feeds."""
    path = RAW_DIR / "news_headlines.json"
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        print(f"  News (cached): {len(data)} dates")
        return data

    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=stock+market+economy+federal+reserve&hl=en-US",
    ]
    data = {}
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if hasattr(entry, "published_parsed") and entry.published_parsed:
                    ds = (
                        f"{entry.published_parsed.tm_year}-"
                        f"{entry.published_parsed.tm_mon:02d}-"
                        f"{entry.published_parsed.tm_mday:02d}"
                    )
                    if ds not in data:
                        data[ds] = []
                    data[ds].append(entry.get("title", ""))
            print(f"    RSS: {len(feed.entries)} entries from {url[:50]}...")
        except Exception as e:
            print(f"    RSS failed: {url[:50]}... ({e})")

    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  News: {len(data)} dates")
    return data


def download_all():
    """Download everything. Returns dict of dataframes."""
    print("=" * 60)
    print("DOWNLOADING DATA")
    print("=" * 60)

    ib = connect_ibkr()

    data = {
        "spx": download_spx_daily(ib),
        "spx_intraday": download_spx_intraday(ib),
        "options": download_options_chain(ib),
        "vix": download_vix(ib),
        "cross_assets": download_cross_assets(),
        "news": download_news(),
    }

    if ib is not None:
        ib.disconnect()
        print("  IBKR disconnected.")

    has_intraday = data["spx_intraday"] is not None
    has_options = data["options"] is not None
    print(f"\nData summary:")
    print(f"  SPX daily: {len(data['spx'])} rows")
    print(f"  SPX intraday: {'YES' if has_intraday else 'NO (using daily RV)'}")
    print(f"  Options chain: {'YES' if has_options else 'NO (using VIX proxies)'}")
    print(f"  VIX: {len(data['vix'])} rows")
    print(f"  Cross-assets: {data['cross_assets'].shape}")
    print(f"  News: {len(data['news'])} dates")

    return data


if __name__ == "__main__":
    download_all()