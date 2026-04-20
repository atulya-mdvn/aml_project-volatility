"""P3: Financial text analysis using real news data + Claude API."""
import json
import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from src.shared.config import PROC_DIR, RAW_DIR, SPLIT_DATE

ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
NYT_KEY = os.environ.get("NYT_API_KEY", "")

# Weekly for training period, DAILY for test period (where evaluation happens)
ANALYZE_EVERY_TRAIN = 5   # every 5 days for training (pre-2020)
ANALYZE_EVERY_TEST = 1    # every day for test period (2020+)


def download_nyt_headlines(start_year=2000, end_year=2025):
    cache_path = RAW_DIR / "nyt_headlines.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"  NYT headlines (cached): {len(data)} dates", flush=True)
        return data

    if not NYT_KEY:
        print("  NYT API key not set — skipping", flush=True)
        return {}

    print(f"  Downloading NYT headlines {start_year}-{end_year}...", flush=True)
    headlines = {}
    business_sections = {"business", "Business", "Business Day", "DealBook", "Financial",
                         "Economy", "Markets", "Your Money", "Economix", "Wall Street"}

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json?api-key={NYT_KEY}"
            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 429:
                    time.sleep(60)
                    resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    time.sleep(12); continue
                articles = resp.json().get("response", {}).get("docs", [])
                for article in articles:
                    section = article.get("section_name", "") or ""
                    news_desk = article.get("news_desk", "") or ""
                    is_fin = (section in business_sections or news_desk in business_sections or
                              "business" in section.lower() or "financial" in section.lower() or
                              "economy" in news_desk.lower() or "market" in news_desk.lower())
                    if not is_fin: continue
                    headline = article.get("headline", {}).get("main", "")
                    pub_date = article.get("pub_date", "")
                    if headline and pub_date:
                        ds = pub_date[:10]
                        if ds not in headlines: headlines[ds] = []
                        headlines[ds].append(headline)
            except Exception as e:
                print(f"    {year}-{month:02d}: error ({e})", flush=True)
            time.sleep(12)
        print(f"    {year}: {sum(1 for d in headlines if d.startswith(str(year)))} dates", flush=True)

    with open(cache_path, 'w') as f:
        json.dump(headlines, f)
    print(f"  NYT headlines: {len(headlines)} total dates", flush=True)
    return headlines


def download_fomc_data():
    cache_path = RAW_DIR / "fomc_data.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            data = json.load(f)
        print(f"  FOMC data (cached): {len(data)} documents", flush=True)
        return data

    print("  Downloading FOMC data...", flush=True)
    fomc_data = {}
    try:
        from FedTools import MonetaryPolicyCommittee, FederalReserveMins
        statements = MonetaryPolicyCommittee().find_statements()
        for date_idx, row in statements.iterrows():
            ds = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
            text = str(row.iloc[0]) if len(row) > 0 else ""
            if text and len(text) > 100:
                fomc_data[ds] = {"type": "statement", "text": text[:3000], "date": ds}
        minutes = FederalReserveMins().find_minutes()
        for date_idx, row in minutes.iterrows():
            ds = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
            text = str(row.iloc[0]) if len(row) > 0 else ""
            if text and len(text) > 100:
                fomc_data[ds] = {"type": "minutes", "text": text[:3000], "date": ds}
    except Exception as e:
        print(f"    FOMC failed: {e}", flush=True)

    with open(cache_path, 'w') as f:
        json.dump(fomc_data, f)
    print(f"  FOMC data: {len(fomc_data)} documents", flush=True)
    return fomc_data


def scrape_rss_headlines():
    import feedparser
    feeds = [
        "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC&region=US&lang=en-US",
        "https://news.google.com/rss/search?q=stock+market+economy+federal+reserve&hl=en-US",
    ]
    headlines = {}
    for url in feeds:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                if hasattr(entry, 'published_parsed') and entry.published_parsed:
                    ds = f"{entry.published_parsed.tm_year}-{entry.published_parsed.tm_mon:02d}-{entry.published_parsed.tm_mday:02d}"
                    if ds not in headlines: headlines[ds] = []
                    title = entry.get('title', '')
                    if title: headlines[ds].append(title)
        except: pass
    return headlines


def build_synthetic_headline(row):
    ret = row.get("log_return", 0)
    vix = row.get("vix_level", 15)
    if ret < -0.02: text = "Markets fell sharply as selling pressure intensified"
    elif ret < -0.005: text = "Stocks declined with cautious sentiment"
    elif ret > 0.02: text = "Markets rallied strongly on improved confidence"
    elif ret > 0.005: text = "Stocks advanced modestly in positive session"
    else: text = "Markets traded flat awaiting economic data"
    if vix > 30: text += ". VIX above 30 indicating stress"
    elif vix > 20: text += ". VIX elevated above average"
    return text


def analyze_with_claude(text_content, date_str, content_type="headlines"):
    if not ANTHROPIC_KEY:
        return None
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_KEY, timeout=30.0)

    if content_type == "fomc":
        prompt = f"""Analyze this FOMC document from {date_str}. Return ONLY a JSON object:
{{"sentiment": <-1 to 1>, "uncertainty": <0 to 1>, "hawkish_dovish": <-1 to 1>, "risk_appetite": <-1 to 1>, "vol_expectation": <-1 to 1>, "systemic_risk": <0 to 1>}}

Document: {text_content[:1500]}"""
    else:
        prompt = f"""Analyze these financial headlines from {date_str}. Return ONLY a JSON object:
{{"sentiment": <-1 to 1>, "uncertainty": <0 to 1>, "hawkish_dovish": <-1 to 1>, "risk_appetite": <-1 to 1>, "vol_expectation": <-1 to 1>, "systemic_risk": <0 to 1>}}

Headlines: {text_content[:1500]}"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}]
        )
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"    API error on {date_str}: {e}", flush=True)
        return None


def rule_based_fallback(row):
    ret = row.get("log_return", 0)
    vix = row.get("vix_level", 15)
    return {
        "sentiment": float(np.clip(ret * 20, -1, 1)),
        "uncertainty": float(np.clip((vix - 15) / 20, 0, 1)),
        "hawkish_dovish": 0.0,
        "risk_appetite": float(np.clip(-((vix - 18) / 15), -1, 1)),
        "vol_expectation": float(np.clip((vix - 18) / 15, -1, 1)),
        "systemic_risk": float(np.clip((vix - 25) / 20, 0, 1)),
    }


def train_finbert(feat, news_data):
    print("\n" + "=" * 60, flush=True)
    print("P3: FINANCIAL TEXT ANALYSIS (Real Data + Claude)", flush=True)
    print("=" * 60, flush=True)

    nyt_headlines = download_nyt_headlines()
    fomc_data = download_fomc_data()
    rss_headlines = scrape_rss_headlines()

    all_headlines = {}
    if news_data: all_headlines.update(news_data)
    all_headlines.update(nyt_headlines)
    for ds, hl in rss_headlines.items():
        if ds not in all_headlines: all_headlines[ds] = hl
        else: all_headlines[ds].extend(hl)

    trading_days = feat.index
    real_coverage = sum(1 for d in trading_days if d.strftime("%Y-%m-%d") in all_headlines)
    print(f"\n  Real headline coverage: {real_coverage}/{len(trading_days)} ({100*real_coverage/len(trading_days):.0f}%)", flush=True)
    print(f"  FOMC documents: {len(fomc_data)}", flush=True)
    print(f"  Using Claude: {'YES' if ANTHROPIC_KEY else 'NO (fallback)'}", flush=True)
    print(f"  Training period: every {ANALYZE_EVERY_TRAIN} days | Test period: every {ANALYZE_EVERY_TEST} day(s)", flush=True)

    # precompute FOMC dates
    fomc_dates_set = set()
    for fd in fomc_data.keys():
        fd_ts = pd.Timestamp(fd)
        for offset in [-1, 0, 1]:
            fomc_dates_set.add((fd_ts + pd.Timedelta(days=offset)).strftime("%Y-%m-%d"))

    cache_path = PROC_DIR / "p3_claude_cache.json"
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"  Cached analyses: {len(cache)}", flush=True)
    else:
        cache = {}

    # find failed FOMC dates in cache (entries that are rule-based fallback on FOMC days)
    # and remove them so they get retried
    retried = 0
    for ds in list(cache.keys()):
        if ds in fomc_dates_set and cache[ds].get("hawkish_dovish", 0) == 0.0 and cache[ds].get("systemic_risk", 0) == 0.0:
            # likely a fallback entry on a FOMC day — retry it
            if ANTHROPIC_KEY:
                del cache[ds]
                retried += 1
    if retried > 0:
        print(f"  Retrying {retried} previously failed FOMC dates", flush=True)

    results = []
    total = len(feat.index)
    claude_calls = 0
    fallback_calls = 0
    last_analysis = None
    split_ts = pd.Timestamp(SPLIT_DATE)

    print(f"  Starting analysis ({total} days)...", flush=True)

    for i, d in enumerate(feat.index):
        ds = d.strftime("%Y-%m-%d")

        if ds in cache:
            results.append(cache[ds])
            last_analysis = cache[ds]
            continue

        is_fomc = ds in fomc_dates_set
        is_test = d >= split_ts
        analyze_every = ANALYZE_EVERY_TEST if is_test else ANALYZE_EVERY_TRAIN

        # skip non-analysis days
        if i % analyze_every != 0 and not is_fomc:
            if last_analysis is not None:
                results.append(last_analysis)
                cache[ds] = last_analysis
            else:
                fb = rule_based_fallback(feat.loc[d])
                results.append(fb)
                cache[ds] = fb
                fallback_calls += 1
            continue

        # FOMC day — highest priority
        if is_fomc and ANTHROPIC_KEY:
            fomc_text = None
            for fd, fdata in fomc_data.items():
                if abs((d - pd.Timestamp(fd)).days) <= 1:
                    fomc_text = fdata["text"]; break
            if fomc_text:
                analysis = analyze_with_claude(fomc_text, ds, "fomc")
                if analysis:
                    cache[ds] = analysis
                    results.append(analysis)
                    last_analysis = analysis
                    claude_calls += 1
                    time.sleep(0.5)
                    if claude_calls % 10 == 0:
                        with open(cache_path, 'w') as f: json.dump(cache, f)
                    if claude_calls % 25 == 0:
                        print(f"  Day {i+1}/{total} | Claude: {claude_calls} | Fallback: {fallback_calls}", flush=True)
                    continue

        # headlines
        day_headlines = all_headlines.get(ds, [])
        if len(day_headlines) > 0 and ANTHROPIC_KEY:
            combined = "\n".join([f"- {h}" for h in day_headlines[:5]])
            analysis = analyze_with_claude(combined, ds, "headlines")
            if analysis:
                cache[ds] = analysis
                results.append(analysis)
                last_analysis = analysis
                claude_calls += 1
                time.sleep(0.5)
                if claude_calls % 10 == 0:
                    with open(cache_path, 'w') as f: json.dump(cache, f)
                if claude_calls % 25 == 0:
                    print(f"  Day {i+1}/{total} | Claude: {claude_calls} | Fallback: {fallback_calls}", flush=True)
                continue

        # synthetic
        if ANTHROPIC_KEY and len(day_headlines) == 0:
            synthetic = build_synthetic_headline(feat.loc[d])
            analysis = analyze_with_claude(synthetic, ds, "headlines")
            if analysis:
                cache[ds] = analysis
                results.append(analysis)
                last_analysis = analysis
                claude_calls += 1
                time.sleep(0.5)
                if claude_calls % 10 == 0:
                    with open(cache_path, 'w') as f: json.dump(cache, f)
                if claude_calls % 25 == 0:
                    print(f"  Day {i+1}/{total} | Claude: {claude_calls} | Fallback: {fallback_calls}", flush=True)
                continue

        fb = rule_based_fallback(feat.loc[d])
        results.append(fb)
        cache[ds] = fb
        last_analysis = fb
        fallback_calls += 1

    with open(cache_path, 'w') as f:
        json.dump(cache, f)
    print(f"\n  Done: claude={claude_calls}, fallback={fallback_calls}", flush=True)

    text_df = pd.DataFrame(results, index=feat.index)
    text_df["finbert_sentiment"] = text_df.get("sentiment", 0)
    text_df["finbert_uncertainty"] = text_df.get("uncertainty", 0)

    for col in ["finbert_sentiment", "finbert_uncertainty"]:
        text_df[f"{col}_5d"] = text_df[col].rolling(5).mean()
        text_df[f"{col}_22d"] = text_df[col].rolling(22).mean()

    keep_cols = ["finbert_sentiment", "finbert_uncertainty",
                 "finbert_sentiment_5d", "finbert_uncertainty_5d",
                 "finbert_sentiment_22d", "finbert_uncertainty_22d"]

    for extra in ["hawkish_dovish", "risk_appetite", "vol_expectation", "systemic_risk"]:
        if extra in text_df.columns:
            keep_cols.append(extra)
            text_df[f"{extra}_5d"] = text_df[extra].rolling(5).mean()
            text_df[f"{extra}_22d"] = text_df[extra].rolling(22).mean()
            keep_cols.extend([f"{extra}_5d", f"{extra}_22d"])

    output = text_df[keep_cols].copy()
    output.to_csv(PROC_DIR / "p3_finbert.csv")
    print(f"  P3 features: {output.shape}", flush=True)
    return output
