#!/usr/bin/env python3
"""
Scrape TechPowerUp SSD specs locally and compute power-per-TB statistics.

What this script does
---------------------
1) Crawls TechPowerUp SSD specs pages starting from:
   https://www.techpowerup.com/ssd-specs/
2) Collects SSD detail-page URLs from listing pages
3) Visits each detail page and extracts:
   - vendor / model
   - capacity
   - interface / form factor / controller / NAND (when available)
   - idle / active / average power fields when present
4) Builds a flat CSV of raw SSD rows
5) Computes summary statistics for:
   - idle_w
   - active_w
   - avg_w
   - idle_w_per_tb
   - active_w_per_tb
   - avg_w_per_tb

Notes
-----
- Run this locally on your machine; this environment cannot scrape TechPowerUp directly.
- The page structure may evolve. This parser is designed to be resilient and uses
  multiple fallback extraction methods.
- You may need: pip install requests beautifulsoup4 pandas lxml numpy
- Respect the website's terms of use and robots rules when you run this.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import numpy as np
import pandas as pd
import requests

print(sys.executable)

from bs4 import BeautifulSoup

BASE_URL = "https://www.techpowerup.com"
START_URL = "https://www.techpowerup.com/ssd-specs/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Candidate labels seen on spec/detail pages. The parser will normalize labels.
POWER_LABEL_PATTERNS = {
    "idle_w": [
        r"\bidle power\b",
        r"\bpower consumption \(idle\)\b",
        r"\bidle\b",
    ],
    "active_w": [
        r"\bactive power\b",
        r"\bpower consumption \(active\)\b",
        r"\bmax active power\b",
        r"\bpower \(active\)\b",
    ],
    "avg_w": [
        r"\baverage power\b",
        r"\bavg\.? power\b",
        r"\btypical power\b",
        r"\bpower consumption \(average\)\b",
        r"\bpower \(average\)\b",
    ],
}

GENERIC_FIELD_PATTERNS = {
    "capacity_tb": [r"\bcapacity\b", r"\bformatted capacity\b"],
    "vendor": [r"\bmanufacturer\b", r"\bbrand\b", r"\bvendor\b"],
    "interface": [r"\binterface\b"],
    "form_factor": [r"\bform factor\b"],
    "controller": [r"\bcontroller\b"],
    "nand": [r"\bnand\b", r"\bflash\b"],
    "drain_class": [r"\benterprise\b", r"\bdatacenter\b", r"\bclient\b", r"\bconsumer\b"],
}

ENTERPRISE_KEYWORDS = [
    "enterprise", "data center", "datacenter", "dc ", "pm", "cd", "cm", "d7", "d5",
    "p4510", "p4610", "p5510", "p5520", "s4520", "s4510", "s4500", "micron 9300",
    "micron 9400", "solidigm d", "kioxia cd", "samsung pm"
]


def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def parse_power_w(text: str) -> Optional[float]:
    if text is None:
        return None
    s = normalize_whitespace(str(text)).lower()
    s = s.replace(",", "")
    # Handle ranges like "3.2 - 4.1 W" by taking midpoint
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*[-–]\s*([0-9]+(?:\.[0-9]+)?)\s*w", s)
    if m:
        return (float(m.group(1)) + float(m.group(2))) / 2.0
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*w", s)
    if m:
        return float(m.group(1))
    return None


def parse_capacity_tb(text: str) -> Optional[float]:
    if text is None:
        return None
    s = normalize_whitespace(str(text)).lower().replace(",", "")
    # Common capacity expressions
    for pat, div in [
        (r"([0-9]+(?:\.[0-9]+)?)\s*tb\b", 1.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*t\b", 1.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*gb\b", 1000.0),
        (r"([0-9]+(?:\.[0-9]+)?)\s*g\b", 1000.0),
    ]:
        m = re.search(pat, s)
        if m:
            return float(m.group(1)) / div
    return None


def maybe_enterprise(row: Dict[str, object]) -> bool:
    blob = " ".join(
        normalize_whitespace(str(v)).lower()
        for v in row.values()
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    )
    return any(k in blob for k in ENTERPRISE_KEYWORDS)


def get(session: requests.Session, url: str, sleep_s: float = 0.75) -> requests.Response:
    time.sleep(sleep_s)
    r = session.get(url, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r


def same_host(url: str) -> bool:
    return urlparse(url).netloc.endswith("techpowerup.com")


def collect_pagination_and_detail_links(session: requests.Session, start_url: str, max_pages: int = 250) -> Tuple[List[str], Set[str]]:
    """
    Crawl listing pages by following pagination links and collect SSD detail links.
    """
    to_visit = [start_url]
    visited_pages: Set[str] = set()
    detail_links: Set[str] = set()

    while to_visit and len(visited_pages) < max_pages:
        page_url = to_visit.pop(0)
        if page_url in visited_pages:
            continue

        print(f"[LIST] {page_url}")
        try:
            r = get(session, page_url)
        except Exception as e:
            print(f"  !! failed: {e}")
            visited_pages.add(page_url)
            continue

        visited_pages.add(page_url)
        soup = BeautifulSoup(r.text, "lxml")

        # Collect likely SSD detail links.
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            if not same_host(href):
                continue
            path = urlparse(href).path.rstrip("/")
            # Heuristic: SSD detail pages often live under /ssd-specs/<vendor-model>.html or similar.
            if "/ssd-specs/" in path and path != "/ssd-specs":
                # Avoid pagination/filter links; prefer paths with at least one extra slug
                if re.search(r"/ssd-specs/[^/?#]+", path):
                    detail_links.add(href)

        # Follow pagination-like links.
        for a in soup.find_all("a", href=True):
            href = urljoin(page_url, a["href"])
            if not same_host(href):
                continue
            path = urlparse(href).path.rstrip("/")
            q = urlparse(href).query.lower()
            if "/ssd-specs" not in path:
                continue
            text = normalize_whitespace(a.get_text(" ", strip=True)).lower()
            if href not in visited_pages and href not in to_visit:
                if (
                    href == start_url
                    or "page=" in q
                    or text in {"next", "older", "»", "›"}
                    or re.fullmatch(r"\d+", text or "")
                ):
                    to_visit.append(href)

    return sorted(visited_pages), detail_links


def find_key_value_pairs(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Extract key/value pairs from various possible detail-page layouts.
    """
    out: Dict[str, str] = {}

    # Strategy 1: tables
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) >= 2:
                key = normalize_whitespace(cells[0].get_text(" ", strip=True)).rstrip(":")
                val = normalize_whitespace(cells[1].get_text(" ", strip=True))
                if key and val and key.lower() not in out:
                    out[key.lower()] = val

    # Strategy 2: definition lists
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        for dt, dd in zip(dts, dds):
            key = normalize_whitespace(dt.get_text(" ", strip=True)).rstrip(":")
            val = normalize_whitespace(dd.get_text(" ", strip=True))
            if key and val and key.lower() not in out:
                out[key.lower()] = val

    # Strategy 3: generic "label: value" blocks
    body_text = soup.get_text("\n", strip=True)
    for line in body_text.splitlines():
        line = normalize_whitespace(line)
        if ":" in line and len(line) < 200:
            key, val = line.split(":", 1)
            key = key.strip().lower()
            val = val.strip()
            if key and val and key not in out:
                out[key] = val

    return out


def match_field(kv: Dict[str, str], patterns: Iterable[str]) -> Optional[str]:
    for key, val in kv.items():
        for pat in patterns:
            if re.search(pat, key, flags=re.I):
                return val
    return None


def parse_detail_page(session: requests.Session, url: str) -> Dict[str, object]:
    print(f"[DETAIL] {url}")
    r = get(session, url)
    soup = BeautifulSoup(r.text, "lxml")

    title = ""
    if soup.title:
        title = normalize_whitespace(soup.title.get_text(" ", strip=True))
    h1 = soup.find(["h1", "h2"])
    if h1:
        heading = normalize_whitespace(h1.get_text(" ", strip=True))
        if len(heading) > len(title):
            title = heading

    kv = find_key_value_pairs(soup)

    row: Dict[str, object] = {
        "url": url,
        "page_title": title,
        "vendor": match_field(kv, GENERIC_FIELD_PATTERNS["vendor"]),
        "capacity_text": match_field(kv, GENERIC_FIELD_PATTERNS["capacity_tb"]),
        "interface": match_field(kv, GENERIC_FIELD_PATTERNS["interface"]),
        "form_factor": match_field(kv, GENERIC_FIELD_PATTERNS["form_factor"]),
        "controller": match_field(kv, GENERIC_FIELD_PATTERNS["controller"]),
        "nand": match_field(kv, GENERIC_FIELD_PATTERNS["nand"]),
        "idle_power_text": match_field(kv, POWER_LABEL_PATTERNS["idle_w"]),
        "active_power_text": match_field(kv, POWER_LABEL_PATTERNS["active_w"]),
        "avg_power_text": match_field(kv, POWER_LABEL_PATTERNS["avg_w"]),
    }

    # Fallback capacity from title if field missing
    if not row["capacity_text"] and title:
        row["capacity_text"] = title

    # Try to parse model/vendor from title
    if title:
        # Strip common suffix
        cleaned = re.sub(r"\s*SSD Specifications.*$", "", title, flags=re.I)
        row["model_guess"] = cleaned
        if not row["vendor"]:
            first = cleaned.split()[0] if cleaned.split() else None
            row["vendor"] = first

    row["capacity_tb"] = parse_capacity_tb(str(row["capacity_text"]) if row["capacity_text"] else None)
    row["idle_w"] = parse_power_w(str(row["idle_power_text"]) if row["idle_power_text"] else None)
    row["active_w"] = parse_power_w(str(row["active_power_text"]) if row["active_power_text"] else None)
    row["avg_w"] = parse_power_w(str(row["avg_power_text"]) if row["avg_power_text"] else None)

    # Fallback average if active exists and avg missing.
    if row["avg_w"] is None and row["active_w"] is not None:
        row["avg_w"] = row["active_w"]

    row["enterprise_flag_heuristic"] = maybe_enterprise(row)
    return row


def summarize(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"count": 0}
    return {
        "count": int(s.count()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std(ddof=1)) if s.count() > 1 else 0.0,
        "min": float(s.min()),
        "p10": float(s.quantile(0.10)),
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "max": float(s.max()),
    }


def build_stats(raw_df: pd.DataFrame, segment: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_df.copy()

    # Basic filters
    df["capacity_tb"] = pd.to_numeric(df["capacity_tb"], errors="coerce")
    df["idle_w"] = pd.to_numeric(df["idle_w"], errors="coerce")
    df["active_w"] = pd.to_numeric(df["active_w"], errors="coerce")
    df["avg_w"] = pd.to_numeric(df["avg_w"], errors="coerce")

    if segment == "enterprise":
        df = df[df["enterprise_flag_heuristic"] == True].copy()
    elif segment == "consumer":
        df = df[df["enterprise_flag_heuristic"] == False].copy()

    valid_cap = df["capacity_tb"].notna() & (df["capacity_tb"] > 0)
    for src, dst in [
        ("idle_w", "idle_w_per_tb"),
        ("active_w", "active_w_per_tb"),
        ("avg_w", "avg_w_per_tb"),
    ]:
        df[dst] = np.where(valid_cap & df[src].notna(), df[src] / df["capacity_tb"], np.nan)

    stats_rows = []
    for metric in [
        "idle_w", "active_w", "avg_w",
        "idle_w_per_tb", "active_w_per_tb", "avg_w_per_tb",
    ]:
        rec = summarize(df[metric])
        rec["metric"] = metric
        rec["segment"] = segment
        stats_rows.append(rec)

    stats_df = pd.DataFrame(stats_rows)

    # Optional grouped summaries
    group_frames = []
    for group_col in ["interface", "form_factor", "vendor"]:
        if group_col in df.columns:
            grouped = []
            for g, gdf in df.groupby(group_col, dropna=True):
                rec = summarize(gdf["avg_w_per_tb"])
                rec["group_by"] = group_col
                rec["group"] = g
                grouped.append(rec)
            if grouped:
                group_frames.append(pd.DataFrame(grouped))
    grouped_df = pd.concat(group_frames, ignore_index=True) if group_frames else pd.DataFrame()

    return stats_df, grouped_df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("techpowerup_ssd_output"))
    ap.add_argument("--max-pages", type=int, default=250)
    ap.add_argument("--max-details", type=int, default=0, help="0 means no cap")
    ap.add_argument("--segment", choices=["all", "enterprise", "consumer"], default="enterprise")
    ap.add_argument("--sleep", type=float, default=0.75)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()

    global get
    def get(session_: requests.Session, url: str, sleep_s: float = args.sleep) -> requests.Response:
        time.sleep(sleep_s)
        r = session_.get(url, headers=HEADERS, timeout=45)
        r.raise_for_status()
        return r

    print("Collecting listing and detail links...")
    listing_pages, detail_links = collect_pagination_and_detail_links(session, START_URL, max_pages=args.max_pages)

    detail_links = sorted(detail_links)
    if args.max_details and args.max_details > 0:
        detail_links = detail_links[: args.max_details]

    print(f"Listing pages visited: {len(listing_pages)}")
    print(f"Detail links found: {len(detail_links)}")

    raw_rows: List[Dict[str, object]] = []
    failures: List[Dict[str, str]] = []

    for i, url in enumerate(detail_links, start=1):
        try:
            print(f"{i}/{len(detail_links)}")
            raw_rows.append(parse_detail_page(session, url))
        except Exception as e:
            failures.append({"url": url, "error": str(e)})
            print(f"  !! detail failed: {e}")

    raw_df = pd.DataFrame(raw_rows)
    failures_df = pd.DataFrame(failures)

    raw_csv = args.outdir / "techpowerup_ssd_raw.csv"
    raw_df.to_csv(raw_csv, index=False)

    failures_csv = args.outdir / "techpowerup_ssd_failures.csv"
    failures_df.to_csv(failures_csv, index=False)

    stats_df, grouped_df = build_stats(raw_df, segment=args.segment)

    stats_csv = args.outdir / f"techpowerup_ssd_stats_{args.segment}.csv"
    stats_df.to_csv(stats_csv, index=False)

    grouped_csv = args.outdir / f"techpowerup_ssd_grouped_avg_w_per_tb_{args.segment}.csv"
    grouped_df.to_csv(grouped_csv, index=False)

    # Also save a filtered analysis set with normalized per-TB columns.
    analysis_df = raw_df.copy()
    analysis_df["capacity_tb"] = pd.to_numeric(analysis_df["capacity_tb"], errors="coerce")
    for src, dst in [("idle_w", "idle_w_per_tb"), ("active_w", "active_w_per_tb"), ("avg_w", "avg_w_per_tb")]:
        analysis_df[src] = pd.to_numeric(analysis_df[src], errors="coerce")
        analysis_df[dst] = np.where(
            analysis_df["capacity_tb"].notna() & (analysis_df["capacity_tb"] > 0) & analysis_df[src].notna(),
            analysis_df[src] / analysis_df["capacity_tb"],
            np.nan,
        )

    if args.segment == "enterprise":
        analysis_df = analysis_df[analysis_df["enterprise_flag_heuristic"] == True].copy()
    elif args.segment == "consumer":
        analysis_df = analysis_df[analysis_df["enterprise_flag_heuristic"] == False].copy()

    analysis_csv = args.outdir / f"techpowerup_ssd_analysis_{args.segment}.csv"
    analysis_df.to_csv(analysis_csv, index=False)

    print("\nDone.")
    print(f"Raw rows      : {raw_csv}")
    print(f"Failures      : {failures_csv}")
    print(f"Stats         : {stats_csv}")
    print(f"Grouped stats : {grouped_csv}")
    print(f"Analysis rows : {analysis_csv}")

    if not stats_df.empty:
        print("\nTop-level stats:")
        print(stats_df.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
