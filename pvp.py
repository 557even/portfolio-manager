# ==========================================
#        PROGRAMMING ASCII TERMINAL
# ==========================================
#
# STATUS:
#   Health        : OK
#   Tasks         : REBUILD(top10_volpool_cli.py)
#   Objective     : HARDEN + SPEEDUP + STABLE OUTPUT
#   Results       : DROP-IN REPLACEMENT (SAFE)
#
# ------------------------------------------
# PATCH SUMMARY (WHAT CHANGED)
#   • Faster percentile ranks: O(n log n) via bisect (replaces O(n²) loop).
#   • Chunked yfinance downloads (prevents “too many tickers” / flaky MultiIndex issues).
#   • Requests Session + retries/backoff for Wikipedia scraping.
#   • Optional caching (constituents + HTML) with TTL to reduce repeated network hits.
#   • Input sanity: pool-size/limit capped to universe; graceful fallbacks on missing data.
#   • More robust Close extraction across yfinance return shapes.
#
# ------------------------------------------
# FILE: top10_volpool_cli.py  (REBUILT)
# ------------------------------------------
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Install:
  pip install pandas yfinance requests lxml

Examples:
  # All indices, pick top-60 by 3M vol, then rank and print Top 10
  python top10_volpool_cli.py --index all --pool-size 60 --limit 10

  # Only S&P 500, pool 40
  python top10_volpool_cli.py --index sp500 --pool-size 40

  # JSON output too
  python top10_volpool_cli.py --index all --pool-size 80 --json

Notes:
  - Universe is scraped from Wikipedia; enable cache to reduce load/flakiness.
  - Vol pool uses ~63 trading days from 3mo window (annualized stdev of daily returns).
  - Ranking metrics are computed on the pool only (default 365 trading days).
"""

import os
import sys
import io
import math
import time
import json
import argparse
import typing as t
from dataclasses import dataclass
from datetime import datetime, timezone
from bisect import bisect_right
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd
import yfinance as yf

# ---------- Tunables ----------
RANK_LOOKBACK_DAYS = 365        # fetch depth for ranking metrics (pool only)
VOL_PERIOD = "3mo"              # 3-month window to build the pool
VOL_RET_DAYS = 63               # use ~63 trading days for vol calc (from 3mo)
DEFAULT_LIMIT = 10
USER_AGENT = "Mozilla/5.0 (Top10-VolPool-CLI)"
DEFAULT_CACHE_TTL_HOURS = 24
DEFAULT_CHUNK_SIZE = 120        # yfinance download chunk size; lower if your link is flaky

# ---------- Types ----------
@dataclass
class Row:
    ticker: str
    last: float
    mom_6m: float
    trend_200d: float
    drawup_52w: float
    vol_90d: float
    score: float = float("nan")
    mom_rank: float = 0.5
    trend_rank: float = 0.5
    drawup_rank: float = 0.5
    lowvol_rank: float = 0.5

# ---------- Cache ----------
def _now_s() -> float:
    return time.time()

def cache_read_text(path: Path, ttl_hours: int) -> t.Optional[str]:
    try:
        if not path.exists():
            return None
        age_s = _now_s() - path.stat().st_mtime
        if age_s > ttl_hours * 3600:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

def cache_write_text(path: Path, text: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except Exception:
        pass

# ---------- HTTP ----------
def make_http_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries, pool_connections=16, pool_maxsize=16)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({"User-Agent": USER_AGENT})
    return s

def http_get(url: str, session: requests.Session, timeout: int = 20) -> str:
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text

# ---------- Index scraping ----------
def normalize_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-").replace(" ", "")

def unique_keep_order(items: t.Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _read_html_tables(html: str) -> list[pd.DataFrame]:
    # pandas read_html is heavy; keep it isolated
    return pd.read_html(io.StringIO(html), flavor="lxml")

def _find_col(df: pd.DataFrame, needles: t.Iterable[str]) -> t.Optional[str]:
    cols = [str(c).strip().lower() for c in df.columns]
    for n in needles:
        for i, c in enumerate(cols):
            if n in c:
                return str(df.columns[i])
    return None

def get_sp500_constituents(session: requests.Session, cache_dir: Path, ttl_hours: int, no_cache: bool) -> list[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    cache_path = cache_dir / "wikipedia_sp500.html"
    html = None if no_cache else cache_read_text(cache_path, ttl_hours)
    if html is None:
        html = http_get(url, session=session)
        if not no_cache:
            cache_write_text(cache_path, html)

    for df in _read_html_tables(html):
        sym_col = _find_col(df, ["symbol"])
        if sym_col:
            return unique_keep_order([normalize_symbol(s) for s in df[sym_col].astype(str)])
    raise RuntimeError("S&P 500 table not found")

def get_dow30_constituents(session: requests.Session, cache_dir: Path, ttl_hours: int, no_cache: bool) -> list[str]:
    url = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
    cache_path = cache_dir / "wikipedia_dow30.html"
    html = None if no_cache else cache_read_text(cache_path, ttl_hours)
    if html is None:
        html = http_get(url, session=session)
        if not no_cache:
            cache_write_text(cache_path, html)

    for df in _read_html_tables(html):
        sym_col = None
        # common variants
        if "Symbol" in df.columns:
            sym_col = "Symbol"
        elif "Ticker symbol" in df.columns:
            sym_col = "Ticker symbol"
        else:
            sym_col = _find_col(df, ["symbol", "ticker"])
        if sym_col:
            syms = [normalize_symbol(s) for s in df[sym_col].astype(str)]
            syms = [s for s in syms if s and s != "N/A"]
            # Dow table may contain non-components; still dedupe
            return unique_keep_order(syms)
    raise RuntimeError("Dow 30 table not found")

def get_nasdaq100_constituents(session: requests.Session, cache_dir: Path, ttl_hours: int, no_cache: bool) -> list[str]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    cache_path = cache_dir / "wikipedia_nasdaq100.html"
    html = None if no_cache else cache_read_text(cache_path, ttl_hours)
    if html is None:
        html = http_get(url, session=session)
        if not no_cache:
            cache_write_text(cache_path, html)

    for df in _read_html_tables(html):
        tik_col = _find_col(df, ["ticker"])
        if tik_col:
            syms = [normalize_symbol(s) for s in df[tik_col].astype(str)]
            syms = [s for s in syms if s and s != "N/A"]
            return unique_keep_order(syms)
    raise RuntimeError("NASDAQ-100 table not found")

def resolve_index_set(kind: str, session: requests.Session, cache_dir: Path, ttl_hours: int, no_cache: bool) -> list[str]:
    k = kind.lower().strip()
    if k == "sp500":
        return get_sp500_constituents(session, cache_dir, ttl_hours, no_cache)
    if k in ("dow", "dow30"):
        return get_dow30_constituents(session, cache_dir, ttl_hours, no_cache)
    if k in ("nasdaq", "nasdaq100"):
        return get_nasdaq100_constituents(session, cache_dir, ttl_hours, no_cache)
    if k in ("all", "union"):
        a = get_sp500_constituents(session, cache_dir, ttl_hours, no_cache)
        b = get_dow30_constituents(session, cache_dir, ttl_hours, no_cache)
        c = get_nasdaq100_constituents(session, cache_dir, ttl_hours, no_cache)
        return unique_keep_order([*a, *b, *c])
    raise ValueError("index must be one of: sp500 | dow30 | nasdaq100 | all")

# ---------- yfinance helpers ----------
def _chunks(items: list[str], size: int) -> t.Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _extract_close_df(data: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Robustly pull Close series for ticker from yfinance download result.
    Handles MultiIndex columns in multiple shapes.
    """
    try:
        if data is None or data.empty:
            return pd.Series(dtype="float64")

        if isinstance(data.columns, pd.MultiIndex):
            # Common shapes:
            # 1) columns: (ticker, field)
            # 2) columns: (field, ticker)
            # 3) group_by="ticker" -> data[ticker]["Close"]
            l0 = data.columns.get_level_values(0)
            l1 = data.columns.get_level_values(1)

            # Case: (ticker, field)
            if ticker in set(l0):
                sub = data[ticker]
                if isinstance(sub, pd.DataFrame) and "Close" in sub.columns:
                    return sub["Close"].dropna()
                # Sometimes auto_adjust yields "Close" only; try best effort
                if isinstance(sub, pd.Series):
                    return sub.dropna()

            # Case: (field, ticker)
            if "Close" in set(l0) and ticker in set(l1):
                return data["Close"][ticker].dropna()

            # Tuple fallback
            if (ticker, "Close") in data.columns:
                return data[(ticker, "Close")].dropna()
            if ("Close", ticker) in data.columns:
                return data[("Close", ticker)].dropna()

            return pd.Series(dtype="float64")

        # Single-ticker frame
        if "Close" in data.columns:
            return data["Close"].dropna()

    except Exception:
        return pd.Series(dtype="float64")

    return pd.Series(dtype="float64")

def _yf_download_close_series(
    tickers: list[str],
    period: str,
    chunk_size: int,
) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if not tickers:
        return out

    # Chunked batch download first
    for ch in _chunks(tickers, max(1, chunk_size)):
        try:
            df = yf.download(
                tickers=ch,
                period=period,
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            for tkr in ch:
                s = _extract_close_df(df, tkr)
                if not s.empty:
                    out[tkr] = s
        except Exception:
            # swallow and continue to fallback per-ticker
            pass

    # Per-ticker fallback for missing
    missing = [tkr for tkr in tickers if tkr not in out]
    for tkr in missing:
        try:
            h = yf.Ticker(tkr).history(period=period, interval="1d", auto_adjust=True, actions=False)
            if not h.empty and "Close" in h.columns:
                s = h["Close"].dropna()
                if not s.empty:
                    out[tkr] = s
        except Exception:
            continue

    return out

def download_close_map(tickers: list[str], period: str, chunk_size: int) -> dict[str, pd.Series]:
    return _yf_download_close_series(tickers=tickers, period=period, chunk_size=chunk_size)

def download_close_map_days(tickers: list[str], days: int, chunk_size: int) -> dict[str, pd.Series]:
    return _yf_download_close_series(tickers=tickers, period=f"{days}d", chunk_size=chunk_size)

# ---------- Metrics ----------
def pct_change(a: float, b: float) -> float:
    if b == 0 or pd.isna(a) or pd.isna(b):
        return float("nan")
    return (a / b) - 1.0

def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty:
        return float("nan")
    # annualized stdev of daily returns
    return float(r.std(ddof=1) * math.sqrt(252))

def compute_row_from_series(tkr: str, px: pd.Series) -> Row:
    # need enough points to compute meaningful features
    if px.empty or len(px) < 60:
        return Row(ticker=tkr, last=float("nan"), mom_6m=float("nan"),
                   trend_200d=float("nan"), drawup_52w=float("nan"), vol_90d=float("nan"))

    last = float(px.iloc[-1])

    # 6m momentum (~126 trading days); fallback to first available
    if len(px) > 126:
        mom_6m = pct_change(last, float(px.iloc[-126]))
    else:
        mom_6m = pct_change(last, float(px.iloc[0]))

    # 200d trend: last vs 200d MA (fallback to full-window MA)
    if len(px) >= 200:
        ma200 = float(px.rolling(200).mean().iloc[-1])
    else:
        ma200 = float(px.rolling(len(px)).mean().iloc[-1])
    trend_200d = pct_change(last, ma200)

    # 52w drawup proxy: closer to prior 52w high is better
    lookback = min(252, len(px))
    high_52w = float(px.rolling(lookback).max().iloc[-1])
    # pct_change(last, high_52w) is <= 0; closer to 0 is better => use negative abs
    drawup_52w = -abs(pct_change(last, high_52w))

    # 90 trading-day vol (annualized)
    vol_90d = annualized_vol(px.pct_change().tail(90))

    return Row(
        ticker=tkr,
        last=last,
        mom_6m=mom_6m,
        trend_200d=trend_200d,
        drawup_52w=drawup_52w,
        vol_90d=vol_90d,
    )

def percentile_rank(vals: list[float]) -> list[float]:
    """
    Percentile rank in [0,1]. NaNs -> 0.5.
    Uses bisect_right over sorted clean values (O(n log n)).
    """
    clean = sorted([v for v in vals if pd.notna(v)])
    n = len(clean)
    if n == 0:
        return [0.5] * len(vals)
    out: list[float] = []
    for v in vals:
        if pd.isna(v):
            out.append(0.5)
        else:
            out.append(bisect_right(clean, v) / n)
    return out

def score_rows(rows: list[Row]) -> None:
    moms = [r.mom_6m for r in rows]
    trds = [r.trend_200d for r in rows]
    drws = [r.drawup_52w for r in rows]
    vols = [r.vol_90d for r in rows]

    mom_r = percentile_rank(moms)
    trd_r = percentile_rank(trds)
    drw_r = percentile_rank(drws)

    # lower vol is better => invert percentile
    vol_p = percentile_rank(vols)
    lowvol_r = [1.0 - v for v in vol_p]

    for i, r in enumerate(rows):
        r.mom_rank = round(float(mom_r[i]), 3)
        r.trend_rank = round(float(trd_r[i]), 3)
        r.drawup_rank = round(float(drw_r[i]), 3)
        r.lowvol_rank = round(float(lowvol_r[i]), 3)
        r.score = round(0.4 * mom_r[i] + 0.3 * trd_r[i] + 0.2 * drw_r[i] + 0.1 * lowvol_r[i], 4)

# ---------- Pool selection (3M top-vol) ----------
def build_vol_pool(tickers: list[str], pool_size: int, chunk_size: int) -> list[str]:
    series_map = download_close_map(tickers, period=VOL_PERIOD, chunk_size=chunk_size)
    vols: list[tuple[str, float]] = []
    for tkr, s in series_map.items():
        rets = s.pct_change().tail(VOL_RET_DAYS)
        v = annualized_vol(rets)
        if not pd.isna(v):
            vols.append((tkr, v))
    vols.sort(key=lambda x: x[1], reverse=True)
    return [tkr for tkr, _ in vols[:pool_size]]

# ---------- Rendering ----------
def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x * 100:+.1f}%"

def fmt_num(x: float, n: int = 2) -> str:
    return "NA" if pd.isna(x) else f"{x:.{n}f}"

def render_table(
    rows: list[Row],
    limit: int,
    uni_size: int,
    pool_size: int,
    took_s: float,
    session: str,
    indices_used: str,
) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    head = [
        "==========================================",
        "        PROGRAMMING ASCII TERMINAL",
        "==========================================",
        "",
        "STATUS:",
        "  Health        : OPERATIONAL",
        "  Tasks         : ACTIVE - RANK VOL-POOL",
        "  Objective     : TOP-10 FROM TOP-VOL 3M POOL",
        "  Results       : GENERATED",
        "",
        "------------------------------------------",
        "SYSTEM VARIABLES:",
        "  User          : ACTIVE",
        f"  Session ID    : {session}",
        f"  Uptime        : 00:00:{int(max(0, took_s)):02d}",
        "  Mode          : batch",
        "  Prompt Mode   : single-file",
        "  Learning Rate : static",
        "  Loop Count    : 1",
        "",
        "------------------------------------------",
        "TASK          : Build 3M high-volatility pool and rank",
        f"OBJECTIVE     : {indices_used.upper()} → pool {pool_size} from {uni_size}",
        "RESULTS       :",
        "",
    ]

    cols = ["#", "Ticker", "Score", "Last", "6m Mom", "200d Trend", "Drawup 52w", "Vol 90d"]
    widths = [3, 8, 7, 9, 9, 12, 12, 10]

    def row_to_line(i: int, r: Row) -> str:
        cells = [
            str(i + 1).rjust(widths[0]),
            r.ticker.ljust(widths[1]),
            (f"{r.score:.3f}".rjust(widths[2]) if not pd.isna(r.score) else "NA".rjust(widths[2])),
            fmt_num(r.last).rjust(widths[3]),
            fmt_pct(r.mom_6m).rjust(widths[4]),
            fmt_pct(r.trend_200d).rjust(widths[5]),
            fmt_pct(r.drawup_52w).rjust(widths[6]),
            fmt_pct(r.vol_90d).rjust(widths[7]),
        ]
        return "  " + " ".join(cells)

    border = "  " + " ".join([("-" * w) for w in widths])
    lines = [
        "  " + " ".join([cols[i].ljust(widths[i]) for i in range(len(cols))]),
        border,
    ]
    for i, r in enumerate(rows[:limit]):
        lines.append(row_to_line(i, r))

    tail = [
        "",
        f"Universe: {uni_size}  |  Pool: {pool_size} (3M top vol)  |  As of: {now_utc} UTC",
        "Method : score = 0.4*mom + 0.3*trend + 0.2*drawup + 0.1*low-vol",
        f"Took   : {took_s:.2f}s",
        "",
        "------------------------------------------",
        "> run: top10_volpool_cli.py --index [sp500|dow30|nasdaq100|all] --pool-size 60 --limit 10 --json",
        "==========================================",
    ]
    return "\n".join(head + lines + tail)

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Top-10 from 3M high-volatility pool — ASCII output")
    ap.add_argument("--index", type=str, default="all", help="sp500 | dow30 | nasdaq100 | all")
    ap.add_argument("--pool-size", type=int, default=60, help="Tickers to keep from 3M top volatility")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many to print from ranked pool")
    ap.add_argument("--json", action="store_true", help="Also print JSON payload after table")
    ap.add_argument("--max", type=int, default=None, help="Optional cap on initial universe size")
    ap.add_argument("--chunk", type=int, default=DEFAULT_CHUNK_SIZE, help="yfinance download chunk size")
    ap.add_argument("--cache-dir", type=str, default=".cache_top10_volpool", help="Cache directory for index HTML")
    ap.add_argument("--ttl-hours", type=int, default=DEFAULT_CACHE_TTL_HOURS, help="Cache TTL (hours)")
    ap.add_argument("--no-cache", action="store_true", help="Disable caching completely")
    return ap.parse_args()

def main() -> int:
    t0 = time.time()
    args = parse_args()

    # Basic sanity
    if args.pool_size <= 0:
        print("pool-size must be > 0", file=sys.stderr)
        return 2
    if args.limit <= 0:
        print("limit must be > 0", file=sys.stderr)
        return 2
    if args.chunk <= 0:
        args.chunk = DEFAULT_CHUNK_SIZE

    cache_dir = Path(args.cache_dir)
    ttl_hours = max(1, int(args.ttl_hours))
    session_http = make_http_session()

    try:
        universe = resolve_index_set(args.index, session_http, cache_dir, ttl_hours, args.no_cache)
    except Exception as e:
        print(f"Error resolving index set: {e}", file=sys.stderr)
        return 2

    if args.max is not None and args.max > 0:
        universe = universe[:args.max]

    universe = [u for u in universe if u]  # cleanup
    uni_size = len(universe)
    if uni_size == 0:
        print("No tickers in universe.", file=sys.stderr)
        return 2

    # Cap pool/limit to universe
    pool_target = min(args.pool_size, uni_size)
    limit = min(args.limit, pool_target)

    # Phase 1: 3M volatility pool
    pool = build_vol_pool(universe, pool_target, chunk_size=args.chunk)
    if not pool:
        print("Could not build volatility pool (no usable price series).", file=sys.stderr)
        return 3

    # Phase 2: rank metrics on pool (1y history, faster vs full universe)
    close_map = download_close_map_days(pool, days=RANK_LOOKBACK_DAYS, chunk_size=args.chunk)
    rows: list[Row] = []
    for tkr in pool:
        px = close_map.get(tkr, pd.Series(dtype="float64"))
        rows.append(compute_row_from_series(tkr, px))

    score_rows(rows)
    # sort by score descending; NaN scores last
    rows.sort(key=lambda r: (pd.isna(r.score), r.score), reverse=True)

    took = time.time() - t0
    session_id = datetime.utcnow().strftime("%Y%m%d") + "-VOLPOOL"

    # Print ASCII table
    print(render_table(rows, limit, uni_size, len(pool), took, session_id, args.index))

    # Optional JSON
    if args.json:
        data = {
            "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "indices": args.index,
            "universe_size": uni_size,
            "pool_size": len(pool),
            "limit": limit,
            "pool_method": {"window": VOL_PERIOD, "rank_by": "annualized_vol", "rets_used_days": VOL_RET_DAYS},
            "rank_method": {
                "lookback_days": RANK_LOOKBACK_DAYS,
                "features": ["mom_6m", "trend_200d", "drawup_52w", "vol_90d"],
                "weights": {"mom_6m": 0.4, "trend_200d": 0.3, "drawup_52w": 0.2, "low_vol": 0.1},
            },
            "results": [
                {
                    "rank": i + 1,
                    "ticker": r.ticker,
                    "score": r.score,
                    "last": r.last,
                    "mom_6m": r.mom_6m,
                    "trend_200d": r.trend_200d,
                    "drawup_52w": r.drawup_52w,
                    "vol_90d": r.vol_90d,
                    "ranks": {
                        "mom": r.mom_rank,
                        "trend": r.trend_rank,
                        "drawup": r.drawup_rank,
                        "lowvol": r.lowvol_rank,
                    },
                }
                for i, r in enumerate(rows[:limit])
            ],
        }
        print(json.dumps(data, indent=2))

    return 0

if __name__ == "__main__":
    sys.exit(main())

# ==========================================
# > READY.
#   If you want this wired into your FastAPI ASCII ledger (/ingest each run),
#   say: "hook volpool output into ingestion service" and I’ll add a POST mode.
# ==========================================
