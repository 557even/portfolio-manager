#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ==========================================
#        PROGRAMMING ASCII TERMINAL
# ==========================================
#
# STATUS:
#   Health        : OPERATIONAL
#   Tasks         : ACTIVE - CLI RANKER (VOLATILITY POOL)
#   Objective     : TOP-10 FROM TOP-VOL 3M POOL (S&P 500, DOW 30, NASDAQ-100)
#   Results       : READY
#
# ------------------------------------------
# SYSTEM VARIABLES:
#   User          : ACTIVE
#   Session ID    : AUTO
#   Uptime        : 00:00:00
#   Mode          : batch
#   Prompt Mode   : single-file
#   Learning Rate : static
#   Loop Count    : 1
#
# ------------------------------------------
# TASK          : Build a 3-month high-volatility pool, then rank that pool
# OBJECTIVE     : Faster runs by limiting universe to top-σ names, then full metrics
# RESULTS       : ASCII table + optional JSON
# ==========================================

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

import requests
import pandas as pd
import yfinance as yf

# ---------- Tunables ----------
RANK_LOOKBACK_DAYS = 365        # fetch depth for ranking metrics (pool only)
VOL_PERIOD = "3mo"              # 3-month window to build the pool
VOL_RET_DAYS = 63               # use ~63 trading days for vol calc
DEFAULT_LIMIT = 10
USER_AGENT = "Mozilla/5.0 (Top10-VolPool-CLI)"

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

# ---------- HTTP + Index scraping ----------
def http_get(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
    r.raise_for_status()
    return r.text

def normalize_symbol(sym: str) -> str:
    return sym.strip().upper().replace(".", "-").replace(" ", "")

def unique_keep_order(items: t.Iterable[str]) -> list[str]:
    seen, out = set(), []
    for x in items:
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def get_sp500_constituents() -> list[str]:
    html = http_get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    dfs = pd.read_html(io.StringIO(html))
    for df in dfs:
        cols = [str(c).strip().lower() for c in df.columns]
        if any("symbol" in c for c in cols):
            sym_col = df.columns[[("symbol" in c) for c in cols]][0]
            return unique_keep_order([normalize_symbol(s) for s in df[sym_col].astype(str)])
    raise RuntimeError("S&P 500 table not found")

def get_dow30_constituents() -> list[str]:
    html = http_get("https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average")
    dfs = pd.read_html(io.StringIO(html))
    for df in dfs:
        cols = [str(c).strip().lower() for c in df.columns]
        if any(("symbol" in c) or ("ticker" in c) for c in cols):
            if "Symbol" in df.columns:
                sym_col = "Symbol"
            elif "Ticker symbol" in df.columns:
                sym_col = "Ticker symbol"
            else:
                sym_col = df.columns[[("symbol" in c) or ("ticker" in c) for c in cols]][0]
            syms = [normalize_symbol(s) for s in df[sym_col].astype(str)]
            syms = [s for s in syms if s and s != "N/A"]
            return unique_keep_order(syms)
    raise RuntimeError("Dow 30 table not found")

def get_nasdaq100_constituents() -> list[str]:
    html = http_get("https://en.wikipedia.org/wiki/NASDAQ-100")
    dfs = pd.read_html(io.StringIO(html))
    for df in dfs:
        cols = [str(c).strip().lower() for c in df.columns]
        if any("ticker" in c for c in cols):
            tik_col = df.columns[[("ticker" in c) for c in cols]][0]
            syms = [normalize_symbol(s) for s in df[tik_col].astype(str)]
            syms = [s for s in syms if s and s != "N/A"]
            return unique_keep_order(syms)
    raise RuntimeError("NASDAQ-100 table not found")

def resolve_index_set(kind: str) -> list[str]:
    k = kind.lower()
    if k == "sp500": return get_sp500_constituents()
    if k in ("dow", "dow30"): return get_dow30_constituents()
    if k in ("nasdaq", "nasdaq100"): return get_nasdaq100_constituents()
    if k in ("all", "union"):
        return unique_keep_order([*get_sp500_constituents(), *get_dow30_constituents(), *get_nasdaq100_constituents()])
    raise ValueError("index must be one of: sp500 | dow30 | nasdaq100 | all")

# ---------- yfinance helpers ----------
def _extract_close_df(data: pd.DataFrame, ticker: str) -> pd.Series:
    """Robustly pull Close series for ticker from yfinance download result."""
    try:
        if isinstance(data.columns, pd.MultiIndex):
            # Try [ticker]['Close'] pattern
            if ticker in data.columns.get_level_values(0):
                s = data[ticker]
                if "Close" in s.columns:
                    return s["Close"].dropna()
            # Try ['Close'][ticker]
            if "Close" in data.columns.get_level_values(0):
                return data["Close"][ticker].dropna()
            # Try tuple selection
            return data[(ticker, "Close")].dropna()
        else:
            # Single-ticker fallback
            if "Close" in data.columns:
                return data["Close"].dropna()
    except Exception:
        pass
    return pd.Series(dtype="float64")

def download_close_map(tickers: list[str], period: str = "3mo") -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if not tickers: return out
    try:
        df = yf.download(
            tickers=tickers,
            period=period,
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        for t in tickers:
            s = _extract_close_df(df, t)
            if not s.empty:
                out[t] = s
    except Exception:
        pass
    # Fallback per-ticker for any missing
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            h = yf.Ticker(t).history(period=period, interval="1d", auto_adjust=True, actions=False)
            if not h.empty:
                out[t] = h["Close"].dropna()
        except Exception:
            continue
    return out

def download_close_map_days(tickers: list[str], days: int) -> dict[str, pd.Series]:
    out: dict[str, pd.Series] = {}
    if not tickers: return out
    try:
        df = yf.download(
            tickers=tickers,
            period=f"{days}d",
            interval="1d",
            auto_adjust=True,
            group_by="ticker",
            threads=True,
            progress=False,
        )
        for t in tickers:
            s = _extract_close_df(df, t)
            if not s.empty:
                out[t] = s
    except Exception:
        pass
    missing = [t for t in tickers if t not in out]
    for t in missing:
        try:
            h = yf.Ticker(t).history(period=f"{days}d", interval="1d", auto_adjust=True, actions=False)
            if not h.empty:
                out[t] = h["Close"].dropna()
        except Exception:
            continue
    return out

# ---------- Metrics ----------
def pct_change(a: float, b: float) -> float:
    if b == 0 or pd.isna(a) or pd.isna(b): return float("nan")
    return (a / b) - 1.0

def annualized_vol(returns: pd.Series) -> float:
    r = returns.dropna()
    if r.empty: return float("nan")
    return float(r.std() * math.sqrt(252))

def compute_row_from_series(t: str, px: pd.Series) -> Row:
    if px.empty or len(px) < 60:
        return Row(ticker=t, last=float("nan"), mom_6m=float("nan"),
                   trend_200d=float("nan"), drawup_52w=float("nan"), vol_90d=float("nan"))
    last = float(px.iloc[-1])
    mom_6m = pct_change(last, float(px.iloc[-126])) if len(px) > 126 else pct_change(last, float(px.iloc[0]))
    ma200 = float(px.rolling(200).mean().iloc[-1]) if len(px) >= 200 else float(px.rolling(len(px)).mean().iloc[-1])
    trend_200d = pct_change(last, ma200)
    high_52w = float(px.rolling(min(252, len(px))).max().iloc[-1])
    drawup_52w = -abs(pct_change(last, high_52w))  # closer to 0 is better ⇒ use negative abs
    vol_90d = annualized_vol(px.pct_change().tail(90))
    return Row(ticker=t, last=last, mom_6m=mom_6m, trend_200d=trend_200d, drawup_52w=drawup_52w, vol_90d=vol_90d)

def percentile_rank(vals: t.List[float]) -> t.List[float]:
    clean = [v for v in vals if pd.notna(v)]
    if not clean: return [0.5] * len(vals)
    s = sorted(clean)
    out: t.List[float] = []
    for v in vals:
        if pd.isna(v): out.append(0.5); continue
        idx = 0
        for x in s:
            if x <= v: idx += 1
        out.append(idx / max(1, len(s)))
    return out

def score_rows(rows: t.List[Row]) -> None:
    moms = [r.mom_6m for r in rows]
    trds = [r.trend_200d for r in rows]
    drws = [r.drawup_52w for r in rows]
    vols = [r.vol_90d for r in rows]
    mom_r = percentile_rank(moms)
    trd_r = percentile_rank(trds)
    drw_r = percentile_rank(drws)
    vol_r = [1.0 - v for v in percentile_rank(vols)]  # lower vol better
    for i, r in enumerate(rows):
        r.mom_rank = round(float(mom_r[i]), 3)
        r.trend_rank = round(float(trd_r[i]), 3)
        r.drawup_rank = round(float(drw_r[i]), 3)
        r.lowvol_rank = round(float(vol_r[i]), 3)
        r.score = round(0.4 * mom_r[i] + 0.3 * trd_r[i] + 0.2 * drw_r[i] + 0.1 * vol_r[i], 4)

# ---------- Pool selection (3M top-vol) ----------
def build_vol_pool(tickers: list[str], pool_size: int) -> list[str]:
    series_map = download_close_map(tickers, period=VOL_PERIOD)
    vols: list[tuple[str, float]] = []
    for t, s in series_map.items():
        rets = s.pct_change().tail(VOL_RET_DAYS)
        v = annualized_vol(rets)
        if not pd.isna(v):
            vols.append((t, v))
    vols.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in vols[:pool_size]]

# ---------- Rendering ----------
def fmt_pct(x: float) -> str:
    return "NA" if pd.isna(x) else f"{x*100:+.1f}%"

def fmt_num(x: float, n: int = 2) -> str:
    return "NA" if pd.isna(x) else f"{x:.{n}f}"

def render_table(rows: t.List[Row], limit: int, uni_size: int, pool_size: int, took_s: float, session: str, indices_used: str) -> str:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
    head = [
        "==========================================",
        "        PROGRAMMING ASCII TERMINAL",
        "==========================================",
        "",
        "STATUS:",
        "  Health        : OPERATIONAL",
        "  Tasks         : ACTIVE - RANK VOL-POOL",
        "  Objective     : TOP-10 STOCKS (ASCII OUTPUT)",
        "  Results       : GENERATED",
        "",
        "------------------------------------------",
        "SYSTEM VARIABLES:",
        "  User          : ACTIVE",
        f"  Session ID    : {session}",
        f"  Uptime        : 00:00:{int(took_s):02d}",
        "  Mode          : batch",
        "  Prompt Mode   : single-file",
        "  Learning Rate : static",
        f"  Loop Count    : 1",
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
            str(i+1).rjust(widths[0]),
            r.ticker.ljust(widths[1]),
            f"{r.score:.3f}".rjust(widths[2]) if not pd.isna(r.score) else "NA".rjust(widths[2]),
            fmt_num(r.last).rjust(widths[3]),
            fmt_pct(r.mom_6m).rjust(widths[4]),
            fmt_pct(r.trend_200d).rjust(widths[5]),
            fmt_pct(r.drawup_52w).rjust(widths[6]),
            fmt_pct(r.vol_90d).rjust(widths[7]),
        ]
        return "  " + " ".join(cells)
    border = "  " + " ".join([("-"*w) for w in widths])
    lines = []
    lines.append("  " + " ".join([cols[i].ljust(widths[i]) for i in range(len(cols))]))
    lines.append(border)
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
    ap.add_argument("--pool-size", type=int, default=60, help="Number of tickers to keep from 3M top volatility")
    ap.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="How many to print from the ranked pool")
    ap.add_argument("--json", action="store_true", help="Also print JSON payload after table")
    ap.add_argument("--max", type=int, default=None, help="Optional cap on initial universe size (for very slow links)")
    return ap.parse_args()

def main() -> int:
    t0 = time.time()
    args = parse_args()

    try:
        universe = resolve_index_set(args.index)
    except Exception as e:
        print(f"Error resolving index set: {e}", file=sys.stderr)
        return 2

    if args.max is not None and args.max > 0:
        universe = universe[:args.max]

    uni_size = len(universe)
    if uni_size == 0:
        print("No tickers in universe.", file=sys.stderr)
        return 2

    # Phase 1: 3M volatility pool
    pool = build_vol_pool(universe, args.pool_size)
    if not pool:
        print("Could not build volatility pool.", file=sys.stderr)
        return 3

    # Phase 2: rank metrics on pool (1y history, faster vs full universe)
    close_map = download_close_map_days(pool, days=RANK_LOOKBACK_DAYS)
    rows: list[Row] = []
    for t in pool:
        rows.append(compute_row_from_series(t, close_map.get(t, pd.Series(dtype="float64"))))

    score_rows(rows)
    rows.sort(key=lambda r: (pd.isna(r.score), r.score), reverse=True)

    took = time.time() - t0
    session = datetime.utcnow().strftime("%Y%m%d") + "-VOLPOOL"

    # Print ASCII table
    print(render_table(rows, args.limit, uni_size, len(pool), took, session, args.index))

    # Optional JSON
    if args.json:
        data = {
            "as_of_utc": datetime.now(timezone.utc).isoformat(timespec="minutes"),
            "indices": args.index,
            "universe_size": uni_size,
            "pool_size": len(pool),
            "limit": args.limit,
            "pool_method": {"window":"3mo","rank_by":"annualized_vol","rets_used_days":VOL_RET_DAYS},
            "rank_method": {"lookback_days":RANK_LOOKBACK_DAYS,
                            "features":["mom_6m","trend_200d","drawup_52w","vol_90d"],
                            "weights":{"mom_6m":0.4,"trend_200d":0.3,"drawup_52w":0.2,"low_vol":0.1}},
            "results":[
                {
                    "rank": i+1,
                    "ticker": r.ticker,
                    "score": r.score,
                    "last": r.last,
                    "mom_6m": r.mom_6m,
                    "trend_200d": r.trend_200d,
                    "drawup_52w": r.drawup_52w,
                    "vol_90d": r.vol_90d
                } for i, r in enumerate(rows[:args.limit])
            ]
        }
        print(json.dumps(data, indent=2))
    return 0

if __name__ == "__main__":
    sys.exit(main())
