"""Macro dashboard: live market numbers fetched by code, handed to the research desk.

Plain code, not a model. The desk was spending many of its turns web-searching for figures a
single request returns, and it still got them wrong: a cycle cited "the 10-year broke 5.04%"
from an article while the live yield was 4.945%. Fetching the numbers directly saves those turns
(and Claude usage) and gives both agents one authoritative figure to reason from.

Source: Yahoo Finance's public chart endpoint, no key. Cached for 15 minutes.
"""

import datetime as dt
import json
import os
import urllib.parse
import urllib.request

from config import settings

CACHE_PATH = os.path.join(settings.log_dir, "macro_cache.json")
CACHE_MINUTES = 15
CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?range=1mo&interval=1d"

# (ticker, label, unit). ^TNX is quoted directly in percent.
SERIES = (
    ("^TNX", "US 10-year yield", "%"),
    ("DX-Y.NYB", "US dollar index", ""),
    ("CL=F", "WTI crude oil", "$"),
    ("GC=F", "Gold", "$"),
    ("^VIX", "VIX (equity fear)", ""),
    ("^GSPC", "S&P 500", ""),
    ("^IXIC", "Nasdaq", ""),
    ("IBIT", "IBIT spot BTC ETF", "$"),
    ("ETHA", "ETHA spot ETH ETF", "$"),
)


def _fetch(url: str) -> bytes:
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=8) as response:
        return response.read()


def _series(symbol: str) -> dict:
    data = json.loads(_fetch(CHART_URL.format(symbol=urllib.parse.quote(symbol))))["chart"]["result"][0]
    quote = data["indicators"]["quote"][0]
    closes = [value for value in quote.get("close") or [] if value is not None]
    volumes = [value for value in quote.get("volume") or [] if value]
    last = data["meta"].get("regularMarketPrice") or (closes[-1] if closes else None)
    if not last or len(closes) < 6:
        return {}
    row = {
        "last": last,
        "change_1d": (last - closes[-2]) / closes[-2] if closes[-2] else None,
        "change_5d": (last - closes[-6]) / closes[-6] if closes[-6] else None,
        "change_1m": (last - closes[0]) / closes[0] if closes[0] else None,
    }
    # ETF volume against its month average: the only keyless proxy for heavy creation or redemption.
    # The last complete day, not today: today's partial volume read as 0.0x before the open.
    if len(volumes) >= 10:
        earlier = volumes[:-2]
        row["volume_ratio"] = volumes[-2] / (sum(earlier) / len(earlier))
    return row


def snapshot() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            cached = json.load(handle)
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(cached["fetched_at"])
        if age < dt.timedelta(minutes=CACHE_MINUTES):
            return cached
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    rows = {}
    for symbol, _label, _unit in SERIES:
        try:
            row = _series(symbol)
        except Exception:
            continue
        if row:
            rows[symbol] = row
    result = {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "rows": rows}
    if rows:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            with open(CACHE_PATH, "w", encoding="utf-8") as handle:
                json.dump(result, handle)
        except OSError:
            pass
    return result


def as_prompt_section() -> str:
    data = snapshot()
    rows = data.get("rows") or {}
    if not rows:
        return "Macro dashboard unavailable this run — research macro yourself."

    def pct(value):
        return f"{value:+.1%}" if value is not None else "n/a"

    lines = ["Live figures fetched by code from Yahoo Finance minutes ago. Treat these as authoritative over any number "
             "in an article; web-search only for what is not here (ETF net flows in dollars, Fed commentary).",
             f"{'':<22}{'last':>10}{'1d':>8}{'5d':>8}{'1m':>8}"]
    for symbol, label, unit in SERIES:
        row = rows.get(symbol)
        if not row:
            continue
        last = f"{row['last']:.3f}{unit}" if unit == "%" else f"{unit}{row['last']:,.2f}"
        extra = (f"   last full day's volume {row['volume_ratio']:.1f}x its 1m average"
                 if row.get("volume_ratio") and symbol in ("IBIT", "ETHA") else "")
        lines.append(f"{label:<22}{last:>10}{pct(row['change_1d']):>8}{pct(row['change_5d']):>8}"
                     f"{pct(row['change_1m']):>8}{extra}")
    return "\n".join(lines)
