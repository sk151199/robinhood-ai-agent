"""Congressional STOCK Act disclosures, summarized into the cycle prompt.

Fetched here rather than by the agent on purpose. The agent has no web tools, so nothing it
reads from the internet can steer a tool call; this module pulls a known endpoint, keeps a
fixed set of fields, and hands over a bounded digest.

The data is public but slow: members may file up to ~45 days after the trade, so treat it as
a positioning signal, not a timing edge.
"""

import datetime as dt
import json
import os
import urllib.error
import urllib.request
from collections import defaultdict

from config import settings
from trade_logger import logger

CAVEAT = ("Disclosed up to ~45 days after the trade, so this is slow positioning data, not timing. "
          "Corroborate with price, liquidity and news; never buy something only because a politician did.")

# One member filing a dozen tickers at once is bookkeeping, not conviction. Cap the digest so
# the names several different members have been accumulating stay at the top and visible.
MAX_DIGEST_ROWS = 15


CACHE_MAX_AGE_HOURS = 12


def _cache_path() -> str:
    return os.path.join(settings.log_dir, "congress_cache.json")


def _read_cache():
    """Last good digest, if recent. The feed rate-limits, and filings that lag 45 days do not
    change hour to hour — a cached digest beats no data at all."""
    try:
        with open(_cache_path(), encoding="utf-8") as handle:
            cached = json.load(handle)
        stamped = dt.datetime.fromisoformat(cached["timestamp"])
        if (dt.datetime.now(dt.timezone.utc) - stamped).total_seconds() <= CACHE_MAX_AGE_HOURS * 3600:
            return cached["section"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return None


def _write_cache(section: str):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as handle:
            json.dump({"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "section": section}, handle)
    except OSError:
        pass


def _fetch(url: str) -> dict:
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "robinhood-ai-agent"})
    with urllib.request.urlopen(request, timeout=settings.congress_timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def _as_date(value):
    try:
        return dt.date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def recent_trades(today: dt.date = None) -> list:
    """Disclosures filed within the lookback window, newest first."""
    today = today or dt.date.today()
    cutoff = today - dt.timedelta(days=settings.congress_lookback_days)

    payload = _fetch(settings.congress_api_url)
    rows = payload.get("trades") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("no trades array in response")

    trades = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = (row.get("ticker") or "").strip().upper()
        disclosed = _as_date(row.get("disclosure_date"))
        if not ticker or disclosed is None or disclosed < cutoff or disclosed > today:
            continue
        trades.append({
            "ticker": ticker,
            "member": row.get("member") or "unknown",
            "chamber": row.get("chamber") or "",
            "side": "buy" if "purchase" in str(row.get("type", "")).lower() else "sell",
            "amount_range": row.get("amount_range") or "",
            "transaction_date": str(row.get("transaction_date") or "")[:10],
            "disclosure_date": disclosed.isoformat(),
        })

    trades.sort(key=lambda trade: trade["disclosure_date"], reverse=True)
    return trades[: settings.congress_max_rows]


def summarize(trades: list) -> list:
    """One row per ticker: how many bought, how many sold, and who."""
    grouped = defaultdict(lambda: {"buys": 0, "sells": 0, "members": [], "latest": "", "amount": ""})
    for trade in trades:
        entry = grouped[trade["ticker"]]
        entry["buys" if trade["side"] == "buy" else "sells"] += 1
        if trade["member"] not in entry["members"]:
            entry["members"].append(trade["member"])
        if trade["disclosure_date"] > entry["latest"]:
            entry["latest"] = trade["disclosure_date"]
            entry["amount"] = trade["amount_range"]

    rows = [{"ticker": ticker, **data} for ticker, data in grouped.items()]
    # Net buying first, then how many distinct members: several people buying the same name is
    # a stronger signal than one person filing a basket.
    rows.sort(
        key=lambda row: (row["buys"] - row["sells"], len(row["members"]), row["buys"], row["latest"]),
        reverse=True,
    )
    return rows


def as_prompt_section() -> str:
    if not settings.congress_enabled:
        return "Congressional disclosure data is disabled this cycle."

    try:
        trades = recent_trades()
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError) as error:
        logger.warning("Congressional disclosures unavailable: %s", error)
        cached = _read_cache()
        if cached:
            return f"{cached}\n\n(Cached from an earlier fetch; the live feed was unavailable this cycle.)"
        return "Congressional disclosure data unavailable this cycle (fetch failed)."

    if not trades:
        return f"No congressional disclosures filed in the last {settings.congress_lookback_days} days."

    rows = summarize(trades)
    lines = [CAVEAT, ""]
    for row in rows[:MAX_DIGEST_ROWS]:
        members = ", ".join(row["members"][:3])
        extra = f" +{len(row['members']) - 3} more" if len(row["members"]) > 3 else ""
        lines.append(
            f"{row['ticker']}: {row['buys']} buy / {row['sells']} sell — {members}{extra}"
            f" (latest filed {row['latest']}, {row['amount']})"
        )
    if len(rows) > MAX_DIGEST_ROWS:
        lines.append(f"...and {len(rows) - MAX_DIGEST_ROWS} more tickers with fewer filings.")
    section = "\n".join(lines)
    _write_cache(section)
    return section
