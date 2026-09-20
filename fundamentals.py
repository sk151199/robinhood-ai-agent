"""On-chain fundamentals and flows, fetched by code for the research and risk agents.

The theses the agents actually act on are fundamental: Uniswap's fee burn, Morpho's deposit
growth, Ondo's tokenised-asset float. Until now the research desk could only reach those numbers
through articles, which are stale by the time they are written and often wrong — one cycle cited
a 10-year yield from a headline that was 0.1pp off the live figure. These endpoints are the
primary sources those articles quote, they are free, and they answer in under a second.

Sources, all keyless:
  DefiLlama   protocol TVL and its 1d/7d momentum, protocol category
  DefiLlama   aggregate stablecoin supply — the closest thing to a money-supply reading for crypto
  CoinGecko   BTC dominance, total market cap, and sector rotation by 24h market-cap change
  alternative.me  Fear & Greed index, today against a week ago
  OKX         perpetual funding rates — what leveraged positioning is paying to stay on

Token unlock schedules were deliberately left out: DefiLlama's emissions endpoint now returns
402, and no free replacement was found. The research desk still checks unlocks by web search,
and that gap is stated in its prompt rather than papered over.
"""

import datetime as dt
import json
import os
import urllib.request

from config import settings

CACHE_PATH = os.path.join(settings.log_dir, "fundamentals_cache.json")
# Protocol TVL and sector data move slowly; sentiment and funding are worth refreshing per run.
TTL_MINUTES = {"protocols": 360, "stables": 360, "sectors": 360, "sentiment": 60, "funding": 30}

PROTOCOLS_URL = "https://api.llama.fi/protocols"
STABLES_URL = "https://stablecoins.llama.fi/stablecoins?includePrices=false"
GLOBAL_URL = "https://api.coingecko.com/api/v3/global"
SECTORS_URL = "https://api.coingecko.com/api/v3/coins/categories"
SENTIMENT_URL = "https://api.alternative.me/fng/?limit=8"
FUNDING_URL = "https://www.okx.com/api/v5/public/funding-rate?instId={symbol}-USD-SWAP"

# A sector needs real size before its 24h move means anything.
MIN_SECTOR_MCAP = 3e8
# Symbols are matched to protocols by ticker, which occasionally lands on a trivial namesake
# (SOL matched a $0M farm). Below this, the row is noise rather than a fundamental.
MIN_PROTOCOL_TVL = 1e7


def _fetch(url: str):
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read())


def _cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            cached = json.load(handle)
        return cached if isinstance(cached, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _save(cache: dict):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{CACHE_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(cache, handle)
        os.replace(tmp_path, CACHE_PATH)
    except OSError:
        pass


def _cached(key: str, build):
    """Return a cached section, rebuilding it when stale. A failed rebuild keeps the old value:
    stale fundamentals are worth more than none, and the section says how old they are."""
    cache = _cache()
    entry = cache.get(key) or {}
    fetched_at = entry.get("fetched_at")
    if fetched_at:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(fetched_at)
        if age < dt.timedelta(minutes=TTL_MINUTES.get(key, 60)):
            return entry.get("data"), int(age.total_seconds() // 60)
    try:
        data = build()
    except Exception:
        return entry.get("data"), None
    cache[key] = {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "data": data}
    _save(cache)
    return data, 0


def _build_protocols() -> dict:
    """symbol -> the largest protocol issuing that token, with TVL and its momentum."""
    best = {}
    for row in _fetch(PROTOCOLS_URL):
        symbol = str(row.get("symbol") or "").upper().lstrip("$")
        tvl = row.get("tvl")
        if not symbol or symbol == "-" or not tvl or tvl <= 0:
            continue
        if symbol not in best or tvl > best[symbol]["tvl"]:
            best[symbol] = {"name": row.get("name"), "category": row.get("category"), "tvl": float(tvl),
                            "change_1d": row.get("change_1d"), "change_7d": row.get("change_7d")}
    return best


def _build_stables() -> dict:
    payload = _fetch(STABLES_URL)
    now = prev = 0.0
    for asset in payload.get("peggedAssets") or []:
        now += float((asset.get("circulating") or {}).get("peggedUSD") or 0)
        prev += float((asset.get("circulatingPrevWeek") or {}).get("peggedUSD") or 0)
    return {"total": now, "change_7d": (now - prev) / prev if prev else None}


def _build_sectors() -> dict:
    rows = [row for row in _fetch(SECTORS_URL)
            if row.get("market_cap_change_24h") is not None and (row.get("market_cap") or 0) >= MIN_SECTOR_MCAP]
    rows.sort(key=lambda row: row["market_cap_change_24h"], reverse=True)
    trim = lambda row: {"name": row["name"], "change_24h": row["market_cap_change_24h"] / 100}
    world = _fetch(GLOBAL_URL).get("data") or {}
    return {
        "leaders": [trim(row) for row in rows[:4]],
        "laggards": [trim(row) for row in rows[-4:]],
        "btc_dominance": (world.get("market_cap_percentage") or {}).get("btc"),
        "total_mcap_change_24h": world.get("market_cap_change_percentage_24h_usd"),
    }


def _build_sentiment() -> dict:
    rows = (_fetch(SENTIMENT_URL) or {}).get("data") or []
    if not rows:
        return {}
    week_ago = rows[7] if len(rows) > 7 else rows[-1]
    return {"value": int(rows[0]["value"]), "label": rows[0].get("value_classification"),
            "week_ago": int(week_ago["value"])}


def _build_funding(symbols: tuple) -> dict:
    """Perp funding per symbol. Positive means longs are paying shorts — crowded upside."""
    rates = {}
    for symbol in symbols:
        try:
            payload = _fetch(FUNDING_URL.format(symbol=symbol))
            rows = payload.get("data") or []
            if rows:
                rates[symbol] = float(rows[0]["fundingRate"])
        except Exception:
            continue
    return rates


def snapshot(symbols=()) -> dict:
    """Everything, cached by section. `symbols` are the pairs worth a funding lookup."""
    protocols, protocols_age = _cached("protocols", _build_protocols)
    stables, _ = _cached("stables", _build_stables)
    sectors, _ = _cached("sectors", _build_sectors)
    sentiment, _ = _cached("sentiment", _build_sentiment)
    wanted = tuple(dict.fromkeys(("BTC", "ETH") + tuple(symbols)))[:8]
    funding, _ = _cached("funding", lambda: _build_funding(wanted))
    return {"protocols": protocols or {}, "stables": stables or {}, "sectors": sectors or {},
            "sentiment": sentiment or {}, "funding": funding or {}, "protocols_age_minutes": protocols_age}


def as_prompt_section(universe=(), held=()) -> str:
    """Fundamentals for the tradable universe, plus the flow backdrop."""
    data = snapshot(held)
    protocols, sectors, stables, sentiment, funding = (
        data["protocols"], data["sectors"], data["stables"], data["sentiment"], data["funding"])
    if not protocols and not sectors:
        return "Fundamentals feed unavailable this run — research these yourself."

    lines = ["Fetched by code from DefiLlama, CoinGecko, alternative.me and OKX. These are the primary sources "
             "that articles quote, so prefer them over a figure in a headline."]

    if sentiment:
        move = sentiment["value"] - sentiment.get("week_ago", sentiment["value"])
        lines.append(f"Fear & Greed: {sentiment['value']} ({sentiment.get('label')}), {move:+d} vs a week ago. "
                     "Extremes mark crowded positioning, not direction.")
    if sectors.get("btc_dominance") is not None:
        lines.append(f"BTC dominance {sectors['btc_dominance']:.1f}%; total crypto market cap "
                     f"{(sectors.get('total_mcap_change_24h') or 0):+.1f}% in 24h.")
    if stables.get("total"):
        change = stables.get("change_7d")
        lines.append(f"Stablecoin supply ${stables['total'] / 1e9:.1f}B, {change:+.2%} over 7 days. "
                     "Rising supply is dry powder entering the system; falling supply is capital leaving."
                     if change is not None else f"Stablecoin supply ${stables['total'] / 1e9:.1f}B.")
    if sectors.get("leaders"):
        lines.append("Sector rotation (24h market cap): up — "
                     + ", ".join(f"{row['name']} {row['change_24h']:+.1%}" for row in sectors["leaders"])
                     + "; down — "
                     + ", ".join(f"{row['name']} {row['change_24h']:+.1%}" for row in reversed(sectors["laggards"])))
    if funding:
        lines.append("Perp funding (8h): "
                     + ", ".join(f"{symbol} {rate:+.3%}" for symbol, rate in funding.items())
                     + ". Positive means longs pay shorts; a high positive rate is crowded, and unwinds hurt.")

    rows = []
    for symbol in dict.fromkeys(universe):
        row = protocols.get(str(symbol).upper())
        if row and row.get("tvl", 0) >= MIN_PROTOCOL_TVL:
            rows.append((symbol.upper(), row))
    if rows:
        rows.sort(key=lambda item: item[1].get("change_7d") or 0, reverse=True)
        age = data.get("protocols_age_minutes")
        lines.append("")
        lines.append(f"Protocol fundamentals for pairs you can trade ({'fresh' if not age else f'{age} min old'}), "
                     "sorted by 7-day deposit growth. TVL is capital actually deposited in the protocol: growth "
                     "that the token price has not followed is the divergence worth investigating, and TVL falling "
                     "while the price rises is the opposite.")
        lines.append(f"{'pair':<8} {'protocol':<20} {'sector':<16} {'TVL':>12} {'1d':>7} {'7d':>7}")
        for symbol, row in rows[:24]:
            tvl = row["tvl"]
            size = f"${tvl / 1e9:.2f}B" if tvl >= 1e9 else f"${tvl / 1e6:.0f}M"
            change_1d = f"{row['change_1d']:+.1f}%" if row.get("change_1d") is not None else "   n/a"
            change_7d = f"{row['change_7d']:+.1f}%" if row.get("change_7d") is not None else "   n/a"
            lines.append(f"{symbol:<8} {str(row['name'])[:20]:<20} {str(row['category'])[:16]:<16} "
                         f"{size:>12} {change_1d:>7} {change_7d:>7}")
        lines.append("Pairs absent from this table are not DeFi protocols (layer-1 tokens, memes, exchange "
                     "tokens); judge those on their own terms, not on missing TVL.")
    return "\n".join(lines)
