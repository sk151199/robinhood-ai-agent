"""Market-wide data for every tradable pair, so no coin is invisible to the research desk.

Two gaps this closes. Coinbase does not list everything Robinhood trades, so eight pairs had no
price history at all and never appeared in the daily table, the RSI list or the mover feed — they
could not be considered because they could not be seen. And nothing carried market cap, dilution
or turnover, so "is this a $50M token with 80% of supply still to unlock" was a question only a
web search could answer.

CoinGecko covers all of it. The hazard is that tickers are not unique there: LIT resolves to four
different coins, and picking the wrong one is precisely the mistake that once presented a
delisted price series as a 40% crash. So every mapping is verified against a live price the
system already trusts — Coinbase's, or Robinhood's own mark — and a candidate that disagrees by
more than a few percent is rejected. A pair that cannot be verified is reported as unmapped
rather than guessed at.
"""

import datetime as dt
import json
import time
import os
import urllib.error
import urllib.parse
import urllib.request

import market_context
import marks_store
import mover_monitor
from config import settings

MAP_PATH = os.path.join(settings.log_dir, "coingecko_map.json")
CACHE_PATH = os.path.join(settings.log_dir, "coin_universe_cache.json")
LIST_URL = "https://api.coingecko.com/api/v3/coins/list"
MARKETS_URL = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={ids}"
               "&price_change_percentage=24h,7d,30d&per_page=250&page=1")
# Ranked pages, used to resolve tickers: one call per 250 coins instead of one per symbol, which
# rate-limited immediately and mapped 3 of 90.
PAGE_URL = ("https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc"
            "&per_page=250&page={page}")
PAGES = 8
CHART_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=daily"

CACHE_MINUTES = 30
MAP_TTL_DAYS = 7
# A candidate whose price differs from the anchor by more than this is a different asset.
PRICE_TOLERANCE = 0.08
# Two coins sharing a ticker can both sit inside the tolerance; accept only a clear winner.
AMBIGUITY_MARGIN = 0.5


def _fetch(url: str, attempts: int = 3):
    """Seam for tests. CoinGecko's free tier answers 429 readily and names its own wait, so a
    burst of calls must back off rather than give up — the first build mapped 0 of 90 because a
    single 429 on page one emptied the candidate pool."""
    last = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=25) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as error:
            last = error
            if error.code != 429:
                raise
            wait = float(error.headers.get("retry-after") or 0) or 5 * (attempt + 1)
            time.sleep(min(wait + 1, 30))
    raise last


def _anchor(symbol: str):
    """A price this system already trusts, freshest first.

    Order matters: yesterday's close rejected five correct matches (ONDO had moved 6% today), so
    the live scan is consulted before the daily close.
    """
    symbol = str(symbol).upper()
    live = {row["symbol"]: row.get("price") for row in (mover_monitor.load_latest(False) or {}).get("rows", [])}
    if live.get(symbol):
        return live[symbol]
    mark = marks_store.price(symbol)
    if mark:
        return mark
    series = market_context.series(symbol)
    return series[max(series)][0] if series else None


def _read(path: str):
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def _write(path: str, payload: dict):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, path)
    except OSError:
        pass


def build_map(symbols) -> dict:
    """symbol -> {id, gap} for every pair whose ticker can be matched to a coin safely.

    Candidates come from the ranked market pages, so the whole universe resolves in a handful of
    calls. Each match must agree with a price this system already trusts; where several coins
    share a ticker, the closest agreement inside the tolerance wins and the rest are ignored.
    """
    pool = {}
    for page in range(1, PAGES + 1):
        try:
            rows = _fetch(PAGE_URL.format(page=page))
        except Exception:
            break
        if not rows:
            break
        for row in rows:
            symbol = str(row.get("symbol", "")).upper()
            if symbol and row.get("current_price"):
                pool.setdefault(symbol, []).append((row["id"], float(row["current_price"]),
                                                    row.get("market_cap") or 0))
        time.sleep(3)   # Stay well inside the free tier's ceiling; the map is built weekly.

    mapping, unresolved = {}, []
    for symbol in dict.fromkeys(str(s).upper() for s in symbols):
        candidates = pool.get(symbol) or []
        if not candidates:
            unresolved.append((symbol, "not in the top ranked coins"))
            continue
        anchor = _anchor(symbol)
        if not anchor:
            unresolved.append((symbol, "no trusted price to verify against"))
            continue
        scored = sorted(((coin_id, abs(price - anchor) / anchor) for coin_id, price, _mcap in candidates),
                        key=lambda item: item[1])
        near = [item for item in scored if item[1] <= PRICE_TOLERANCE]
        if not near:
            unresolved.append((symbol, f"no candidate priced near {anchor:g}"))
        elif len(near) > 1 and near[0][1] > near[1][1] * AMBIGUITY_MARGIN:
            unresolved.append((symbol, "two coins share this ticker at a similar price"))
        else:
            mapping[symbol] = {"id": near[0][0], "gap": round(near[0][1], 4)}
    return {"built_at": dt.datetime.now(dt.timezone.utc).isoformat(), "map": mapping,
            "unresolved": [{"symbol": s, "why": why} for s, why in unresolved]}


def coin_map(symbols) -> dict:
    stored = _read(MAP_PATH)
    if stored:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stored["built_at"])
        known = set(stored.get("map", {})) | {row["symbol"] for row in stored.get("unresolved", [])}
        if age < dt.timedelta(days=MAP_TTL_DAYS) and set(str(s).upper() for s in symbols) <= known:
            return stored
    try:
        built = build_map(symbols)
    except Exception:
        return stored or {"map": {}, "unresolved": []}
    _write(MAP_PATH, built)
    return built


def markets(symbols) -> tuple:
    """symbol -> market data for every mapped pair, plus the list that could not be mapped."""
    stored = _read(CACHE_PATH)
    if stored:
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(stored["fetched_at"])
        if age < dt.timedelta(minutes=CACHE_MINUTES):
            return stored["data"], stored.get("unresolved", []), int(age.total_seconds() // 60)

    mapping = coin_map(symbols)
    ids = {entry["id"]: symbol for symbol, entry in (mapping.get("map") or {}).items()}
    data = {}
    try:
        for start in range(0, len(ids), 200):
            batch = list(ids)[start:start + 200]
            for row in _fetch(MARKETS_URL.format(ids=urllib.parse.quote(",".join(batch)))):
                symbol = ids.get(row["id"])
                if not symbol:
                    continue
                market_cap = row.get("market_cap") or 0
                data[symbol] = {
                    "price": row.get("current_price"),
                    "market_cap": market_cap,
                    "fdv": row.get("fully_diluted_valuation"),
                    "volume_24h": row.get("total_volume") or 0,
                    "change_24h": row.get("price_change_percentage_24h_in_currency"),
                    "change_7d": row.get("price_change_percentage_7d_in_currency"),
                    "change_30d": row.get("price_change_percentage_30d_in_currency"),
                    "from_ath": row.get("ath_change_percentage"),
                    "circulating": row.get("circulating_supply"),
                    "total_supply": row.get("total_supply"),
                }
    except Exception:
        if stored:
            return stored["data"], stored.get("unresolved", []), None
        return {}, mapping.get("unresolved", []), None
    unresolved = mapping.get("unresolved", [])
    _write(CACHE_PATH, {"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                        "data": data, "unresolved": unresolved})
    return data, unresolved, 0


def daily_closes(symbol: str, days: int = 90) -> dict:
    """Daily closes for a pair no public exchange in our set lists. date -> (close, volume)."""
    mapping = (coin_map([symbol]).get("map") or {}).get(str(symbol).upper())
    if not mapping:
        return {}
    try:
        payload = _fetch(CHART_URL.format(coin_id=mapping["id"], days=days))
    except Exception:
        return {}
    volumes = {int(stamp / 86400000): value for stamp, value in payload.get("total_volumes") or []}
    out = {}
    for stamp, price in payload.get("prices") or []:
        day = dt.datetime.fromtimestamp(stamp / 1000, dt.timezone.utc).date()
        out[day] = (float(price), float(volumes.get(int(stamp / 86400000), 0.0)))
    return out


def as_prompt_section(symbols) -> str:
    data, unresolved, age = markets(symbols)
    if not data:
        return "Market-wide coin data unavailable this run."

    lines = [f"Every tradable pair with market data, from CoinGecko ({'fresh' if not age else f'{age} min old'}). "
             "Each ticker was matched by verifying the candidate's price against this system's own, so a "
             "ticker shared by several coins is either matched correctly or left out below.",
             "'turnover' is 24h volume over market cap — under about 0.02 the pair is thin, and a position is "
             "easier to enter than to leave. 'dilution' is fully diluted value over market cap: 2.0 means half "
             "the supply is still to be issued, which is future selling pressure regardless of the story. "
             "'from ATH' is distance from the all-time high.",
             f"{'pair':<8} {'mcap':>9} {'turnover':>9} {'dilution':>9} {'24h':>7} {'7d':>7} {'30d':>7} {'from ATH':>9}"]
    rows = sorted(data.items(), key=lambda item: -(item[1].get("market_cap") or 0))
    for symbol, row in rows:
        market_cap = row.get("market_cap") or 0
        size = (f"${market_cap / 1e9:.1f}B" if market_cap >= 1e9 else
                f"${market_cap / 1e6:.0f}M" if market_cap else "n/a")
        turnover = f"{(row['volume_24h'] / market_cap):.3f}" if market_cap else "n/a"
        dilution = f"{(row['fdv'] / market_cap):.2f}x" if row.get("fdv") and market_cap else "n/a"
        pct = lambda v: f"{v:+.1f}%" if v is not None else "    n/a"
        lines.append(f"{symbol:<8} {size:>9} {turnover:>9} {dilution:>9} {pct(row.get('change_24h')):>7} "
                     f"{pct(row.get('change_7d')):>7} {pct(row.get('change_30d')):>7} "
                     f"{pct(row.get('from_ath')):>9}")
    if unresolved:
        lines.append("")
        lines.append("No verified market data (ticker shared with other coins, or no price to verify against) — "
                     "quote these directly and judge them on the broker's own data: "
                     + ", ".join(f"{row['symbol']}" for row in unresolved))
    return "\n".join(lines)
