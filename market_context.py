"""Price history, volume and range context for crypto pairs.

Robinhood's MCP exposes no crypto historicals, so without this the agent can only see the
current mark against the previous close — it cannot tell a pullback from a peak. Coinbase's
public candles fill the gap: 350 days of daily closes and volume, no API key.

Candles are cached once per UTC day. History does not change, and a daily cycle would
otherwise re-download the same series every run. This is also the single fetch point for
scorecard.py, so scoring and the prompt share one cache.
"""

import datetime as dt
import json
import math
import os
import urllib.error
import urllib.request
from statistics import mean, pstdev

from config import settings

CACHE_PATH = os.path.join(settings.log_dir, "candles_cache.json")
CANDLES_URL = "https://api.exchange.coinbase.com/products/{symbol}-USD/candles?granularity=86400"
BENCHMARK = "BTC"
# Coinbase keeps serving candles for products it has delisted. LIT's history ended in May 2025
# while Robinhood's LIT trades today at 10x that last close, and the stale series was presented
# as a live 40% daily crash. History whose newest candle is older than this is discarded.
MAX_CANDLE_AGE_DAYS = 3

# Liquid names worth showing even when the agent has never traded them.
MAJORS = ("BTC", "ETH", "SOL", "XRP", "DOGE", "ADA", "LINK", "AVAX", "DOT", "LTC",
          "BCH", "XLM", "UNI", "ATOM", "NEAR", "ETC", "SHIB", "AAVE", "ALGO", "HBAR")


def _fetch(url: str) -> bytes:
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "robinhood-ai-agent"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def _load_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            cached = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return {}
    if cached.get("fetched_on") != dt.datetime.now(dt.timezone.utc).date().isoformat():
        return {}
    symbols = cached.get("symbols")
    return symbols if isinstance(symbols, dict) else {}


def _save_cache(symbols: dict):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{CACHE_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"fetched_on": dt.datetime.now(dt.timezone.utc).date().isoformat(),
                       "symbols": symbols}, handle)
        os.replace(tmp_path, CACHE_PATH)
    except OSError:
        pass


def _row_values(value) -> tuple:
    """Cache entries are [close, volume]; tolerate a bare close from an older cache."""
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return float(value), 0.0


def _current(parsed: dict) -> dict:
    if not parsed:
        return {}
    age = (dt.datetime.now(dt.timezone.utc).date() - max(parsed)).days
    return parsed if age <= MAX_CANDLE_AGE_DAYS else {}


def series(symbol: str) -> dict:
    """date -> (close, volume). Empty when Coinbase does not list the pair, or only has
    history for a delisted product under the same ticker."""
    symbol = symbol.upper()
    cache = _load_cache()
    if symbol in cache:
        cached = _current({dt.date.fromisoformat(day): _row_values(value) for day, value in cache[symbol].items()})
        if cached:
            return cached
        # The cached series belongs to a delisted product under this ticker. Fall through so the
        # fallback source can replace it, rather than reporting the pair as having no history.

    try:
        rows = json.loads(_fetch(CANDLES_URL.format(symbol=symbol)))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError, TimeoutError):
        rows = []   # Coinbase does not list this pair; the fallback below covers it.

    # Coinbase rows are [time, low, high, open, close, volume], newest first.
    parsed = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, list) and len(row) >= 6:
            parsed[dt.datetime.fromtimestamp(row[0], dt.timezone.utc).date()] = (float(row[4]), float(row[5]))
    parsed = _current(parsed)
    if not parsed:
        # Coinbase lists 82 of the 90 tradable pairs. The rest had no history at all, so they
        # never appeared in the table, the RSI list or the mover feed — invisible, and therefore
        # unconsiderable. The fallback series is ticker-verified against a live price.
        try:
            import coin_universe
            parsed = _current(coin_universe.daily_closes(symbol))
        except Exception:
            parsed = {}
    if parsed:
        cache[symbol] = {day.isoformat(): [close, volume] for day, (close, volume) in parsed.items()}
        _save_cache(cache)
    return parsed


def closes(symbol: str) -> dict:
    """date -> close."""
    return {day: values[0] for day, values in series(symbol).items()}


def _change(prices: dict, days: int, latest_day: dt.date, latest: float):
    for offset in range(days, days + 5):
        earlier = prices.get(latest_day - dt.timedelta(days=offset))
        if earlier:
            return (latest - earlier) / earlier
    return None


def stats(symbol: str) -> dict:
    """Recent moves, 30-day range position, and whether volume is unusual."""
    data = series(symbol)
    if not data:
        return {}
    prices = {day: values[0] for day, values in data.items()}
    latest_day = max(prices)
    latest = prices[latest_day]

    recent_30 = [day for day in prices if day > latest_day - dt.timedelta(days=30)]
    window = [prices[day] for day in recent_30]
    high, low = max(window), min(window)

    # A drop on heavy volume is capitulation; the same drop on nothing is drift.
    volumes_30 = [data[day][1] for day in recent_30 if data[day][1] > 0]
    volumes_7 = [data[day][1] for day in recent_30
                 if day > latest_day - dt.timedelta(days=7) and data[day][1] > 0]
    volume_ratio = (mean(volumes_7) / mean(volumes_30)) if volumes_7 and volumes_30 else None

    # Realized daily volatility: the standard deviation of daily log returns. This is what
    # makes "crypto is more variable" a number the sizing rule can act on.
    ordered = sorted(recent_30)
    returns = [math.log(prices[day] / prices[previous])
               for previous, day in zip(ordered, ordered[1:])
               if prices[previous] > 0 and prices[day] > 0]
    volatility = pstdev(returns) if len(returns) > 5 else None

    return {
        "volatility_30d": volatility,
        "symbol": symbol.upper(),
        "price": latest,
        "change_1d": _change(prices, 1, latest_day, latest),
        "change_7d": _change(prices, 7, latest_day, latest),
        "change_30d": _change(prices, 30, latest_day, latest),
        "below_30d_high": (high - latest) / high if high else None,
        "above_30d_low": (latest - low) / low if low else None,
        "volume_ratio": volume_ratio,
    }


def excess_returns(symbol: str, days: int = 30) -> dict:
    """date -> daily return minus BTC's that day.

    Raw crypto correlations are all high because everything follows Bitcoin. Stripping BTC's
    move leaves what is specific to the pair, which is what makes two holdings the same bet.
    """
    prices = closes(symbol)
    benchmark = closes(BENCHMARK)
    if not prices or not benchmark:
        return {}
    ordered = sorted(day for day in prices if day in benchmark)[-(days + 1):]
    out = {}
    for previous, day in zip(ordered, ordered[1:]):
        if prices[previous] > 0 and benchmark[previous] > 0:
            out[day] = ((prices[day] - prices[previous]) / prices[previous]
                        - (benchmark[day] - benchmark[previous]) / benchmark[previous])
    return out


def correlation(first: str, second: str, days: int = 30):
    """Correlation of two pairs' BTC-excess returns, or None without enough overlap."""
    first, second = first.upper(), second.upper()
    if first == second:
        return 1.0
    a, b = excess_returns(first, days), excess_returns(second, days)
    shared = sorted(set(a) & set(b))
    if len(shared) < 10:
        return None
    xs = [a[day] for day in shared]
    ys = [b[day] for day in shared]
    mean_x, mean_y = mean(xs), mean(ys)
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    return cov / math.sqrt(var_x * var_y)


FALLERS_SHOWN = 20
GAINERS_SHOWN = 10
# 7-day volume this many times its 30-day average is worth naming outright: unusual volume is
# the earliest honest signal available here, and it was previously buried in a column.
BREAKOUT_MULTIPLE = 2.0


def as_prompt_section(symbols, always_include=(), focused=False) -> str:
    """Biggest 7-day fallers first, measured against BTC so beta is not mistaken for opportunity.

    The full tradable universe is scanned, but only the worst fallers are printed — a table of
    every pair would cost tokens every cycle to show names that have not moved. Anything held
    is always shown, whatever it did, because a holding cannot be ruled on unseen.
    """
    rows = [stats(symbol) for symbol in dict.fromkeys(symbols)]
    rows = [row for row in rows if row and row.get("change_7d") is not None]
    if not rows:
        return "Price history unavailable this cycle."

    # Every tradable pair gets a line. Truncating to the extremes hid the middle of the
    # distribution entirely, and a pair cannot be considered if it is never shown.
    scanned_rows = rows
    scanned = len(rows)
    rows.sort(key=lambda row: row["change_7d"])
    priced = {row["symbol"] for row in rows}
    unpriced = [symbol.upper() for symbol in dict.fromkeys(symbols) if symbol.upper() not in priced]

    benchmark = next((row for row in rows if row["symbol"] == BENCHMARK), None) or stats(BENCHMARK)
    benchmark_7d = benchmark.get("change_7d") if benchmark else None

    scope = (f"Only the {scanned} pairs the research desk named, plus your holdings and BTC, are listed"
             if focused else f"All {scanned} priced pairs are listed")
    lines = [
        f"Daily closes and volume from Coinbase, independent of Robinhood. {scope}, biggest 7-day fallers "
        f"first — every one of them is a candidate.",
        "'vs BTC' is the 7-day move minus Bitcoin's: negative means it fell harder than the market, which is "
        "where an idiosyncratic drop hides. 'vol' is 7-day volume against its 30-day average. 'sd' is realized "
        "daily volatility — a 6% pair swings twice as hard as a 3% one, and your order size is scaled down "
        "accordingly.",
        f"{'pair':<6} {'price':>12} {'1d':>7} {'7d':>7} {'30d':>7} {'vsBTC':>7} {'vol':>6} {'sd':>6}  range",
    ]
    for row in rows:
        def pct(value):
            return f"{value:+.1%}" if value is not None else "  n/a"
        excess = (row["change_7d"] - benchmark_7d) if benchmark_7d is not None else None
        volume = f"{row['volume_ratio']:.1f}x" if row.get("volume_ratio") else "  n/a"
        sigma = f"{row['volatility_30d']:.1%}" if row.get("volatility_30d") else "  n/a"
        span = ""
        if row["below_30d_high"] is not None and row["above_30d_low"] is not None:
            span = f"{row['below_30d_high']:.0%} below 30d high, {row['above_30d_low']:.0%} above 30d low"
        lines.append(f"{row['symbol']:<6} {row['price']:>12,.4f} {pct(row['change_1d']):>7} "
                     f"{pct(row['change_7d']):>7} {pct(row['change_30d']):>7} {pct(excess):>7} "
                     f"{volume:>6} {sigma:>6}  {span}")

    if unpriced:
        lines.append("")
        lines.append("Tradable but with no Coinbase price history, so no row above — quote them directly with "
                     f"get_crypto_quotes if a case exists: {', '.join(unpriced)}")

    breakouts = sorted((row for row in scanned_rows
                        if (row.get("volume_ratio") or 0) >= BREAKOUT_MULTIPLE),
                       key=lambda row: row["volume_ratio"], reverse=True)
    if breakouts:
        lines.append("")
        lines.append(f"Volume breakouts ({BREAKOUT_MULTIPLE:.0f}x+ the 30-day average) — something is happening "
                     "in these, for better or worse:")
        for row in breakouts[:8]:
            lines.append(f"  {row['symbol']:<6} {row['volume_ratio']:.1f}x volume, "
                         f"7d {row['change_7d']:+.1%}, {row['below_30d_high']:.0%} below its 30d high")
    return "\n".join(lines)
