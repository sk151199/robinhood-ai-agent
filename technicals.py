"""Volume and price structure, computed from exchange data for the research desk.

The operator asked for TradingView's volume analytics. TradingView publishes no free market-data
API, and its scanner endpoint is undocumented and off-limits under its terms, so this computes
the same measures from the venues TradingView itself reads: Coinbase for candles and order-book
depth, OKX for perpetual open interest. Nothing here is scraped and nothing needs a key.

What it answers that the daily table cannot:
  - is today's volume actually unusual for this pair, or does it just look big?
  - is volume rising into a fall (distribution) or into a rise (accumulation)?
  - where is price against the volume-weighted average the day's traders actually paid?
  - how much size can the book absorb before the price moves, which bounds a sensible order?
  - is open interest building, meaning the move is leveraged rather than spot-funded?

Broad measures (RSI, on-balance volume) come from the daily candles already cached by
market_context, so they cost nothing. The intraday measures are fetched per symbol and are
therefore computed only for a focus list: what is held, plus what the monitor has flagged.
"""

import datetime as dt
import json
import os
import urllib.request

import market_context
from config import settings

CACHE_PATH = os.path.join(settings.log_dir, "technicals_cache.json")
CACHE_MINUTES = 20
HOURLY_URL = "https://api.exchange.coinbase.com/products/{symbol}-USD/candles?granularity=3600"
BOOK_URL = "https://api.exchange.coinbase.com/products/{symbol}-USD/book?level=2"
# OKX quotes a few perps against USD and most against USDT; try both before giving up.
OPEN_INTEREST_URL = "https://www.okx.com/api/v5/public/open-interest?instType=SWAP&instId={symbol}-{quote}-SWAP"

RSI_PERIOD = 14
OBV_DAYS = 14
FOCUS_MAX = 10
# Depth is measured inside this band: size that can trade without moving the price much.
DEPTH_BAND = 0.01


def _fetch(url: str):
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "robinhood-ai-agent", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=12) as response:
        return json.loads(response.read())


def rsi(closes: list, period: int = RSI_PERIOD):
    """Wilder's RSI. None when there is not enough history."""
    if len(closes) <= period:
        return None
    gains = losses = 0.0
    for previous, current in zip(closes[:period], closes[1:period + 1]):
        change = current - previous
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    for previous, current in zip(closes[period:-1], closes[period + 1:]):
        change = current - previous
        avg_gain = (avg_gain * (period - 1) + max(change, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-change, 0.0)) / period
    if avg_loss == 0:
        # A flat series is not overbought; only real gains are.
        return 100.0 if avg_gain > 0 else 50.0
    strength = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + strength))


def obv_trend(series: dict, days: int = OBV_DAYS):
    """On-balance volume slope, normalised by average daily volume.

    Volume is added on up days and subtracted on down days. A rising OBV while price is flat is
    accumulation; a falling OBV while price rises means the advance is not being paid for.
    """
    ordered = sorted(series)[-(days + 1):]
    if len(ordered) < 6:
        return None
    balance, points, volumes = 0.0, [], []
    for previous, day in zip(ordered, ordered[1:]):
        close_before, close_now = series[previous][0], series[day][0]
        volume = series[day][1]
        volumes.append(volume)
        balance += volume if close_now > close_before else (-volume if close_now < close_before else 0.0)
        points.append(balance)
    average = sum(volumes) / len(volumes) if volumes else 0
    if not average or len(points) < 4:
        return None
    # Slope per day of the normalised OBV line, by least squares.
    xs = list(range(len(points)))
    mean_x = sum(xs) / len(xs)
    mean_y = sum(points) / len(points)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if not denominator:
        return None
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, points)) / denominator
    return slope / average


def broad(symbols) -> dict:
    """RSI and OBV slope for every pair with cached daily history. No network calls."""
    out = {}
    for symbol in dict.fromkeys(symbols):
        series = market_context.series(symbol)
        if len(series) < RSI_PERIOD + 2:
            continue
        ordered = sorted(series)
        closes = [series[day][0] for day in ordered]
        out[symbol.upper()] = {"rsi": rsi(closes), "obv": obv_trend(series)}
    return out


def _intraday(symbol: str) -> dict:
    """Hourly volume, VWAP and open interest for one pair."""
    rows = _fetch(HOURLY_URL.format(symbol=symbol))
    # Coinbase rows are [time, low, high, open, close, volume], newest first.
    rows = [row for row in rows if isinstance(row, list) and len(row) >= 6]
    if len(rows) < 48:
        return {}
    recent = rows[:6]
    baseline = rows[6:]
    recent_volume = sum(row[5] for row in recent) / len(recent)
    baseline_volume = sum(row[5] for row in baseline) / len(baseline)
    day = rows[:24]
    typical = [((row[1] + row[2] + row[4]) / 3, row[5]) for row in day]
    traded = sum(volume for _price, volume in typical)
    vwap = sum(price * volume for price, volume in typical) / traded if traded else None
    price = rows[0][4]

    out = {
        "price": price,
        "volume_6h_ratio": (recent_volume / baseline_volume) if baseline_volume else None,
        "change_6h": (price - rows[5][4]) / rows[5][4] if len(rows) > 5 and rows[5][4] else None,
        "change_24h": (price - rows[23][4]) / rows[23][4] if len(rows) > 23 and rows[23][4] else None,
        "vs_vwap": (price - vwap) / vwap if vwap else None,
    }
    try:
        book = _fetch(BOOK_URL.format(symbol=symbol))
        bids, asks = book.get("bids") or [], book.get("asks") or []
        if bids and asks:
            mid = (float(bids[0][0]) + float(asks[0][0])) / 2
            depth = sum(float(p) * float(s) for p, s, *_ in bids if float(p) >= mid * (1 - DEPTH_BAND))
            depth += sum(float(p) * float(s) for p, s, *_ in asks if float(p) <= mid * (1 + DEPTH_BAND))
            out["depth_1pct"] = depth
    except Exception:
        pass
    for quote in ("USD", "USDT"):
        try:
            payload = _fetch(OPEN_INTEREST_URL.format(symbol=symbol, quote=quote))
            rows = payload.get("data") or []
            if rows and price:
                out["open_interest_usd"] = float(rows[0]["oiCcy"]) * price
                break
        except Exception:
            continue
    return out


def intraday(symbols) -> tuple:
    """(symbol -> intraday measures, age in minutes). Cached, because each symbol costs 3 calls."""
    try:
        with open(CACHE_PATH, encoding="utf-8") as handle:
            cached = json.load(handle)
        age = dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(cached["fetched_at"])
        if age < dt.timedelta(minutes=CACHE_MINUTES) and set(symbols) <= set(cached.get("data", {})):
            return cached["data"], int(age.total_seconds() // 60)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass

    data = {}
    for symbol in list(dict.fromkeys(symbols))[:FOCUS_MAX]:
        try:
            measures = _intraday(symbol)
        except Exception:
            continue
        if measures:
            data[symbol.upper()] = measures
    if data:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            tmp_path = f"{CACHE_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump({"fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(), "data": data}, handle)
            os.replace(tmp_path, CACHE_PATH)
        except OSError:
            pass
    return data, 0


def focus_symbols(held=(), flagged=(), limit: int = FOCUS_MAX) -> list:
    """What deserves the per-symbol calls: holdings first, then what the monitor flagged."""
    return list(dict.fromkeys([str(s).upper() for s in held] + [str(s).upper() for s in flagged]))[:limit]


def as_prompt_section(universe=(), held=(), flagged=()) -> str:
    measures = broad(universe)
    focus = focus_symbols(held, flagged)
    detail, age = intraday(focus) if focus else ({}, None)
    if not measures and not detail:
        return "Volume and structure feed unavailable this run."

    lines = ["Computed by code from Coinbase candles and order book and OKX open interest — the same venue data "
             "a charting site derives its indicators from."]

    if detail:
        lines.append("")
        lines.append(f"Intraday structure for your holdings and today's flagged movers "
                     f"({'fresh' if not age else f'{age} min old'}):")
        lines.append(f"{'pair':<7} {'6h vol':>7} {'6h':>7} {'24h':>7} {'vs VWAP':>8} {'book ±1%':>10} {'OI':>9}")
        for symbol, row in sorted(detail.items(), key=lambda item: -(item[1].get("volume_6h_ratio") or 0)):
            pct = lambda v: f"{v:+.1%}" if v is not None else "    n/a"
            vol = f"{row['volume_6h_ratio']:.1f}x" if row.get("volume_6h_ratio") else "    n/a"
            depth = f"${row['depth_1pct'] / 1e6:.1f}M" if row.get("depth_1pct") else "       n/a"
            interest = f"${row['open_interest_usd'] / 1e6:.0f}M" if row.get("open_interest_usd") else "      n/a"
            lines.append(f"{symbol:<7} {vol:>7} {pct(row.get('change_6h')):>7} {pct(row.get('change_24h')):>7} "
                         f"{pct(row.get('vs_vwap')):>8} {depth:>10} {interest:>9}")
        lines.append("'6h vol' is the last six hours against this pair's own 15-day hourly average, so 3x means "
                     "volume is genuinely unusual for it rather than merely large. 'vs VWAP' is price against the "
                     "volume-weighted average of the last 24 hours — above it, today's buyers are in profit and "
                     "supply comes from them; below it, the opposite. 'book ±1%' is the size the order book absorbs "
                     "within one percent, which bounds what can be traded without moving the price. Rising open "
                     "interest alongside a rally means leverage is funding it, and leveraged rallies unwind hard.")

    ranked = [(symbol, row) for symbol, row in measures.items() if row.get("rsi") is not None]
    if ranked:
        hot = sorted(ranked, key=lambda item: -item[1]["rsi"])[:6]
        cold = sorted(ranked, key=lambda item: item[1]["rsi"])[:6]
        lines.append("")
        lines.append("Daily RSI across the tradable universe (14-day; above 70 is stretched, below 30 is washed out, "
                     "and neither is a signal on its own):")
        lines.append("  stretched: " + ", ".join(f"{s} {r['rsi']:.0f}" for s, r in hot))
        lines.append("  washed out: " + ", ".join(f"{s} {r['rsi']:.0f}" for s, r in cold))

    accumulating = sorted((item for item in measures.items() if item[1].get("obv") is not None),
                          key=lambda item: -item[1]["obv"])
    if accumulating:
        up = [f"{s} {r['obv']:+.2f}" for s, r in accumulating[:5]]
        down = [f"{s} {r['obv']:+.2f}" for s, r in accumulating[-5:]]
        lines.append("")
        lines.append("On-balance volume slope over 14 days, in daily volumes per day. Positive means volume is "
                     "arriving on up days (accumulation), negative on down days (distribution). Compare it with the "
                     "price move: accumulation while the price is flat is the setup worth investigating, and a price "
                     "rising on negative OBV is an advance nobody is paying for.")
        lines.append("  accumulating: " + ", ".join(up))
        lines.append("  distributing: " + ", ".join(reversed(down)))
    return "\n".join(lines)
