"""Live mover monitor: watches every tradable pair between research runs and flags big moves.

Deliberately not an LLM. Spotting movers is arithmetic on prices, and it has to run every few
minutes — a model session per check would add latency and cost while contributing no judgement.
The judgement belongs to the research desk, which reads what this file records and investigates
why a pair moved.

One public Coinbase request returns price, 24h change and 24h volume change for every pair. The
endpoint has no short-window change, so each run appends a snapshot and 15m / 1h / 4h moves are
computed from our own history. After the machine sleeps that history has a gap, and the prompt
section says how much history actually exists rather than implying continuous coverage.

Run once per invocation: `python mover_monitor.py`. Scheduled every 5 minutes.
"""

import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    # Task Scheduler starts with no working directory of its own.
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)

import pairs_catalog  # noqa: E402
import rh_parse  # noqa: E402
from config import settings  # noqa: E402

PRODUCTS_URL = "https://api.coinbase.com/api/v3/brokerage/market/products?product_type=SPOT"
SNAPSHOTS_PATH = os.path.join(settings.log_dir, "mover_snapshots.jsonl")
LATEST_PATH = os.path.join(settings.log_dir, "movers_latest.json")
ALERTS_PATH = os.path.join(settings.log_dir, "mover_alerts.jsonl")

SNAPSHOT_RETENTION = dt.timedelta(hours=6)
WINDOWS = {"15m": dt.timedelta(minutes=15), "1h": dt.timedelta(hours=1), "4h": dt.timedelta(hours=4)}
# A snapshot this far from the target time is too far off to call it that window's move.
WINDOW_TOLERANCE = {"15m": dt.timedelta(minutes=7), "1h": dt.timedelta(minutes=15), "4h": dt.timedelta(minutes=40)}
# The same pair crossing the same threshold again inside this span is one event, not several.
ALERT_REPEAT_AFTER = dt.timedelta(hours=2)
STALE_AFTER = dt.timedelta(minutes=10)


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _fetch(url: str) -> bytes:
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "robinhood-ai-agent"})
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read()


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _universe() -> set:
    known, _ = pairs_catalog.load()
    return {symbol for symbol, pair in known.items() if pair.get("tradable") and not rh_parse.blocks_orders(pair)}


def fetch_market() -> dict:
    """symbol -> live price, 24h change and 24h volume change, for Robinhood-tradable pairs."""
    payload = json.loads(_fetch(PRODUCTS_URL))
    universe = _universe()
    market = {}
    for product in payload.get("products") or []:
        if product.get("quote_currency_id") != "USD" or product.get("trading_disabled"):
            continue
        symbol = rh_parse.crypto_symbol(product.get("base_currency_id") or "")
        price = _float(product.get("price"))
        if not symbol or not price or (universe and symbol not in universe):
            continue
        change = _float(product.get("price_percentage_change_24h"))
        volume_change = _float(product.get("volume_percentage_change_24h"))
        market[symbol] = {
            "price": price,
            "change_24h": change / 100 if change is not None else None,
            "volume_change_24h": volume_change / 100 if volume_change is not None else None,
        }
    return market


def _read_jsonl(path: str) -> list:
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _write_jsonl(path: str, entries: list):
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")
    os.replace(tmp_path, path)


def _price_at(snapshots: list, symbol: str, target: dt.datetime, tolerance: dt.timedelta):
    best, best_gap = None, None
    for snapshot in snapshots:
        gap = abs(dt.datetime.fromisoformat(snapshot["timestamp"]) - target)
        price = snapshot["prices"].get(symbol)
        if price and gap <= tolerance and (best_gap is None or gap < best_gap):
            best, best_gap = price, gap
    return best


def scan(now: dt.datetime = None) -> dict:
    """Take a snapshot, compute windowed moves, record any new alerts. Returns the latest view."""
    now = now or _now()
    market = fetch_market()
    if not market:
        raise RuntimeError("Coinbase returned no usable pairs")

    os.makedirs(settings.log_dir, exist_ok=True)
    history = [snapshot for snapshot in _read_jsonl(SNAPSHOTS_PATH)
               if now - dt.datetime.fromisoformat(snapshot["timestamp"]) <= SNAPSHOT_RETENTION]

    rows = []
    for symbol, live in market.items():
        row = {"symbol": symbol, **live}
        for window, span in WINDOWS.items():
            then = _price_at(history, symbol, now - span, WINDOW_TOLERANCE[window])
            row[f"change_{window}"] = (live["price"] - then) / then if then else None
        rows.append(row)

    history.append({"timestamp": now.isoformat(), "prices": {symbol: live["price"] for symbol, live in market.items()}})
    _write_jsonl(SNAPSHOTS_PATH, history)

    oldest = min(dt.datetime.fromisoformat(snapshot["timestamp"]) for snapshot in history)
    latest = {"timestamp": now.isoformat(), "history_minutes": int((now - oldest).total_seconds() // 60),
              "snapshots": len(history), "rows": rows}
    tmp_path = f"{LATEST_PATH}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(latest, handle)
    os.replace(tmp_path, LATEST_PATH)

    _record_alerts(rows, now)
    return latest


def _triggers(row: dict) -> list:
    """(kind, value) for every threshold this row crosses."""
    fired = []
    if row.get("change_1h") is not None and abs(row["change_1h"]) >= settings.mover_1h_pct:
        fired.append(("1h move", row["change_1h"]))
    if row.get("change_24h") is not None and abs(row["change_24h"]) >= settings.mover_24h_pct:
        fired.append(("24h move", row["change_24h"]))
    if row.get("volume_change_24h") is not None and row["volume_change_24h"] >= settings.mover_volume_surge_pct:
        fired.append(("volume surge", row["volume_change_24h"]))
    return fired


def _record_alerts(rows: list, now: dt.datetime):
    existing = _read_jsonl(ALERTS_PATH)
    recent = {(alert["symbol"], alert["kind"], alert["value"] > 0) for alert in existing
              if now - dt.datetime.fromisoformat(alert["timestamp"]) < ALERT_REPEAT_AFTER}
    fresh = []
    for row in rows:
        for kind, value in _triggers(row):
            if (row["symbol"], kind, value > 0) in recent:
                continue
            fresh.append({"timestamp": now.isoformat(), "symbol": row["symbol"], "kind": kind,
                          "value": round(value, 4), "price": row["price"]})
    if fresh:
        # Alerts older than two days no longer inform a research run; keep the file bounded.
        kept = [alert for alert in existing if now - dt.datetime.fromisoformat(alert["timestamp"]) < dt.timedelta(days=2)]
        _write_jsonl(ALERTS_PATH, kept + fresh)


def load_latest(refresh_if_stale: bool = True) -> dict:
    try:
        with open(LATEST_PATH, encoding="utf-8") as handle:
            latest = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        latest = None
    stale = not latest or _now() - dt.datetime.fromisoformat(latest["timestamp"]) > STALE_AFTER
    if stale and refresh_if_stale:
        try:
            return scan()
        except (urllib.error.URLError, OSError, ValueError, RuntimeError, TimeoutError):
            return latest
    return latest


def _pct(value) -> str:
    return f"{value:+.1%}" if value is not None else "n/a"


def as_prompt_section(alert_hours: int = 8, top: int = 8) -> str:
    latest = load_latest()
    if not latest or not latest.get("rows"):
        return "Live mover monitor unavailable this run."

    now = _now()
    age = int((now - dt.datetime.fromisoformat(latest["timestamp"])).total_seconds() // 60)
    rows = latest["rows"]
    history = latest.get("history_minutes", 0)
    lines = [f"Scanned {len(rows)} tradable pairs {age} minute(s) ago. The monitor holds {history} minutes of "
             "5-minute snapshots; a window longer than that shows n/a (the machine may have been asleep)."]

    def ranked(key, label):
        usable = [row for row in rows if row.get(key) is not None]
        if not usable:
            return
        usable.sort(key=lambda row: row[key])
        gainers = [row for row in reversed(usable) if row[key] > 0][:top]
        losers = [row for row in usable if row[key] < 0][:top]
        lines.append("")
        lines.append(f"Biggest {label} movers:")
        lines.append("  up:   " + (", ".join(f"{row['symbol']} {_pct(row[key])}" for row in gainers) or "none"))
        lines.append("  down: " + (", ".join(f"{row['symbol']} {_pct(row[key])}" for row in losers) or "none"))

    ranked("change_1h", "1-hour")
    ranked("change_4h", "4-hour")
    ranked("change_24h", "24-hour")

    surges = sorted((row for row in rows if (row.get("volume_change_24h") or 0) >= settings.mover_volume_surge_pct),
                    key=lambda row: row["volume_change_24h"], reverse=True)[:top]
    if surges:
        lines.append("")
        lines.append("Volume surging against the prior 24h:")
        lines += [f"  {row['symbol']}: volume {_pct(row['volume_change_24h'])}, price 24h {_pct(row['change_24h'])}"
                  for row in surges]

    cutoff = now - dt.timedelta(hours=alert_hours)
    alerts = [alert for alert in _read_jsonl(ALERTS_PATH) if dt.datetime.fromisoformat(alert["timestamp"]) >= cutoff]
    lines.append("")
    if alerts:
        lines.append(f"Threshold alerts in the last {alert_hours}h, oldest first (1h move >= "
                     f"{settings.mover_1h_pct:.0%}, 24h move >= {settings.mover_24h_pct:.0%}, volume >= "
                     f"+{settings.mover_volume_surge_pct:.0%}):")
        local = dt.datetime.now().astimezone().tzinfo
        for alert in alerts[-25:]:
            when = dt.datetime.fromisoformat(alert["timestamp"]).astimezone(local).strftime("%H:%M")
            lines.append(f"  {when} {alert['symbol']} {alert['kind']} {_pct(alert['value'])} at {alert['price']:g}")
    else:
        lines.append(f"No threshold alerts in the last {alert_hours}h.")
    return "\n".join(lines)


def main():
    from trade_logger import logger
    try:
        latest = scan()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError, TimeoutError) as error:
        logger.warning("Mover monitor scan failed: %s", error)
        return 1
    fired = sum(1 for row in latest["rows"] for _ in _triggers(row))
    logger.info("Mover monitor: %d pairs, %d min of history, %d threshold crossing(s)",
                len(latest["rows"]), latest["history_minutes"], fired)
    return 0


if __name__ == "__main__":
    sys.exit(main())
