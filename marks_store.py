"""Robinhood's own marks for pairs the agents have quoted.

Public venues do not list everything Robinhood trades, and several tickers are ambiguous across
sources. These marks are the broker's own view of the price, so they anchor any mapping from a
Robinhood symbol to an outside data source: a candidate that disagrees with the mark is a
different asset wearing the same ticker.
"""

import datetime as dt
import json
import os

from config import settings

MARKS_PATH = os.path.join(settings.log_dir, "rh_marks.json")


def load() -> dict:
    try:
        with open(MARKS_PATH, encoding="utf-8") as handle:
            marks = json.load(handle)
        return marks if isinstance(marks, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def remember(quotes: dict):
    """quotes is symbol -> {"price": float, ...} as parsed from get_crypto_quotes."""
    if not quotes:
        return
    marks = load()
    stamped = dt.datetime.now(dt.timezone.utc).isoformat()
    for symbol, quote in quotes.items():
        price = (quote or {}).get("price")
        if price and price > 0:
            marks[str(symbol).upper()] = {"price": float(price), "at": stamped}
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{MARKS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(marks, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, MARKS_PATH)
    except OSError:
        pass


def price(symbol: str):
    return (load().get(str(symbol).upper()) or {}).get("price")
