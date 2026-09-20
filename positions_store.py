"""Last observed crypto holdings, persisted between cycles.

The agent fetches positions from Robinhood; this code cannot. Recording what the gate parsed
lets the next prompt show each holding with its entry price and unrealized move, which is what
makes an exit decision possible rather than guesswork.

Written from the PostToolUse hook, so it always reflects Robinhood's own numbers.
"""

import datetime as dt
import json
import os

from config import settings

POSITIONS_PATH = os.path.join(settings.log_dir, "positions_cache.json")


def save(positions: dict):
    """positions: symbol -> (quantity, sellable)."""
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        payload = {"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                   "positions": {symbol: [float(quantity), float(sellable)]
                                 for symbol, (quantity, sellable) in positions.items()}}
        tmp_path = f"{POSITIONS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp_path, POSITIONS_PATH)
    except (OSError, TypeError, ValueError):
        pass


def load() -> tuple:
    """(positions, updated_at_iso) with positions as symbol -> (quantity, sellable)."""
    try:
        with open(POSITIONS_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        rows = payload.get("positions") or {}
        positions = {symbol: (float(values[0]), float(values[1]))
                     for symbol, values in rows.items() if isinstance(values, (list, tuple))}
        return positions, str(payload.get("updated_at", ""))
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return {}, ""
