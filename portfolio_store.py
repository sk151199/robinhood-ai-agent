"""Last observed account state, so code outside a session can reason about affordability.

The opportunity watch must decide whether a setup is worth waking four agents for. Without the
account's cash it would wake them for a $2 balance, spending a full cycle of Claude usage to be
told there is nothing to buy with. The gate observes the portfolio on every run, so the figure
is as fresh as the last cycle.
"""

import datetime as dt
import json
import os

from config import settings

PORTFOLIO_PATH = os.path.join(settings.log_dir, "portfolio_snapshot.json")


def save(total_value: float, buying_power: float):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{PORTFOLIO_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"total_value": float(total_value), "crypto_buying_power": float(buying_power),
                       "at": dt.datetime.now(dt.timezone.utc).isoformat()}, handle)
        os.replace(tmp_path, PORTFOLIO_PATH)
    except (OSError, TypeError, ValueError):
        pass


def load() -> dict:
    try:
        with open(PORTFOLIO_PATH, encoding="utf-8") as handle:
            snapshot = json.load(handle)
        return snapshot if isinstance(snapshot, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def age_minutes():
    at = load().get("at")
    if not at:
        return None
    return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(at)).total_seconds() / 60
