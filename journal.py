"""Decision journal: what the agent did each cycle, fed back to it on the next one.

This is the closest thing to learning available here. The model's weights never change;
instead each cycle sees its own recent decisions, the reasoning behind them, and what the
orders actually were, so it can build on them instead of starting blind every hour.
"""

import datetime as dt
import json
import os

from config import settings

JOURNAL_PATH = os.path.join(settings.log_dir, "journal.jsonl")
MAX_SUMMARY_CHARS = 1200


def append(mode: str, summary: str, orders: list, cost_usd: float, usage: dict = None):
    entry = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "mode": mode,
        "cost_usd": round(cost_usd or 0.0, 4),
        # Token split, so cost per cycle can be attributed to reading vs deciding.
        "usage": usage or {},
        "orders": orders,
        "summary": (summary or "").strip()[:MAX_SUMMARY_CHARS],
    }
    os.makedirs(settings.log_dir, exist_ok=True)
    with open(JOURNAL_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")


def recent(limit: int) -> list:
    if limit <= 0:
        # lines[-0:] is the whole file, not nothing — guard before slicing.
        return []
    try:
        with open(JOURNAL_PATH, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []

    entries = []
    for line in lines[-limit:]:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def as_prompt_section(limit: int) -> str:
    entries = recent(limit)
    if not entries:
        return "No previous cycles recorded. This is your first."

    lines = []
    for entry in entries:
        when = entry["timestamp"][:16].replace("T", " ")
        orders = entry.get("orders") or []
        if orders:
            placed = "; ".join(
                f"{o.get('side')} {o.get('symbol')} ${o.get('dollars')} ({o.get('decision')})" for o in orders
            )
        else:
            placed = "no orders"
        lines.append(f"[{when} UTC, {entry.get('mode')}] {placed}\n  {entry.get('summary', '')}")
    return "\n\n".join(lines)
