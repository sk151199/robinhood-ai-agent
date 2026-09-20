"""Persisted catalog of Robinhood crypto pairs.

The catalog is paginated and large, so fetching all of it every cycle would cost more context
than the decision it informs. The order gate already parses every page the agent fetches, so
those pairs are accumulated here and replayed into later prompts. The agent pages once, then
sees the whole tradable universe for free on every subsequent cycle.

Halt status recorded here ages: a pair halted after the last fetch still looks tradable. That
is tolerable because Robinhood rejects a halted pair at order time regardless — this is the
outer of two checks, not the only one.
"""

import datetime as dt
import json
import os

import rh_parse
from config import settings

CATALOG_PATH = os.path.join(settings.log_dir, "pairs_catalog.json")
MAX_SYMBOLS_IN_PROMPT = 400


def load() -> tuple:
    """(pairs, updated_at_iso). Empty when nothing has been cached yet."""
    try:
        with open(CATALOG_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        pairs = data.get("pairs")
        return (pairs if isinstance(pairs, dict) else {}), str(data.get("updated_at", ""))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {}, ""


def update(pairs: dict):
    """Merge freshly parsed pairs in. Newer data wins per symbol."""
    if not pairs:
        return
    known, _ = load()
    known.update(pairs)
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{CATALOG_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump({"updated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "pairs": known}, handle)
        os.replace(tmp_path, CATALOG_PATH)
    except OSError:
        pass


def as_prompt_section() -> str:
    pairs, updated = load()
    if not pairs:
        return ("No pair catalog cached yet. Page get_currency_pairs (limit 50, follow the cursor) to build "
                "it — every page you fetch is remembered, so this cost is paid once, not every cycle.")

    tradable = sorted(symbol for symbol, pair in pairs.items()
                      if pair.get("tradable") and not rh_parse.blocks_orders(pair))
    halted = sorted(symbol for symbol, pair in pairs.items() if rh_parse.blocks_orders(pair))

    lines = [f"{len(tradable)} tradable pairs known (catalog updated {updated[:16].replace('T', ' ')} UTC):",
             ", ".join(tradable[:MAX_SYMBOLS_IN_PROMPT])]
    if len(tradable) > MAX_SYMBOLS_IN_PROMPT:
        lines.append(f"...and {len(tradable) - MAX_SYMBOLS_IN_PROMPT} more.")
    if halted:
        lines.append(f"Halted at last check, do not order: {', '.join(halted[:40])}")
    lines.append("Halt status above can be stale. Any pair here can be quoted directly with get_crypto_quotes; "
                 "page get_currency_pairs only for a symbol that is not listed.")
    return "\n".join(lines)
