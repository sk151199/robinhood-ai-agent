"""Watches open positions between runs and wakes the sell agent when one breaks a level.

Scheduled runs are hours apart, so until now nothing looked at a position overnight: a thesis
could break at 2am and first be considered at 7:30. This is the cheap half of the answer — code
checking live prices every five minutes — and it spends no Claude usage until something actually
trips, which is what makes continuous cover affordable on a small account.

It never trades. It decides only whether a position deserves a look, and the sell agent decides
what to do about it.

Run: `python exit_watch.py` (scheduled every 5 minutes). `--dry-run` reports without waking anything.
"""

import asyncio
import datetime as dt
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    # Must be set before config is imported. A protective exit that can only be simulated is
    # not protection: on 2026-09-18 the 03:52 wake decided a UNI trim, found only the preview
    # tool bound, and the sale waited 11 hours for the live 7:30 session.
    if "--live" in sys.argv:
        os.environ["DRY_RUN"] = "false"

import mover_monitor  # noqa: E402
import positions_store  # noqa: E402
import scorecard  # noqa: E402
from config import settings  # noqa: E402
from trade_logger import logger  # noqa: E402

STATE_PATH = os.path.join(settings.log_dir, "exit_watch.json")
# One wake per position per window: a position sitting below its stop must not wake a session
# every five minutes.
RETRIGGER_HOURS = 4.0
# Ceiling on protective exits in a day, whatever the triggers say. Exits are exempt from the
# trading budget, so this is what stops a loop.
MAX_SELL_RUNS_PER_DAY = 4
# A fast break needs no threshold breach to be worth a look.
FAST_DROP_1H = -0.06


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _state() -> dict:
    try:
        with open(STATE_PATH, encoding="utf-8") as handle:
            state = json.load(handle)
        return state if isinstance(state, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _save_state(state: dict):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{STATE_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(state, handle, indent=2)
        os.replace(tmp_path, STATE_PATH)
    except OSError:
        pass


def _runs_today(state: dict) -> int:
    today = _now().date().isoformat()
    return int(state.get("runs", {}).get(today, 0))


def triggers(state: dict = None) -> list:
    """Positions worth a look right now, each with why.

    Prices come from the mover monitor's live scan, entries from the decision log. A position
    whose entry price is unknown is still checked for a fast break, since that needs no entry.
    """
    state = state if state is not None else _state()
    market = {row["symbol"]: row for row in (mover_monitor.load_latest() or {}).get("rows", [])}
    positions, _updated = positions_store.load()
    last_fired = state.get("fired", {})
    found = []

    for symbol, (quantity, _sellable) in sorted(positions.items()):
        if quantity <= 0 or symbol not in market:
            continue
        row = market[symbol]
        price = row.get("price")
        entry = (scorecard._last_buy(symbol) or {}).get("price")
        move = (price - entry) / entry if entry and price else None
        reasons = []
        if move is not None and move <= -settings.stop_loss_pct:
            reasons.append(f"down {move:.1%} from its ${entry:,.4f} entry, past the "
                           f"-{settings.stop_loss_pct:.0%} stop guideline")
        if move is not None and move >= settings.take_profit_pct:
            reasons.append(f"up {move:.1%} from its ${entry:,.4f} entry, past the "
                           f"+{settings.take_profit_pct:.0%} take-profit guideline")
        if (row.get("change_1h") or 0) <= FAST_DROP_1H:
            reasons.append(f"fell {row['change_1h']:.1%} in the last hour")
        if not reasons:
            continue

        fired_at = last_fired.get(symbol)
        if fired_at:
            hours = (_now() - dt.datetime.fromisoformat(fired_at)).total_seconds() / 3600
            if hours < RETRIGGER_HOURS:
                continue
        found.append({"symbol": symbol, "price": price, "entry": entry, "move": move,
                      "quantity": quantity, "reasons": reasons})
    return found


def as_prompt_section(found: list) -> str:
    lines = ["These positions tripped a level, which is why this session was woken:"]
    for row in found:
        move = f"{row['move']:+.1%} from entry" if row["move"] is not None else "entry price unknown"
        lines.append(f"  {row['symbol']}: {row['quantity']:.8f} units at ${row['price']:,.6f} ({move}) — "
                     + "; ".join(row["reasons"]))
    lines.append("A tripped level is a reason to look, not a reason to sell. Re-quote before deciding.")
    return "\n".join(lines)


def main() -> int:
    dry = "--dry-run" in sys.argv
    if not dry and settings.dry_run:
        logger.warning("Exit watch is running in DRY RUN: any exit it decides will only be simulated. "
                       "Pass --live to place real protective sells.")
    state = _state()
    found = triggers(state)
    if not found:
        return 0

    summary = "; ".join(f"{row['symbol']} ({'; '.join(row['reasons'])})" for row in found)
    if dry:
        logger.info("Exit watch (dry run) would wake the sell agent: %s", summary)
        return 0

    if _runs_today(state) >= MAX_SELL_RUNS_PER_DAY:
        logger.warning("Exit watch: %s tripped, but the daily cap of %d protective sell runs is used.",
                       summary, MAX_SELL_RUNS_PER_DAY)
        return 0

    if not settings.sell_agent_enabled:
        logger.info("Exit watch: %s tripped, but the sell agent is disabled.", summary)
        return 0

    logger.info("Exit watch waking the sell agent: %s", summary)
    import sell_agent
    placed = asyncio.run(sell_agent.run(found))

    today = _now().date().isoformat()
    state.setdefault("runs", {})[today] = _runs_today(state) + 1
    fired = state.setdefault("fired", {})
    for row in found:
        fired[row["symbol"]] = _now().isoformat()
    state["last_wake"] = {"at": _now().isoformat(), "symbols": [row["symbol"] for row in found],
                          "sold": placed}
    _save_state(state)
    return 0


if __name__ == "__main__":
    sys.exit(main())
