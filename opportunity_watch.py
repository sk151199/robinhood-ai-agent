"""Watches for entry setups between scheduled runs and wakes the full pipeline when one appears.

The mirror of exit_watch. That one asks whether a position deserves a look; this asks whether a
pair does. Both are code, and both spend no Claude usage until something actually trips, which is
what makes fifteen-minute cover affordable.

It does not decide what to buy. It decides whether the market has produced something worth the
cost of a full cycle — research, risk, exits, entry — and hands the agents the reason it woke
them. A setup is a reason to look, and a cycle that looks and passes is a correct outcome.

Three guards keep it from being expensive noise:
  - affordability: no wake when the last known cash cannot fund the smallest sensible order, or
    when the day's order budget is already spent. Waking four agents to be told there is no money
    is the most wasteful thing this file could do.
  - exclusivity: no wake while another live cycle holds the lock, so a catch-up run and a setup
    cannot trade against each other.
  - scarcity: a daily cap on wakes, and a per-symbol cooldown, so one pair thrashing does not
    consume the day.

Run: `python opportunity_watch.py --live` (scheduled every 15 minutes). `--dry-run` reports only.
"""

import datetime as dt
import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    if "--live" in sys.argv:
        os.environ["DRY_RUN"] = "false"

import market_context  # noqa: E402
import mover_monitor  # noqa: E402
import portfolio_store  # noqa: E402
import positions_store  # noqa: E402
import risk_policy  # noqa: E402
import scorecard  # noqa: E402
import themes  # noqa: E402
from config import settings  # noqa: E402
from risk_manager import RiskManager  # noqa: E402
from trade_logger import logger  # noqa: E402

STATE_PATH = os.path.join(settings.log_dir, "opportunity_watch.json")
INSTRUCTION_PATH = os.path.join(settings.log_dir, "next_run_instruction.txt")
LOCK_PATH = os.path.join(settings.log_dir, "cycle.lock")

# What counts as a setup. Deliberately narrow: these fire on their own, unsupervised, and a
# cycle costs real usage.
DIP_1H = -0.05            # a sharp fall in an hour, where a thesis may be mispriced
SURGE_VOLUME = 3.0        # 24h volume this many times its prior day, with the price following
SURGE_24H = 0.08
CAPITULATION_24H = -0.12  # a large daily fall, where the question is whether the story broke

MIN_CASH = 15.0           # below this an order cannot clear minimums and costs meaningfully
MAX_WAKES_PER_DAY = 2
SYMBOL_COOLDOWN_HOURS = 6.0
PORTFOLIO_MAX_AGE_MINUTES = 24 * 60


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


def _wakes_today(state: dict) -> int:
    return int((state.get("wakes") or {}).get(_now().date().isoformat(), 0))


def _cycle_running() -> bool:
    try:
        with open(LOCK_PATH, encoding="utf-8") as handle:
            held = json.load(handle)
        age = (_now() - dt.datetime.fromisoformat(held["at"])).total_seconds() / 60
        return age < 120
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def setups(state: dict = None) -> list:
    """Pairs whose price action is worth a cycle, each with the reason and the numbers."""
    state = state if state is not None else _state()
    rows = (mover_monitor.load_latest() or {}).get("rows", [])
    held = positions_store.load()[0]
    recent_buys = scorecard.last_buy_times()
    limits = risk_policy.effective()
    fired = state.get("fired", {})
    found = []

    for row in rows:
        symbol = row["symbol"]
        change_1h, change_24h = row.get("change_1h"), row.get("change_24h")
        volume_change = row.get("volume_change_24h")

        reasons = []
        if change_1h is not None and change_1h <= DIP_1H:
            reasons.append(f"fell {change_1h:.1%} in an hour")
        if (volume_change is not None and volume_change >= SURGE_VOLUME - 1
                and change_24h is not None and change_24h >= SURGE_24H):
            reasons.append(f"volume {volume_change:+.0%} against yesterday with price {change_24h:+.1%}")
        if change_24h is not None and change_24h <= CAPITULATION_24H:
            reasons.append(f"down {change_24h:.1%} on the day")
        if not reasons:
            continue

        # A pair that cannot be bought is not an opportunity.
        last_buy = recent_buys.get(symbol)
        if last_buy and (_now() - last_buy).total_seconds() / 3600 < limits.cooldown_hours:
            continue
        fired_at = fired.get(symbol)
        if fired_at and (_now() - dt.datetime.fromisoformat(fired_at)).total_seconds() / 3600 < SYMBOL_COOLDOWN_HOURS:
            continue

        stats = market_context.stats(symbol) or {}
        found.append({
            "symbol": symbol, "price": row.get("price"), "reasons": reasons,
            "held": symbol in held,
            "theme": themes.theme_of(symbol) or "unlabelled",
            "change_7d": stats.get("change_7d"), "below_30d_high": stats.get("below_30d_high"),
            "volatility": stats.get("volatility_30d"),
        })
    # Strongest first: the largest single move is the most likely to be mispriced.
    found.sort(key=lambda row: min([r for r in (row.get("change_7d"), 0) if r is not None] or [0]))
    return found[:6]


def as_instruction(found: list) -> str:
    lines = ["Operator instruction for this run: this cycle was not scheduled. A fifteen-minute watch woke it "
             "because the market produced one or more of the setups below. Treat them as the reason you are "
             "here, not as a recommendation: verify each against live data, and pass on all of them if none "
             "survives your own checks — an unscheduled cycle that buys nothing is a correct outcome.",
             ""]
    for row in found:
        detail = []
        if row.get("change_7d") is not None:
            detail.append(f"7d {row['change_7d']:+.1%}")
        if row.get("below_30d_high") is not None:
            detail.append(f"{row['below_30d_high']:.0%} below its 30d high")
        if row.get("volatility"):
            detail.append(f"{row['volatility']:.1%} daily vol")
        lines.append(f"  {row['symbol']} at {row['price']:g} ({row['theme']}"
                     + (", already held" if row["held"] else "") + "): "
                     + "; ".join(row["reasons"])
                     + (" — " + ", ".join(detail) if detail else ""))
    lines.append("")
    lines.append("The research desk has not been told which pairs tripped; it will scan the market as usual, so "
                 "treat agreement between its brief and this list as meaningful and disagreement as a question "
                 "worth resolving before committing capital.")
    return "\n".join(lines)


def main() -> int:
    dry = "--dry-run" in sys.argv
    state = _state()
    found = setups(state)
    if not found:
        return 0

    summary = "; ".join(f"{row['symbol']} ({'; '.join(row['reasons'])})" for row in found)
    if dry:
        logger.info("Opportunity watch (dry run) would wake a cycle: %s", summary)
        return 0

    risk = RiskManager()
    if not risk.trade_budget_remaining():
        logger.info("Opportunity watch: %s tripped, but the day's order budget is spent.", summary)
        return 0

    snapshot = portfolio_store.load()
    age = portfolio_store.age_minutes()
    cash = snapshot.get("crypto_buying_power")
    if cash is None or (age is not None and age > PORTFOLIO_MAX_AGE_MINUTES):
        logger.info("Opportunity watch: %s tripped, but the account snapshot is missing or stale.", summary)
        return 0
    if cash < MIN_CASH:
        logger.info("Opportunity watch: %s tripped, but only $%.2f is available to buy with.", summary, cash)
        return 0

    if _cycle_running():
        logger.info("Opportunity watch: %s tripped while a cycle is already running; leaving it to that cycle.",
                    summary)
        return 0

    if _wakes_today(state) >= MAX_WAKES_PER_DAY:
        logger.info("Opportunity watch: %s tripped, but today's %d unscheduled cycles are used.",
                    summary, MAX_WAKES_PER_DAY)
        return 0

    if os.path.exists(INSTRUCTION_PATH):
        logger.info("Opportunity watch: an operator instruction is already waiting; not overwriting it.")
        return 0

    logger.info("Opportunity watch waking a live cycle: %s", summary)
    try:
        with open(INSTRUCTION_PATH, "w", encoding="utf-8") as handle:
            handle.write(as_instruction(found))
    except OSError as error:
        logger.error("Could not write the instruction file: %s", error)
        return 1

    state.setdefault("wakes", {})[_now().date().isoformat()] = _wakes_today(state) + 1
    fired = state.setdefault("fired", {})
    for row in found:
        fired[row["symbol"]] = _now().isoformat()
    state["last_wake"] = {"at": _now().isoformat(), "symbols": [row["symbol"] for row in found]}
    _save_state(state)

    result = subprocess.run([sys.executable, os.path.join(PROJECT_DIR, "run_once.py"), "--live"],
                            cwd=PROJECT_DIR)
    logger.info("Opportunity cycle finished with exit code %s.", result.returncode)
    return 0


if __name__ == "__main__":
    sys.exit(main())
