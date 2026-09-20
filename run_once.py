"""Run a single trading cycle and exit. Entry point for scheduled runs.

Windows Task Scheduler starts a task with no working directory of its own, so this resolves
the project directory before anything reads a relative path — otherwise logs, the journal and
the daily state file land wherever the scheduler happened to start.
"""

import argparse
import asyncio
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
INSTRUCTION_PATH = os.path.join(PROJECT_DIR, "logs", "next_run_instruction.txt")
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)


LOCK_PATH = os.path.join(PROJECT_DIR, "logs", "cycle.lock")
LOCK_STALE_MINUTES = 120


def _acquire_lock(logger) -> bool:
    """False when another live cycle is already running. A lock older than the task time limit
    is treated as abandoned, so a killed run cannot block trading indefinitely."""
    import datetime as dt
    try:
        with open(LOCK_PATH, encoding="utf-8") as handle:
            held = json.load(handle)
        age = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(held["at"])).total_seconds() / 60
        if age < LOCK_STALE_MINUTES:
            logger.warning("Another cycle started %.0f minutes ago (pid %s) still holds the lock; skipping.",
                           age, held.get("pid"))
            return False
        logger.warning("Ignoring a stale cycle lock from %.0f minutes ago.", age)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        pass
    os.makedirs(os.path.dirname(LOCK_PATH), exist_ok=True)
    with open(LOCK_PATH, "w", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "at": dt.datetime.now(dt.timezone.utc).isoformat()}, handle)
    return True


def _release_lock():
    try:
        os.remove(LOCK_PATH)
    except OSError:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run one trading cycle and exit.")
    parser.add_argument("--live", action="store_true", help="place real orders (default: dry run)")
    parser.add_argument("--require-trade", action="store_true",
                        help="demand exactly one buy this cycle instead of leaving it to the agent")
    args = parser.parse_args()

    # Must be set before config is imported: load_dotenv does not override a real env var,
    # so this wins over whatever DRY_RUN says in .env.
    os.environ["DRY_RUN"] = "false" if args.live else "true"

    import agent
    import main as loop
    from trade_logger import logger

    bot = agent.TradingAgent()
    bot.log_startup()

    if args.live and not _acquire_lock(logger):
        return

    # A one-shot operator instruction for the next live run (e.g. "deploy all buying power"). It is
    # removed only after a session actually ran, so a usage-limit or crashed run does not lose it,
    # and it can never apply to more than one completed run.
    instruction = ""
    if args.live and os.path.exists(INSTRUCTION_PATH):
        with open(INSTRUCTION_PATH, encoding="utf-8") as handle:
            instruction = handle.read().strip()
        if instruction:
            logger.info("One-shot operator instruction for this run: %s", instruction[:300])

    # On an hourly schedule most cycles arrive after the daily order budget is spent. Research,
    # risk and the trading session are skipped then, because they cannot act. Exits are not: the
    # budget curbs churn and must never stop a position being closed, so the sell agent still
    # rules on every holding. Skipping it here once left a spent budget silently disabling selling.
    if not bot.risk.trade_budget_remaining() and not instruction:
        logger.info("No buy budget left (%d used); skipping research and the trading session, "
                    "running the exit review only.", bot.risk.trades_today())
        if loop.settings.sell_agent_enabled and args.live:
            import sell_agent
            asyncio.run(sell_agent.review_all())
        return

    if instruction and not bot.risk.trade_budget_remaining():
        logger.info("Today's budget is spent, but an operator instruction is present: running in full so the "
                    "risk agent can re-set it.")

    async def cycle():
        # Research, then risk, then trade — each to completion before the next.
        brief, limits = await bot.prepare(bot.risk.day_start_value(), instruction)
        return await bot.run_cycle(loop.equity_market_open(), require_trade=args.require_trade,
                                   instruction=instruction, brief=brief, limits=limits)

    try:
        ran = asyncio.run(cycle())
    finally:
        if args.live:
            _release_lock()
    if instruction and ran:
        os.remove(INSTRUCTION_PATH)
        logger.info("One-shot instruction consumed and removed.")


if __name__ == "__main__":
    main()
