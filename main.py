import asyncio
import datetime as dt
import os
import signal
import time
from zoneinfo import ZoneInfo

from agent import PROJECT_DIR, TradingAgent
from config import settings
from trade_logger import logger

EASTERN = ZoneInfo("America/New_York")


def equity_market_open(now: dt.datetime = None) -> bool:
    """Regular session by the clock. Exchange holidays are not modeled; on those days the
    server rejects equity orders, so the cost is one wasted cycle, not a bad trade."""
    now = (now or dt.datetime.now(EASTERN)).astimezone(EASTERN)
    return now.weekday() < 5 and dt.time(9, 30) <= now.time() < dt.time(16, 0)


def main():
    if not os.path.exists(os.path.join(PROJECT_DIR, ".mcp.json")):
        raise SystemExit(".mcp.json not found. See README: connect the robinhood-trading MCP server first.")

    agent = TradingAgent()
    agent.log_startup()
    state = {"running": True}

    def shutdown(_signum, _frame):
        logger.info("Shutdown requested; exiting after the current cycle.")
        state["running"] = False

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while state["running"]:
        market_open = equity_market_open()
        if settings.market_hours_only and not market_open:
            logger.info("US equity market closed; skipping this cycle (MARKET_HOURS_ONLY=true).")
        else:
            try:
                async def cycle():
                    brief, limits = await agent.prepare(agent.risk.day_start_value())
                    return await agent.run_cycle(market_open, brief=brief, limits=limits)

                asyncio.run(cycle())
            except Exception:
                logger.exception("Trading cycle failed; retrying next interval.")

        for _ in range(settings.poll_interval_seconds):
            if not state["running"]:
                break
            time.sleep(1)


if __name__ == "__main__":
    main()
