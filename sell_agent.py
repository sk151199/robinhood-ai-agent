"""Sell agent: woken by exit_watch when a position breaks a level, and can only sell.

The trading agent rules on holdings at 7:30, 11:30 and 1:30. Between those, nothing did — a
thesis could break overnight and wait hours for a decision. This session exists for that gap.

It is given the sell tool and nothing that opens a position: the gate runs in sell_only mode, so
a buy is rejected in code no matter what this agent concludes. Its exits are exempt from the
daily trade budget, because that budget exists to curb churn and must not trap a position; the
ceiling on how often it can be woken lives in exit_watch instead.

Holding is a legitimate outcome. A tripped level is a reason to look, not an instruction to sell.
"""

import datetime as dt
import os

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage, SystemMessage

import exit_watch
import journal
import macro_feed
import positions_store
import rh_tools
import risk_policy
import scorecard
from config import settings
from order_gate import OrderGate
from risk_manager import RiskManager
from trade_logger import log_trade, logger

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = """You manage exits for a small crypto account on Robinhood, and you can only sell. Buying is
rejected in code, so every decision you make is: sell all of a position, sell part of it, or hold it.

You were woken because a position crossed a level — a stop guideline, a take-profit guideline, or a fast drop.
That is a reason to look, not an instruction to act. Most of the time the right answer is to hold, and saying so
with a reason is a complete, successful session.

How to decide:
- Start from the thesis. You are shown why each position was bought and what the buyer said would prove it
  wrong. Sell when that has happened: the driver is gone, the invalidation level is through, or the story has
  changed. A price move on its own is not an answer.
- A stop is a guideline, not a trigger. Ask whether the drop is this asset breaking or the whole market being
  down. If everything fell together and the thesis is intact, selling into it locks a loss and pays the spread.
- Take profit deliberately. If the thesis is playing out and has further to run, trimming part beats selling all.
  If the move was the whole thesis, take it.
- Costs are real and asymmetric here: a round trip costs about 1.9%, and you cannot buy back — only the trading
  agent can, and not until its next scheduled run, hours away. So an exit you regret cannot be undone quickly.
  That argues for holding through noise and selling decisively when the thesis is actually broken.
- Size the exit. Selling part is often right: it takes risk off without ending the position.
- Refresh before acting. Quote the pair immediately before ordering; account and position data older than the
  freshness limits are rejected.

Before any order: get_accounts (use the agentic one), get_portfolio, get_crypto_positions, then get_crypto_quotes
for what you will sell. Sell with dollar_amount, market orders, time_in_force gtc. Never sell more than is held.

Finish with a short plain-text summary: what you looked at, what you did or did not do, and why."""


def _prompt(found: list, scheduled: bool = False) -> str:
    opening = ("This is a scheduled review: rule on every holding below — sell, trim or hold — before the "
               "trading agent looks for new buys with whatever cash you leave it. Realising a profit that has "
               "run its course is as much your job as cutting a broken thesis."
               if scheduled else
               "You were woken between runs because a position broke a level.")
    return "\n".join([
        opening,
        f"Time: {dt.datetime.now().astimezone().isoformat(timespec='minutes')}",
        "LIVE. place_crypto_order sells real holdings. You cannot buy.",
        "",
        "" if scheduled else exit_watch.as_prompt_section(found),
        "",
        "Every open position, with the thesis recorded when it was bought:",
        scorecard.position_review(),
        "",
        "Macro tape (live):",
        macro_feed.as_prompt_section(),
        "",
        "Recent cycles from the trading agent:",
        journal.as_prompt_section(3),
        "",
        "Decide on every position above: sell all, sell part, or hold, with reasons.",
    ])


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def allowed_tools() -> list:
    """Account reads, quotes and the order tool. No research tools: this is a decision about
    positions already held, made in minutes, not a research session."""
    names = set(rh_tools.ACCOUNT_TOOLS) | set(rh_tools.HARNESS_TOOLS) | {
        rh_tools.PREFIX + "get_crypto_quotes", rh_tools.PREFIX + "get_currency_pairs"}
    return sorted(names | set(rh_tools.gated_order_tools()))


async def review_all() -> list:
    """Every holding gets a ruling, for the scheduled runs.

    exit_watch covers the hours between runs by exception; this covers the runs themselves, so
    realising a profit does not wait for a threshold to trip.
    """
    positions, _updated = positions_store.load()
    held = [{"symbol": symbol, "quantity": quantity, "price": None, "entry": None, "move": None,
             "reasons": ["scheduled review of every holding"]}
            for symbol, (quantity, _sellable) in sorted(positions.items()) if quantity > 0]
    if not held:
        logger.info("Sell agent: no open positions to review.")
        return []
    return await run(held, scheduled=True)


async def run(found: list, scheduled: bool = False) -> list:
    """Wake the sell agent for these positions. Returns the sells it placed."""
    risk = RiskManager(risk_policy.effective())
    gate = OrderGate(risk, sell_only=True, exempt_daily_limit=True)
    passthrough = rh_tools.passthrough_tools()
    pending, placed = {}, []
    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    outcome = {"summary": "", "cost": 0.0}

    async def pre_tool_use(input_data, tool_use_id, context):
        name = input_data["tool_name"]
        key = input_data.get("tool_use_id") or tool_use_id
        if name in gate.order_tools:
            verdict = gate.evaluate(name, input_data.get("tool_input") or {})
            if not verdict.allowed:
                logger.warning("SELL DENIED %s $%.2f: %s", verdict.symbol, verdict.requested, verdict.reason)
                return _deny(verdict.reason)
            logger.info("SELLING %s $%.2f (%s)", verdict.symbol, verdict.dollars, verdict.reason)
            pending[key] = verdict
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "allow",
                                           "permissionDecisionReason": verdict.reason,
                                           "updatedInput": verdict.updated_input}}
        if name in passthrough:
            return {}
        logger.warning("Sell agent blocked from %s", name)
        return _deny(f"{name} is not available to the sell agent")

    async def post_tool_use(input_data, tool_use_id, context):
        key = input_data.get("tool_use_id") or tool_use_id
        gate.observe(input_data["tool_name"], input_data.get("tool_input") or {}, input_data.get("tool_response"))
        verdict = pending.pop(key, None)
        if verdict:
            response = str(input_data.get("tool_response"))
            price = _fill_price(response, verdict.price, verdict.spread_pct)
            log_trade({
                "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "asset_class": verdict.asset_class,
                "symbol": verdict.symbol, "side": "sell", "requested_dollars": f"{verdict.requested:.2f}",
                "dollars": f"{verdict.dollars:.2f}", "price": f"{price:.8f}",
                "spread_pct": f"{verdict.spread_pct:.4f}", "decision": "placed",
                "reason": f"exit agent: {verdict.reason}", "dry_run": settings.dry_run,
                "ref_id": (verdict.updated_input or {}).get("ref_id", ""), "order_result": response[:500],
            })
            scorecard.record(verdict.symbol, "sell", verdict.dollars, price, verdict.spread_pct,
                             "placed", f"exit agent: {verdict.reason}")
            placed.append({"symbol": verdict.symbol, "side": "sell", "dollars": f"{verdict.dollars:.2f}",
                           "decision": "placed"})
        return {}

    options = ClaudeAgentOptions(
        cwd=PROJECT_DIR,
        setting_sources=["project"],
        model=settings.claude_model,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=allowed_tools(),
        disallowed_tools=rh_tools.disallowed_tools(),
        permission_mode="dontAsk",
        hooks={"PreToolUse": [HookMatcher(hooks=[pre_tool_use])],
               "PostToolUse": [HookMatcher(hooks=[post_tool_use])]},
        env={"MAX_MCP_OUTPUT_TOKENS": "50000"},
        max_turns=settings.sell_max_turns,
        max_budget_usd=settings.sell_budget_usd,
        thinking={"type": "adaptive"},
        effort=settings.sell_effort,
    )

    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(_prompt(found, scheduled))
            async for message in client.receive_response():
                if isinstance(message, SystemMessage) and message.subtype == "init":
                    statuses = {s.get("name"): s.get("status") for s in message.data.get("mcp_servers", [])}
                    if statuses.get(rh_tools.SERVER) in ("failed", "needs-auth", "disabled", None):
                        logger.error("Sell agent: robinhood-trading MCP unavailable; no exit decision made.")
                        return []
                elif isinstance(message, ResultMessage):
                    outcome["summary"] = (message.result or "").strip()
                    outcome["cost"] = message.total_cost_usd or 0.0
                    logger.info("Sell agent ended: %s after %d turns, cost $%.4f",
                                message.subtype, message.num_turns, message.total_cost_usd or 0.0)
    except Exception:
        logger.exception("Sell agent failed; positions left untouched.")
        return []

    import agent
    if agent.usage_limited(outcome["summary"]):
        logger.error("Sell agent stopped by the Claude usage limit; positions left untouched.")
        return []

    logger.info("Sell agent: %s", outcome["summary"][:2000])
    journal.append(mode="exit" if not settings.dry_run else "exit_dry_run", summary=outcome["summary"],
                   orders=placed, cost_usd=outcome["cost"])
    scorecard.attach_summary(outcome["summary"], started_at)
    for error in gate.parse_errors:
        logger.warning("Unparseable tool result (orders fail closed): %s", error)
    return placed


def _fill_price(response: str, mid: float, spread_pct: float) -> float:
    import agent
    return agent.TradingAgent._fill_price(response, mid, spread_pct, "sell")
