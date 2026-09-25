"""Risk agent: sets how much the trader may risk this run, and cannot trade.

Deliberately a separate session from the trader. If the agent placing the orders also set its
own limits, one bad judgement would be unbounded — and this system has already shown it can be
confidently wrong about a number (it cited a 5.04% 10-year yield when the live figure was 4.945%).
Separating the two means the trader's conviction cannot quietly widen its own leash.

It reads the growth target, the live account, its own scored record, its past policies and the
macro tape, then returns one policy for the run. Code validates and enforces it. The operator's
daily loss breaker is outside its reach, and an unusable policy falls back to fixed limits.
"""

import datetime as dt
import json
import os

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage

import fundamentals
import lessons
import macro_feed
import research_agent
import rh_tools
import risk_policy
import scorecard
import themes
from config import settings
from trade_logger import logger

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Account reads only: it must see the real balance it is sizing against, and nothing that trades.
ACCOUNT_READS = rh_tools._full("get_accounts", "get_portfolio", "get_crypto_positions",
                               "get_realized_pnl", "get_pnl_trade_history")

SYSTEM_PROMPT = """You set the risk limits for one trading session of a small crypto account on Robinhood, then
stop. You do not trade and cannot: you have no order tools. A separate trading agent then trades inside the
limits you set and cannot change them, and code enforces them on every order.

Your job is position sizing and pacing against a growth target, not stock picking. You are given the target and
the account's progress toward it, the live balance, how the trader's past decisions actually scored against
simply holding Bitcoin, your own past policies with what happened next, the macro tape, and today's research
brief so you know what kind of ideas the limits will be applied to.

How to think about it:
- Start from the account, not the ambition. On a small balance the fastest way to never reach the target is a
  drawdown that leaves too little capital to compound. Most of the distance to the target will come from
  deposits, not from trading a hundred dollars, so protecting the base matters more than maximising a session.
- Read positioning, not just price. Fear & Greed at an extreme, funding rates well above normal and falling
  stablecoin supply all say the same thing: the move is crowded and an unwind will be violent. Size down into
  that, regardless of how good the research brief looks.
- Let evidence move the dial. Tighten after losses, an unproven record, or a hostile tape (rising real yields,
  falling ETF flows, a stretched market where everything is at its highs). Loosen when the record is genuinely
  positive, conditions are calm, and the research brief has a high-conviction idea worth concentrating in.
- A small sample is not a record. A handful of trades, none of them closed, is weak evidence — do not read a
  winning streak of three as skill.
- Trading costs bound how often trading can pay. Every round trip costs roughly 1.9% in spread on these pairs,
  so a high trade count only makes sense if each idea is expected to clear that comfortably. Fewer, better sized
  orders beat many small ones.
- Concentration is the real risk on an account this size, and it hides at the theme level. The per-pair cap does
  not stop four names in the same trade becoming the portfolio: after one recent buy, AAVE + UNI + ONDO + LINK
  were about 78% of the account with every pair inside its own cap. max_theme_pct governs that — pairs sharing
  the research desk's theme label count as one bet, and a new buy is cut to the room left in its cluster.
- Volatility scaling is your main tool for adapting size to the market rather than to your mood: orders in a
  pair whose daily swing exceeds your target shrink in proportion. A lower target means smaller orders in hot
  pairs; 0 disables scaling entirely.
- The cooldown governs whether the trader can add to a pair it already bought. Short cooldowns allow pyramiding
  into a winner and also churn; long ones force patience.
- Explain each number against the target and the evidence. "Because the market looks good" is not a reason.
  Name what would make you set it differently next session.

Set every field. Be specific and be able to defend it: your policy is logged and you will be shown it next
session beside what the account did afterwards."""

POLICY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["per_order_pct", "max_position_pct", "max_trades_today", "vol_target_pct",
                 "cooldown_hours", "max_spread_pct", "max_theme_pct", "rationale", "stance"],
    "properties": {
        "per_order_pct": {"type": "number", "minimum": 0.01, "maximum": 1.0,
                          "description": "Largest single order as a fraction of account value, e.g. 0.2 for 20%."},
        "max_position_pct": {"type": "number", "minimum": 0.05, "maximum": 1.0,
                             "description": "Largest holding in one pair as a fraction of account value."},
        "max_trades_today": {"type": "integer", "minimum": 0, "maximum": 20,
                             "description": "Orders allowed today in total, including any already placed."},
        "vol_target_pct": {"type": "number", "minimum": 0.0, "maximum": 1.0,
                           "description": "Daily volatility above which orders shrink proportionally; 0 disables."},
        "cooldown_hours": {"type": "number", "minimum": 0.0, "maximum": 168.0,
                           "description": "Hours before the same pair may be bought again; 0 disables."},
        "max_spread_pct": {"type": "number", "minimum": 0.005, "maximum": 0.10,
                           "description": "Widest bid/ask spread the trader may pay, e.g. 0.03 for 3%."},
        "max_theme_pct": {"type": "number", "minimum": 0.1, "maximum": 1.0,
                          "description": "Largest share of the account in one correlated cluster of pairs "
                                         "(pairs sharing the research desk's theme label), e.g. 0.5 for 50%."},
        "stance": {"type": "string", "enum": ["defensive", "neutral", "aggressive"],
                   "description": "One word for the posture these numbers express."},
        "rationale": {"type": "string",
                      "description": "Why these numbers, against the growth target and the evidence, and what "
                                     "would change them next session."},
    },
}


def _prompt(account_value: float, brief: dict, trades_today: int, instruction: str = "") -> str:
    operator = ([f"OPERATOR INSTRUCTION FOR THIS SESSION, it overrides your own judgement on the numbers it "
                 f"names: {instruction}"] if instruction else [])
    return "\n".join(operator + [
        f"Time: {dt.datetime.now().astimezone().isoformat(timespec='minutes')}",
        (f"Account value at the start of today: ${account_value:,.2f}. Call get_portfolio for the live figure "
         "before deciding — it is the base every percentage you set applies to."
         if account_value else
         "No opening account value recorded yet today: call get_portfolio for the live figure before deciding."),
        f"Orders already placed today: {trades_today}. Your max_trades_today covers the whole day, so a value at "
        f"or below {trades_today} leaves the trader none.",
        "",
        risk_policy.as_prompt_section(account_value),
        "",
        "Open positions:",
        scorecard.position_review(),
        "",
        "How the trader's decisions have actually scored:",
        scorecard.as_prompt_section(),
        "",
        "Macro tape (live):",
        macro_feed.as_prompt_section(),
        "",
        "Crypto flows and positioning (live):",
        fundamentals.as_prompt_section(),
        "",
        "Today's research brief (what the limits will be applied to):",
        research_agent.as_prompt_section(brief) if brief else "No brief this session.",
        "",
        lessons.as_prompt_section("risk"),
        "",
        "Set this session's risk policy.",
    ])


def _deny(reason: str) -> dict:
    return {"hookSpecificOutput": {"hookEventName": "PreToolUse", "permissionDecision": "deny",
                                   "permissionDecisionReason": reason}}


def allowed_tools() -> list:
    return sorted(set(ACCOUNT_READS) | set(rh_tools.HARNESS_TOOLS) | {research_agent.STRUCTURED_OUTPUT_TOOL})


def disallowed_tools() -> list:
    return (list(rh_tools.BUILTIN_BLOCKED) + list(rh_tools.NEVER_EXPOSED)
            + list(rh_tools.PLACE.values()) + list(rh_tools.SIMULATE.values()))


async def run(account_value: float, brief: dict, trades_today: int = 0, instruction: str = ""):
    """The run's Limits. Falls back to the operator's fixed limits on any failure."""
    if not settings.risk_agent_enabled:
        return risk_policy.FALLBACK

    allowed = set(allowed_tools())

    async def pre_tool_use(input_data, tool_use_id, context):
        name = input_data["tool_name"]
        if name in allowed:
            return {}
        logger.warning("Risk agent blocked from %s", name)
        return _deny(f"{name} is not available to the risk agent")

    options = ClaudeAgentOptions(
        cwd=PROJECT_DIR,
        setting_sources=["project"],
        model=settings.claude_model,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=sorted(allowed),
        disallowed_tools=disallowed_tools(),
        permission_mode="dontAsk",
        hooks={"PreToolUse": [HookMatcher(hooks=[pre_tool_use])]},
        env={"MAX_MCP_OUTPUT_TOKENS": "50000"},
        max_turns=settings.risk_max_turns,
        max_budget_usd=settings.risk_budget_usd,
        thinking={"type": "adaptive"},
        effort=settings.risk_effort,
        output_format={"type": "json_schema", "schema": POLICY_SCHEMA},
    )

    raw, text = None, ""
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(_prompt(account_value, brief, trades_today, instruction))
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    raw = message.structured_output
                    text = (message.result or "").strip()
                    logger.info("Risk agent ended: %s after %d turns, cost $%.4f",
                                message.subtype, message.num_turns, message.total_cost_usd or 0.0)
    except Exception:
        logger.exception("Risk agent failed; falling back to fixed limits.")
        return risk_policy.FALLBACK

    import agent  # local import: agent imports this module
    if agent.usage_limited(text):
        logger.error("Risk agent stopped by the Claude usage limit; falling back to fixed limits.")
        return risk_policy.FALLBACK

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        logger.warning("Risk agent returned no structured policy; falling back to fixed limits. Text: %s", text[:200])
        return risk_policy.FALLBACK

    clean, problems = risk_policy.validate(raw)
    if problems:
        logger.error("Risk policy rejected (%s); falling back to fixed limits.", "; ".join(problems))
        return risk_policy.FALLBACK

    clean["stance"] = str(raw.get("stance") or "")[:20]
    risk_policy.save(clean, account_value)
    limits = risk_policy.effective(clean)
    logger.info("Risk policy (%s): order %.0f%%, position %.0f%%, theme %.0f%%, trades %d, vol %.1f%%, "
                "cooldown %.0fh, spread %.1f%%",
                clean["stance"] or "set", limits.per_order_pct * 100, limits.max_position_pct * 100,
                limits.max_theme_pct * 100, limits.max_trades_today, limits.vol_target_pct * 100,
                limits.cooldown_hours, limits.max_spread_pct * 100)
    return limits


def _holdings_by_theme() -> str:
    import positions_store
    labels = {}
    for symbol, (quantity, _sellable) in positions_store.load()[0].items():
        if quantity > 0:
            labels.setdefault(themes.theme_of(symbol) or "unlabelled", []).append(symbol)
    return "; ".join(f"{theme}: {', '.join(sorted(names))}" for theme, names in sorted(labels.items())) or "nothing"


def as_prompt_section(limits) -> str:
    """What the trader is told about the limits it must work inside."""
    policy = risk_policy.today() if limits.agent_set else {}
    lines = [
        ("Set for this session by the risk agent, a separate agent you cannot instruct or override."
         if limits.agent_set else
         "The risk agent did not set a policy this session, so these are the operator's fixed fallback limits."),
        f"  Largest order: {limits.per_order_pct:.0%} of account value (your discretion under it)",
        f"  Most in one pair: {limits.max_position_pct:.0%} of the portfolio",
        f"  Orders allowed today: {limits.max_trades_today}",
        f"  Volatility target: {limits.vol_target_pct:.1%} daily — orders in hotter pairs shrink in proportion"
        if limits.vol_target_pct else "  Volatility scaling: off",
        f"  Cooldown before re-buying a pair: {limits.cooldown_hours:.0f}h" if limits.cooldown_hours
        else "  Cooldown: off — you may add to a pair you already hold",
        f"  Widest spread you may pay: {limits.max_spread_pct:.1%}",
        f"  Most in one theme: {limits.max_theme_pct:.0%} of the account across pairs sharing a research theme "
        f"(you hold — {_holdings_by_theme()})",
        f"  Fixed by the operator: {settings.max_daily_loss_pct:.0%} daily loss breaker",
    ]
    if policy.get("rationale"):
        lines.append(f"  Risk agent's reasoning ({policy.get('stance', 'set')}): {policy['rationale'][:600]}")
    lines.append("These are enforced in code. Trade inside them; if you think they are wrong, say so in your "
                 "summary rather than working around them.")
    return "\n".join(lines)
