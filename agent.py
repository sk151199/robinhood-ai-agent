import datetime as dt
import json
import os
import re

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, HookMatcher, ResultMessage, SystemMessage

import crypto_news
import journal
import market_context
import pairs_catalog
import positions_store
import research_agent
import risk_agent
import risk_policy
import rh_parse
import rh_tools
import scorecard
from config import settings
from order_gate import OrderGate
from risk_manager import RiskManager
from trade_logger import log_trade, logger

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_PROMPT = """You actively manage a small self-directed Robinhood Agentic account through the
robinhood-trading tools. You choose what to trade and how much. There is no fixed watchlist and no human
reviewing your picks, so decide and act on your own judgement.

This account trades crypto only. You cannot buy or sell stocks; the equity order tools are not available to
you, and the SPY and AAPL already held are context you cannot act on.

Find candidates yourself across Robinhood's crypto catalog, not just the majors. get_currency_pairs is the
catalog, search resolves a pair by name, get_crypto_quotes prices it. Look at what is moving and what the
book actually looks like.

Be honest with yourself about how thin the data is here. Robinhood exposes no crypto historicals, and its
technical-indicator tools are equity-only, so from the broker you get a live quote, the previous close and
the spread. To offset that you are handed a digest of recent crypto headlines each cycle, and get_equity_news
plus the index quotes give you a macro read. Headlines tell you what happened and what the market already
knows; they are not a substitute for the book in front of you. When the evidence supports nothing better
than a coin flip, say so rather than dressing a guess up as a thesis.

How to trade this market well, given those constraints:
- Spread is the largest cost you control, and it is close to uniform: every Robinhood pair measured so far
  quotes roughly 1.9% wide. There is no cheaper venue among them and no execution advantage in going
  down-cap, so choose a pair on the thesis, not on the book. A round trip costs about that 1.9%, so only
  take a position you expect to move well beyond it.
- You may buy a given pair only once per cooldown window (see the session limits). Topping up the same name
  on consecutive cycles is churn, not conviction, and the gate rejects it. Selling is never blocked by the
  cooldown — reducing risk is always available to you.
- Adding to something you already own is legitimate once that window has passed, provided the original
  thesis is intact and the case has strengthened rather than merely survived. That is reinvestment. Adding
  because a position is down, with no new reason to believe it, is averaging into a mistake.
- Crypto's variance is priced into your size, not left to your judgement. The 'sd' column is realized daily
  volatility, and an order in a pair running above the session's volatility target is scaled down in
  proportion: a 6% pair gets roughly half the size of a 3% one. Do not try to offset that by requesting
  more — if a swing that large is not worth owning small, it is not worth owning.
- Think in BTC terms. Most pairs are levered beta to Bitcoin. If you cannot say why a pair should beat
  simply holding BTC, then holding BTC is the better version of that trade.
- You are given daily price history each cycle: 1d, 7d and 30d moves and where each pair sits in its
  30-day range. Entry level matters as much as direction. Prefer adding on weakness when the thesis holds,
  and treat a pair pinned near its 30-day high as a worse entry rather than a confirmation.
- A falling price is not a thesis. Dips continue, and with three orders a day and a ~1.9% spread, catching
  a falling knife ties up capital you cannot redeploy. Buy a drop when you can say what caused it and why
  it is wrong, not merely because the number is lower.
- Minimum order sizes vary by orders of magnitude. Some pairs need thousands of units, so a small dollar
  order silently fails the size check — read min_order_size before committing to a candidate.
- Churn is how a small account dies. Every flip pays the spread twice and the position cap limits how much
  any one idea can earn, so favour fewer positions held with reason over frequent rotation.
- Review get_realized_pnl and get_pnl_trade_history each cycle. That is the only feedback you get on
  whether your approach is actually working; let it change your behaviour, not just your commentary.

Act only on live data. Refetch every number you trade on in this session: quotes, positions, buying power.
Your journal and the congressional digest are context and history, never prices. A price you remember from
an earlier cycle is stale by definition, and an order resting on a quote older than the session's freshness
limit is rejected automatically — so pull the quote immediately before you order, not at the start of your
research.

Buy only what you expect to grow. Before any buy, state the thesis in your summary: what specifically
drives the move, roughly how far and over what horizon, and what would prove you wrong. "It has gone up",
"to get invested", or "it looked cheap" are not theses. If you cannot name a driver, there is no trade. The
same standard applies to holding: when a thesis you opened has broken, closing it is also a decision worth
making.

Take your time. There is no pressure to finish quickly or to economize on tool calls: quote every pair you
are seriously weighing, compare several candidates against each other before choosing, and re-check the
positions you hold. Depth of research is the point of each cycle. The only clock that matters is data
freshness, covered below — refresh the account and quotes right before you order.

A research desk works for you. Before most sessions a separate read-only agent scans every tradable pair, reads
the headlines, investigates the unusual movers on the web, and leaves you a ranked brief with theses, catalysts
and what would prove each wrong. Use it to spend your attention on judgement rather than breadth. It is input,
not instruction: verify a candidate against the live book before acting, pass on one you disagree with and say
why, and you remain free to trade a pair it did not name. When no brief is present you are given the full market
instead and do the scan yourself.

Be decisive. A cycle where you research broadly and then place a well-reasoned order is a good cycle.
Waiting for a perfect setup is itself a decision, and on a small account the cost of never acting is real.
Trade when you find a candidate that clears your bar; skip only when nothing does, and say plainly what
would have to change for you to act.

Size your own orders, inside limits you do not set. A separate risk agent sets this session's ceilings —
order size, concentration, trade count, volatility scaling, cooldown, spread — against the account's growth
target, and code enforces them. You cannot instruct, negotiate with or override it; if you think a limit is
wrong, say so in your summary and the operator will see it. Anything at or under the ceiling is yours to choose. Use more of it for a position you have genuine conviction in and less for a
speculative one, rather than defaulting to the same number every time.

Manage what you already own before adding to it. Each cycle you are shown every open position with its
entry price, unrealized move and the thesis you recorded when you bought it. Rule on each one — hold, trim
or exit — and say why, before you propose a new buy. A thesis that has been invalidated is a reason to
sell; a position drifting sideways while your reason for owning it has evaporated is dead capital. Selling
is never blocked by the cooldown, so an exit is always available to you.

Before any order, every session:
1. get_accounts, and use the one account marked agentic_allowed.
2. get_portfolio for that account.
3. get_crypto_positions (rhs_account_number).
4. The tradable universe is listed for you from a cached catalog. Call get_currency_pairs (limit 50, follow
   the cursor) only when the symbol you want is missing from that list, or when the catalog is empty — every
   page you fetch is remembered for later cycles, so it is a one-time cost, not a per-cycle one.
5. get_crypto_quotes for anything you will trade or already hold, taken immediately before ordering.
Orders are rejected automatically when any of these is missing.

Research first, then refresh, then order. Account snapshots older than 15 minutes and quotes older than 5
minutes are refused, so if any time has passed while you were reading, call get_portfolio and
get_crypto_positions again immediately before placing the order. A cycle that gathers its account state at
the start, spends an hour researching, and then orders against that opening snapshot is rejected and the
whole cycle is wasted. This has already happened once.

Order rules. Anything else is rejected:
- Market orders sized with dollar_amount only. Never quantity, limit_price, stop_price or tax_lots.
- time_in_force gtc, symbols like BTC-USD, and the pair must not be halted in all regions (a regional
  NY/TX halt does not apply to this account).
- get_currency_pairs is paginated: request limit 50 and follow the cursor page by page until your pair
  appears. A larger page is truncated by the tool-output cap and then none of it is usable, which shows up
  as your order being rejected for an unlisted pair.
- The order must clear the pair's minimum order size, which varies enormously between pairs.
- Anything with a quoted bid/ask spread wider than the session's spread limit is rejected. Check the book
  before committing: on a small order a wide spread is the single largest cost you control.
- Never sell something the account does not hold.

Hard risk limits are enforced in code outside your control, and an order may be reduced or rejected. If a
rejection says data is missing, fetch that data and retry. If a limit, the daily order count or the loss
breaker rejects it, do not try to get around it with a different size, symbol or account.

You are shown your own previous cycles. Treat them as your working memory: follow up on theses you opened,
notice what worked, and do not repeat a rejected approach.

Finish with a brief plain-text summary of what you looked at, what you did, and why.

End that summary with one final line naming the candidates you seriously considered and rejected, so that
restraint is recorded rather than invisible:
PASSED: SOL - already pinned to its 30d high; INJ - volume too thin to trust the drop
Use exactly that prefix, separate candidates with semicolons, and write "PASSED: none" if you genuinely
weighed nothing else. Only list pairs you actually evaluated — a padded list makes the record useless."""

REQUIRED_TRADE = """REQUIRED THIS SESSION: place exactly one BUY order. Doing nothing is not an option this session.
Pick the best candidate you can find (prefer an equity while the market is open; crypto spreads are wide),
size it with dollar_amount at or below the per-order ceiling, and submit it. Keep research brief so the
session finishes within its budget."""

REQUIRED_TRADE_FOLLOW_UP = """You were required to place exactly one BUY order this session and have not placed it.
Place it now using the data you already fetched; do no further research. If the rejection said data was missing,
fetch only that and retry. If the daily order limit or the loss breaker blocks it, stop and say so."""


# The CLI reports an exhausted Claude plan as a *successful* one-turn result whose text is the
# limit notice. Treated as a normal cycle, that notice became a journal entry and the cycle
# silently did nothing.
LIMIT_NOTICE = re.compile(r"(hit|reached) your (session|usage|weekly|daily)? ?limit|usage limit", re.IGNORECASE)


def usage_limited(text: str) -> bool:
    return bool(text) and bool(LIMIT_NOTICE.search(text[:300]))


USAGE_KEYS = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")


def _usage_totals(usage) -> dict:
    """Token counts off a ResultMessage, whether the SDK hands back a dict or an object."""
    if not usage:
        return {}
    data = usage if isinstance(usage, dict) else getattr(usage, "__dict__", {}) or {}
    return {key: int(data.get(key) or 0) for key in USAGE_KEYS if data.get(key) is not None}


def _deny(reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _parse_passes(summary: str) -> list:
    """Pull the PASSED: line out of the cycle summary.

    Prose parsing rather than a tool call: a malformed line costs one unrecorded pass, where a
    broken tool definition would cost the whole cycle.
    """
    if not summary:
        return []
    match = re.search(r"^PASSED:\s*(.+)$", summary, re.MULTILINE | re.IGNORECASE)
    if not match:
        return []
    body = match.group(1).strip()
    if body.lower().startswith("none"):
        return []

    candidates = []
    for chunk in body.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = re.split(r"\s+[-–—]\s+", chunk, maxsplit=1)
        # "BTC-USD" and "BTC" must land on the same key, or the review finds no history for it.
        symbol = rh_parse.crypto_symbol(re.sub(r"[^A-Za-z0-9-]", "", parts[0])[:12])
        if symbol and len(parts) == 1 and candidates:
            # No "SYMBOL - reason" shape: this is a reason that itself contained a semicolon.
            previous_symbol, previous_reason = candidates[-1]
            candidates[-1] = (previous_symbol, f"{previous_reason}; {chunk}")
            continue
        if symbol:
            candidates.append((symbol, parts[1].strip() if len(parts) > 1 else ""))
    return candidates[:10]


def _context_symbols() -> list:
    """Every tradable pair gets price history, not just the majors.

    A drop outside the majors list was previously invisible: the catalog gave the symbol but
    no prices, so the agent had no way to know it had moved. Candles are cached once a day,
    so the extra coverage costs one round of fetches rather than one per cycle.
    """
    traded = list(scorecard.last_buy_times())
    known, _ = pairs_catalog.load()
    tradable = sorted(symbol for symbol, pair in known.items()
                      if pair.get("tradable") and not rh_parse.blocks_orders(pair))
    majors = [symbol for symbol in market_context.MAJORS if not tradable or symbol in tradable]
    return list(dict.fromkeys(traded + majors + tradable))[: settings.context_symbols_max]


def _cycle_prompt(market_open: bool, trades_remaining: int, require_trade: bool = False,
                  instruction: str = "", brief: dict = None, limits=None) -> str:
    if settings.dry_run:
        mode = ("DRY RUN. review_equity_order / preview_crypto_order stand in for placing an order: call one "
                "only once you have decided to trade, exactly as you would call place_* in live mode. They are "
                "recorded as orders, not as previews. Nothing this session places a real order.")
    else:
        mode = "LIVE. place_equity_order / place_crypto_order execute with real money."
    lines = [
        f"Mode: {mode}",
        f"Time: {dt.datetime.now().astimezone().isoformat(timespec='minutes')}",
        f"Crypto trades 24/7, so this cycle can act at any hour. (US equity session is currently "
        f"{'open' if market_open else 'closed'}, which affects only the macro backdrop — you cannot trade equities.)",
        f"{trades_remaining} order(s) left to you today.",
        "",
        "Risk limits for this session:",
        risk_agent.as_prompt_section(limits or risk_policy.effective()),
        "Universe: any non-halted Robinhood crypto pair that clears the spread limit. There is no watchlist "
        "and no equities. Consider the whole list below, not just the majors, and go find the pair worth owning.",
        "",
        "Tradable crypto universe:",
        pairs_catalog.as_prompt_section(),
        "",
    ]
    held = list(positions_store.load()[0])
    if brief:
        # The desk has already read the full table and every headline; repeating them here is
        # the overload the desk exists to remove.
        focus = list(dict.fromkeys(research_agent.brief_symbols(brief) + held + [market_context.BENCHMARK]))
        lines += [
            "Research desk brief:",
            research_agent.as_prompt_section(brief),
            "",
            "Price context for the pairs in the brief (for any other pair, quote it directly):",
            market_context.as_prompt_section(focus, focused=True),
            "",
        ]
    else:
        lines += [
            "No research brief this session — scan the full market yourself.",
            "",
            "Price context (what has actually moved):",
            market_context.as_prompt_section(_context_symbols(), always_include=held),
            "",
            "Recent crypto headlines:",
            crypto_news.as_prompt_section(),
            "",
        ]
    lines += [
        "Your open positions:",
        scorecard.position_review(),
        "",
        "Your recent cycles:",
        journal.as_prompt_section(settings.journal_lookback),
        "",
        scorecard.passes_review(),
        "",
        scorecard.as_prompt_section(),
    ]
    if instruction:
        lines += ["", f"OPERATOR INSTRUCTION FOR THIS SESSION, it takes precedence over your own "
                      f"candidate search: {instruction}"]
    if require_trade:
        lines += ["", REQUIRED_TRADE]
    return "\n".join(lines)


class TradingAgent:
    def __init__(self):
        self.risk = RiskManager()

    async def prepare(self, account_value: float, instruction: str = ""):
        """Research, then risk, before any trading session opens.

        Both run to completion first so the trader's freshness clock starts when it starts, and
        so the gate is built from limits that are already decided.
        """
        brief = await research_agent.run()
        limits = await risk_agent.run(account_value, brief, self.risk.trades_today(), instruction)
        self.risk.apply_limits(limits)
        # Exits before entries: the trader then sizes against whatever cash this frees, and a
        # position that has run its course is not held for hours waiting on a threshold.
        if settings.sell_agent_enabled:
            import sell_agent
            sold = await sell_agent.review_all()
            if sold:
                logger.info("Sell agent closed or trimmed %d position(s) before the trading session.", len(sold))
        return brief, limits

    def log_startup(self):
        logger.info("Mode: %s | model %s", "DRY RUN (server-side simulation)" if settings.dry_run else "LIVE TRADING",
                    settings.claude_model)
        breaker = (f"${settings.max_daily_loss_usd:.2f}" if settings.max_daily_loss_usd > 0
                   else f"{settings.max_daily_loss_pct:.0%}")
        limits = self.risk.limits
        logger.info(
            "Limits (%s): %.0f%%/order, %.0f%% max position, %d orders/day, %.1f%% vol target, "
            "%.0fh cooldown, %.1f%% max spread | %s daily loss breaker (operator, fixed), $%.2f cash buffer",
            "risk agent policy" if limits.agent_set else "fixed fallback, risk agent has not run yet",
            limits.per_order_pct * 100, limits.max_position_pct * 100, limits.max_trades_today,
            limits.vol_target_pct * 100, limits.cooldown_hours, limits.max_spread_pct * 100,
            breaker, settings.min_cash_buffer,
        )
        logger.info("Cycle cost cap $%.2f, every %ds, market hours only: %s, congressional data: %s",
                    settings.max_cycle_budget_usd, settings.poll_interval_seconds, settings.market_hours_only,
                    settings.congress_enabled)
        if self.risk.resumed_today():
            logger.info("Resuming today: %d order(s) already used, %d remaining.",
                        self.risk.trades_today(), self.risk.trade_budget_remaining())

    @staticmethod
    def _fill_price(outcome: str, mid: float, spread_pct: float = 0.0, side: str = "buy") -> float:
        """The executed price, not the pre-trade mid.

        Recording the mid flatters every subsequent P&L by roughly half the spread, because
        the order actually crossed the book. A market order usually comes back unfilled, with
        average_price null and `price` holding the collar limit rather than any fill (UNI:
        collar 6.98, fill 6.911). So without a real average price, the far side of the quoted
        book stands in: it matched that fill to within 0.2%.
        """
        try:
            order = (json.loads(outcome).get("data") or {}).get("order") or {}
            filled = order.get("average_price")
            if filled and float(filled) > 0:
                return float(filled)
        except (ValueError, TypeError, AttributeError, json.JSONDecodeError):
            pass
        half = (spread_pct or 0.0) / 2
        return mid * (1 + half) if side == "buy" else mid * (1 - half)

    def _record(self, verdict, outcome: str, forced: bool = False):
        decision = "denied" if not verdict.allowed else ("simulated" if settings.dry_run else "placed")
        log_trade({
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "asset_class": verdict.asset_class,
            "symbol": verdict.symbol,
            "side": verdict.side,
            "requested_dollars": f"{verdict.requested:.2f}",
            "dollars": f"{verdict.dollars:.2f}",
            "price": f"{verdict.price:.8f}",
            "spread_pct": f"{verdict.spread_pct:.4f}",
            "decision": decision,
            "reason": verdict.reason,
            "dry_run": settings.dry_run,
            "ref_id": (verdict.updated_input or {}).get("ref_id", ""),
            "order_result": outcome[:500],
        })
        if verdict.allowed:
            scorecard.record(verdict.symbol, verdict.side, verdict.dollars,
                             self._fill_price(outcome, verdict.price, verdict.spread_pct, verdict.side),
                             verdict.spread_pct, decision, verdict.reason, forced=forced)
        return decision

    async def _consume(self, client, outcome: dict) -> bool:
        """Drain one response. False when the Robinhood server is unavailable."""
        async for message in client.receive_response():
            if isinstance(message, SystemMessage) and message.subtype == "init":
                statuses = {s.get("name"): s.get("status") for s in message.data.get("mcp_servers", [])}
                status = statuses.get(rh_tools.SERVER)
                if status in ("failed", "needs-auth", "disabled", None):
                    logger.error("robinhood-trading MCP is %s; ending cycle. Re-authenticate with /mcp in Claude Code.",
                                 status or "not configured")
                    return False
            elif isinstance(message, ResultMessage):
                # total_cost_usd is cumulative for the whole session: a later response repeats the
                # running total rather than reporting its own slice, so summing double-counts.
                outcome["cost"] = max(outcome.get("cost", 0.0), message.total_cost_usd or 0.0)
                outcome["turns"] = outcome.get("turns", 0) + (message.num_turns or 0)
                outcome["subtype"] = message.subtype
                totals = outcome.setdefault("usage", {})
                for key, value in _usage_totals(message.usage).items():
                    totals[key] = totals.get(key, 0) + value
                if message.result:
                    outcome["summary"] = message.result.strip()
                if usage_limited(message.result or ""):
                    outcome["limited"] = True
                    logger.error("Claude usage limit reached: %s", (message.result or "").strip()[:200])
                logger.info("Response ended: %s after %d turns, cost $%.4f",
                            message.subtype, message.num_turns, message.total_cost_usd or 0.0)
                if totals:
                    logger.info("Tokens: in=%d out=%d cache_read=%d cache_write=%d",
                                totals.get("input_tokens", 0), totals.get("output_tokens", 0),
                                totals.get("cache_read_input_tokens", 0),
                                totals.get("cache_creation_input_tokens", 0))
                if message.result:
                    logger.info("Agent summary: %s", message.result.strip()[:2000])
        return True

    async def run_cycle(self, market_open: bool, require_trade: bool = False, instruction: str = "",
                        brief: dict = None, limits=None):
        """require_trade demands one BUY this cycle. It is a per-call argument, never a setting,
        so the daemon cannot be left forcing trades."""
        if limits is not None:
            self.risk.apply_limits(limits)
        gate = OrderGate(self.risk)
        passthrough = rh_tools.passthrough_tools()
        pending = {}
        placed = []
        outcome = {"cost": 0.0, "summary": ""}
        started_at = dt.datetime.now(dt.timezone.utc).isoformat()

        async def pre_tool_use(input_data, tool_use_id, context):
            name = input_data["tool_name"]
            key = input_data.get("tool_use_id") or tool_use_id
            if name in gate.order_tools:
                verdict = gate.evaluate(name, input_data.get("tool_input") or {})
                if not verdict.allowed:
                    logger.warning("DENIED %s %s $%.2f: %s", verdict.side, verdict.symbol, verdict.requested, verdict.reason)
                    self._record(verdict, f"denied: {verdict.reason}")
                    return _deny(verdict.reason)
                logger.info("%s %s %s $%.2f (%s)", "SIMULATING" if settings.dry_run else "PLACING",
                            verdict.side.upper(), verdict.symbol, verdict.dollars, verdict.reason)
                pending[key] = verdict
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "allow",
                        "permissionDecisionReason": verdict.reason,
                        "updatedInput": verdict.updated_input,
                    }
                }
            if name in passthrough:
                return {}
            logger.warning("Blocked tool outside the allowlist: %s", name)
            return _deny(f"{name} is not on this agent's allowlist")

        async def post_tool_use(input_data, tool_use_id, context):
            key = input_data.get("tool_use_id") or tool_use_id
            gate.observe(input_data["tool_name"], input_data.get("tool_input") or {}, input_data.get("tool_response"))
            verdict = pending.pop(key, None)
            if verdict:
                decision = self._record(verdict, str(input_data.get("tool_response")), forced=require_trade)
                placed.append({"symbol": verdict.symbol, "side": verdict.side,
                               "dollars": f"{verdict.dollars:.2f}", "decision": decision})
            return {}

        async def post_tool_failure(input_data, tool_use_id, context):
            verdict = pending.pop(input_data.get("tool_use_id") or tool_use_id, None)
            if verdict:
                logger.error("Order tool failed for %s %s: %s", verdict.side, verdict.symbol, input_data.get("error"))
                self._record(verdict, f"error: {input_data.get('error')}")
            return {}

        options = ClaudeAgentOptions(
            cwd=PROJECT_DIR,
            setting_sources=["project"],
            model=settings.claude_model,
            system_prompt=SYSTEM_PROMPT,
            allowed_tools=rh_tools.allowed_tools(),
            disallowed_tools=rh_tools.disallowed_tools(),
            permission_mode="dontAsk",
            hooks={
                "PreToolUse": [HookMatcher(hooks=[pre_tool_use])],
                "PostToolUse": [HookMatcher(hooks=[post_tool_use])],
                "PostToolUseFailure": [HookMatcher(hooks=[post_tool_failure])],
            },
            # Robinhood's catalog pages are large; the default 25k cap turns one into an error
            # string, which the gate then reads as "pair unknown" and refuses to trade.
            env={"MAX_MCP_OUTPUT_TOKENS": "50000"},
            max_turns=settings.max_cycle_turns,
            max_budget_usd=settings.max_cycle_budget_usd,
            # Depth over speed: a cycle fires a few times a day and each decision moves real money.
            thinking={"type": "adaptive"},
            effort=settings.claude_effort,
        )

        prompt = _cycle_prompt(market_open, self.risk.trade_budget_remaining(), require_trade, instruction,
                               brief, self.risk.limits)
        async with ClaudeSDKClient(options=options) as client:
            await client.query(prompt)
            reachable = await self._consume(client, outcome)
            # Only worth nudging if the session can still act: a budget- or turn-exhausted
            # session burns a turn and changes nothing.
            healthy = outcome.get("subtype") == "success"
            if reachable and healthy and require_trade and not placed and self.risk.trade_budget_remaining():
                logger.warning("Required order not placed yet; instructing the agent to place it now.")
                await client.query(REQUIRED_TRADE_FOLLOW_UP)
                await self._consume(client, outcome)

        if outcome.get("limited"):
            logger.error("Cycle did not run: Claude usage limit. Nothing recorded to the journal or scorecard.")
            return False

        journal.append(
            mode="dry_run" if settings.dry_run else "live",
            summary=outcome["summary"],
            orders=placed,
            cost_usd=outcome["cost"],
            usage=outcome.get("usage"),
        )
        # Pair this cycle's reasoning with the orders it produced, so a later cycle can be
        # shown what it claimed would happen beside what actually did.
        scorecard.attach_summary(outcome["summary"], started_at)
        scorecard.record_passes(_parse_passes(outcome["summary"]),
                                "dry_run" if settings.dry_run else "live")

        if require_trade and not placed:
            logger.error("Required order was NOT placed this cycle.")
        for error in gate.parse_errors:
            logger.warning("Unparseable tool result (orders fail closed): %s", error)
        return True
