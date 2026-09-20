"""Deterministic gate between the model and every order tool.

State comes only from Robinhood's own tool results, observed in PostToolUse — never from
numbers the model reports. Anything not yet observed this cycle is unknown, and unknown
means no order.

Two guards replace the watchlist now that the agent picks its own symbols:
  * Robinhood must say the symbol is tradable for this account, and its quoted spread must
    be inside MAX_SPREAD_PCT.
  * Every figure behind an order must be fresh. A quote or an account snapshot older than
    its limit is refused, so an order can never rest on a remembered or long-stale price.
"""

import datetime as dt
import json
import math
import time
from dataclasses import dataclass

import market_context
import marks_store
import pairs_catalog
import portfolio_store
import positions_store
import rh_parse
import scorecard
import themes
from config import settings
from rh_tools import PREFIX, gated_order_tools

MIN_ORDER_DOLLARS = 1.0
FORBIDDEN_FIELDS = ("quantity", "limit_price", "stop_price", "tax_lots")


@dataclass
class Verdict:
    allowed: bool
    reason: str
    asset_class: str = ""
    symbol: str = ""
    side: str = ""
    requested: float = 0.0
    dollars: float = 0.0
    # Captured so a decision can be scored later against what the price actually did.
    price: float = 0.0
    spread_pct: float = 0.0
    updated_input: dict = None


def _cents_down(value: float) -> float:
    return math.floor(round(max(0.0, value) * 100, 6)) / 100


class OrderGate:
    def __init__(self, risk, clock=time.monotonic, sell_only=False, exempt_daily_limit=False):
        # The sell agent gets a gate that cannot buy: a session woken to protect a position must
        # not be able to open one, whatever it concludes.
        self.sell_only = sell_only
        # A protective exit is not churn, so the daily order count neither blocks it nor counts
        # it. Its own cap lives in exit_watch, which decides when to wake that agent at all.
        self.exempt_daily_limit = exempt_daily_limit
        self.risk = risk
        # One source of limits per run, so the gate and the risk manager cannot disagree.
        self.limits = risk.limits
        self.clock = clock
        self.order_tools = gated_order_tools()
        self.account = None
        self.portfolio = None
        self.held = {"stock": None, "crypto": None}
        self.prices = {}
        self.tradability = {}
        # Seeded from the persisted catalog so a pair the agent paged to in an earlier cycle
        # stays orderable without paying to page the whole catalog again.
        self.pairs = pairs_catalog.load()[0]
        self.seen_at = {}
        self.cycle_flow = {}
        self.bought = 0.0
        self.parse_errors = []
        # When each symbol was last bought, so the cooldown survives across cycles and
        # restarts rather than living only in this process.
        self.recent_buys = scorecard.last_buy_times()

    def _is_agentic(self, value, field: str) -> bool:
        return self.account is not None and str(value) == self.account[field]

    def _age(self, key) -> float:
        stamped = self.seen_at.get(key)
        return float("inf") if stamped is None else self.clock() - stamped

    def _volatility(self, symbol: str):
        """Realized daily volatility, or None when there is no history to measure.

        Priced from Coinbase, so equities return None and simply are not vol-scaled; their
        other caps still bind.
        """
        try:
            return market_context.stats(symbol).get("volatility_30d")
        except Exception:
            return None

    def _theme_room(self, asset: str, symbol: str, position_value: float):
        """(dollars still investable in this pair's cluster, cluster label, its current value).

        Cluster holdings are valued at the last live quote where the agent has one, else the
        last daily close — a valuation good enough for a concentration check, and it fails
        towards caution by counting a stale price rather than ignoring the holding.
        """
        portfolio_value = self.portfolio["total_value"]
        if not portfolio_value:
            return None, "", 0.0
        holdings = {code: quantity for code, (quantity, _sellable) in (self.held[asset] or {}).items() if quantity > 0}
        members = themes.cluster(symbol, holdings)
        theme = themes.theme_of(symbol) or "correlated"

        held_value = 0.0
        for code in members:
            if code == symbol:
                held_value += max(0.0, position_value)
                continue
            price = self.prices.get((asset, code), {}).get("price")
            if not price:
                price = (market_context.stats(code) or {}).get("price") or 0.0
            held_value += holdings.get(code, 0.0) * price + self.cycle_flow.get((asset, code), 0.0)
        return max(0.0, self.limits.max_theme_pct * portfolio_value - held_value), theme, held_value

    def _hours_since_buy(self, symbol: str):
        """Wall-clock hours since this symbol was last bought, or None if never."""
        last = self.recent_buys.get(symbol)
        if last is None:
            return None
        return (dt.datetime.now(dt.timezone.utc) - last).total_seconds() / 3600

    def observe(self, tool_name: str, tool_input: dict, raw):
        name = tool_name.removeprefix(PREFIX)
        try:
            if name == "get_accounts":
                self.account = rh_parse.agentic_account(raw)
            elif name == "get_portfolio" and self._is_agentic(tool_input.get("account_number"), "account_number"):
                self.portfolio = rh_parse.portfolio(raw)
                self.seen_at["portfolio"] = self.clock()
                self.risk.start_cycle(self.portfolio["total_value"])
                # Persist for code that runs outside a session, such as the opportunity watch
                # deciding whether a setup is worth waking the agents for.
                portfolio_store.save(self.portfolio["total_value"], self.portfolio["crypto_buying_power"])
            elif name == "get_equity_positions" and not tool_input.get("cursor"):
                if self._is_agentic(tool_input.get("account_number"), "account_number"):
                    self.held["stock"] = rh_parse.equity_positions(raw)
                    self.seen_at["positions:stock"] = self.clock()
            elif name == "get_crypto_positions" and not tool_input.get("cursor"):
                if self._is_agentic(tool_input.get("rhs_account_number"), "rhs_account_number"):
                    self.held["crypto"] = rh_parse.crypto_positions(raw)
                    self.seen_at["positions:crypto"] = self.clock()
                    # Persist so the next cycle can show entry vs current without a broker call.
                    positions_store.save(self.held["crypto"])
            elif name == "get_equity_quotes":
                for symbol, quote in rh_parse.equity_prices(raw).items():
                    self.prices[("stock", symbol)] = quote
                    self.seen_at[("stock", symbol)] = self.clock()
            elif name == "get_crypto_quotes":
                quotes = rh_parse.crypto_prices(raw)
                for symbol, quote in quotes.items():
                    self.prices[("crypto", symbol)] = quote
                    self.seen_at[("crypto", symbol)] = self.clock()
                # Persist the broker's own marks. They are the only price anchor that exists for
                # pairs no public venue lists, and mapping a ticker to the wrong asset is the
                # mistake that showed a delisted LIT series as a 40% crash.
                marks_store.remember(quotes)
            elif name == "get_equity_tradability" and self._is_agentic(
                tool_input.get("account_number"), "account_number"
            ):
                self.tradability.update(rh_parse.equity_tradability(raw))
            elif name == "get_currency_pairs":
                parsed = rh_parse.currency_pairs(raw)
                self.pairs.update(parsed)
                pairs_catalog.update(parsed)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            self.parse_errors.append(f"{name}: {error}")

    def evaluate(self, tool_name: str, tool_input: dict) -> Verdict:
        asset = self.order_tools.get(tool_name)
        if asset is None:
            return Verdict(False, "not an order tool enabled for this run")

        raw_symbol = str(tool_input.get("symbol", ""))
        symbol = raw_symbol.strip().upper() if asset == "stock" else rh_parse.crypto_symbol(raw_symbol)
        side = str(tool_input.get("side", "")).lower()
        verdict = Verdict(False, "", asset, symbol, side)

        def deny(reason: str) -> Verdict:
            verdict.reason = reason
            return verdict

        kind = "equity" if asset == "stock" else "crypto"
        account_field = "account_number" if asset == "stock" else "rhs_account_number"

        if self.account is None:
            return deny("agentic account unknown: call get_accounts first")
        if str(tool_input.get(account_field, "")) != self.account[account_field]:
            return deny(f"{account_field} is not the agentic account")
        if self.portfolio is None:
            return deny("portfolio unknown: call get_portfolio for the agentic account first")
        if self.held[asset] is None:
            return deny(f"positions unknown: call get_{kind}_positions for the agentic account first")

        # Freshness: act on what the broker says now, not on what it said a while ago.
        if self._age("portfolio") > settings.max_state_age_seconds:
            return deny(f"portfolio snapshot is {self._age('portfolio'):.0f}s old: call get_portfolio again")
        if self._age(f"positions:{asset}") > settings.max_state_age_seconds:
            return deny(f"{kind} positions are stale: call get_{kind}_positions again")

        if self.risk.circuit_breaker_tripped(self.portfolio["total_value"]):
            return deny("daily loss breaker tripped")
        if not self.exempt_daily_limit and self.risk.trade_budget_remaining() <= 0:
            return deny(f"daily limit of {self.limits.max_trades_today} orders reached")
        if not symbol:
            return deny("missing symbol")

        if side not in ("buy", "sell"):
            return deny("side must be buy or sell")
        if self.sell_only and side != "sell":
            return deny("this session may only sell; it cannot open or add to a position")
        if tool_input.get("type") != "market":
            return deny("only market orders are allowed")
        present = [field for field in FORBIDDEN_FIELDS if tool_input.get(field) not in (None, "", [])]
        if present:
            return deny(f"not allowed: {', '.join(present)}; size orders with dollar_amount only")
        if asset == "stock":
            if tool_input.get("market_hours", "regular_hours") != "regular_hours":
                return deny("equity orders must use regular_hours")
            if tool_input.get("time_in_force", "gfd") not in ("gfd", "gtc"):
                return deny("equity time_in_force must be gfd or gtc")
        elif tool_input.get("time_in_force", "gtc") != "gtc":
            return deny("crypto market orders must use time_in_force gtc")

        # Universe check: Robinhood must confirm this symbol is tradable for this account.
        if asset == "stock":
            tradable = self.tradability.get(symbol)
            if tradable is None:
                return deny(f"tradability of {symbol} unknown: call get_equity_tradability for it first")
            if not tradable["tradable"]:
                return deny(f"{symbol} is not fractionally tradable on this account")
        else:
            pair = self.pairs.get(symbol)
            if pair is None:
                return deny(f"{symbol} not found in get_currency_pairs: call it first")
            if not pair["tradable"]:
                return deny(f"{symbol} is not tradable on this account")
            if rh_parse.blocks_orders(pair):
                return deny(f"{symbol} is halted in all regions")

        # Anti-churn. Buying the same name again days apart is a position; doing it on
        # consecutive cycles is paying the spread twice for the same idea.
        since_buy = self._hours_since_buy(symbol)
        if since_buy is not None:
            if side == "buy" and since_buy < self.limits.cooldown_hours:
                return deny(f"{symbol} was bought {since_buy:.1f}h ago and the cooldown is "
                            f"{self.limits.cooldown_hours:.0f}h; adding again is churn, not conviction")
            if side == "sell" and settings.min_hold_hours and since_buy < settings.min_hold_hours:
                return deny(f"{symbol} was bought {since_buy:.1f}h ago and the minimum hold is "
                            f"{settings.min_hold_hours:.0f}h")

        try:
            requested = float(tool_input.get("dollar_amount"))
        except (TypeError, ValueError):
            return deny("dollar_amount must be a number")
        if not math.isfinite(requested) or requested <= 0:
            return deny("dollar_amount must be positive")
        verdict.requested = requested

        quantity, sellable = self.held[asset].get(symbol, (0.0, 0.0))
        quote = self.prices.get((asset, symbol))
        if quote is None:
            return deny(f"no quote for {symbol} this cycle: call get_{kind}_quotes first")
        quote_age = self._age((asset, symbol))
        if quote_age > settings.max_quote_age_seconds:
            return deny(f"{symbol} quote is {quote_age:.0f}s old: refetch get_{kind}_quotes before ordering")
        price = quote["price"]
        spread = quote["spread_pct"]
        if spread is None:
            return deny(f"{symbol} has no two-sided quote; refusing to trade a one-sided book")
        if spread > self.limits.max_spread_pct:
            return deny(f"{symbol} spread {spread:.2%} exceeds the {self.limits.max_spread_pct:.2%} limit")

        flow = self.cycle_flow.get((asset, symbol), 0.0)
        position_value = quantity * price + flow
        sellable_value = sellable * price + min(flow, 0.0)
        buying_power = self.portfolio["buying_power" if asset == "stock" else "crypto_buying_power"] - self.bought

        limit = self.risk.max_order_dollars(
            side, self.portfolio["total_value"], buying_power, position_value, sellable_value
        )

        # Concentration by story, not just by pair: four names in one trade, each under the
        # per-pair cap, were 78% of this account. Sells are never restricted by it.
        theme_note = ""
        if side == "buy" and self.limits.max_theme_pct < 1.0:
            room, theme, held_value = self._theme_room(asset, symbol, position_value)
            if room is not None and room < limit:
                if room <= 0:
                    return deny(f"{symbol} is part of the '{theme}' cluster, already "
                                f"${held_value:.2f} of a ${self.limits.max_theme_pct * self.portfolio['total_value']:.2f} "
                                f"limit ({self.limits.max_theme_pct:.0%} of the account): no room to add to this bet")
                theme_note = f"; capped by the '{theme}' cluster already holding ${held_value:.2f}"
                limit = room

        # Volatility targeting: a pair swinging twice as hard gets half the size. Only ever
        # shrinks an order, and only on the way in.
        vol_note = ""
        if side == "buy" and self.limits.vol_target_pct > 0:
            volatility = self._volatility(symbol)
            if volatility and volatility > self.limits.vol_target_pct:
                scaled = limit * (self.limits.vol_target_pct / volatility)
                if scaled < limit:
                    vol_note = (f"; scaled for {volatility:.1%} daily volatility against a "
                                f"{self.limits.vol_target_pct:.1%} target")
                    limit = scaled

        limit = _cents_down(limit)
        dollars = min(_cents_down(requested), limit)
        if dollars < MIN_ORDER_DOLLARS:
            return deny(f"limits allow ${limit:.2f} for this {side}, below the ${MIN_ORDER_DOLLARS:.2f} minimum")

        if asset == "crypto":
            minimum = self.pairs[symbol]["min_order_size"]
            if price > 0 and minimum and dollars / price < minimum:
                return deny(f"${dollars:.2f} buys less than {symbol}'s {minimum} unit minimum order size")

        if not self.exempt_daily_limit:
            self.risk.record_trade()
        self.cycle_flow[(asset, symbol)] = flow + (dollars if side == "buy" else -dollars)
        if side == "buy":
            self.bought += dollars
            # Start the cooldown immediately: the decision log is only written once the tool
            # returns, and a second buy can be proposed before that happens.
            self.recent_buys[symbol] = dt.datetime.now(dt.timezone.utc)

        verdict.allowed = True
        verdict.dollars = dollars
        verdict.price = price
        verdict.spread_pct = spread
        verdict.reason = (
            f"within limits{theme_note}" if dollars >= _cents_down(requested)
            else f"reduced from ${requested:.2f} to ${dollars:.2f} by risk limits{theme_note}{vol_note}"
        )
        verdict.updated_input = {**tool_input, "dollar_amount": f"{dollars:.2f}"}
        return verdict
