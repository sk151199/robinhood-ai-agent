import datetime as dt
import json
import math
import os

import risk_policy
from config import settings

# Separate files so a morning of dry runs never consumes the live trade budget.
STATE_PATH = os.path.join(settings.log_dir, "daily_state_dry_run.json" if settings.dry_run else "daily_state.json")


class RiskManager:
    """Hard limits. The day's trade count and opening portfolio value are persisted, so a
    restart mid-day does not grant a fresh trade budget or re-anchor the loss breaker."""

    def __init__(self, limits=None):
        self._state = self._load()
        # Set once per run: the risk agent's policy for today, or the fixed fallback.
        self.limits = limits or risk_policy.effective()

    def apply_limits(self, limits):
        """Adopt the policy the risk agent set for this run."""
        self.limits = limits or risk_policy.FALLBACK

    def _load(self) -> dict:
        try:
            with open(STATE_PATH, encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, json.JSONDecodeError):
            return {}
        return state if isinstance(state, dict) else {}

    def _save(self):
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{STATE_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(self._state, handle)
        os.replace(tmp_path, STATE_PATH)

    def start_cycle(self, portfolio_value: float):
        today = dt.date.today().isoformat()
        if self._state.get("day") != today:
            self._state = {"day": today, "trades_today": 0, "day_start_value": portfolio_value}
            self._save()

    def resumed_today(self) -> bool:
        return self._state.get("day") == dt.date.today().isoformat() and bool(self._state.get("trades_today"))

    def trades_today(self) -> int:
        if self._state.get("day") != dt.date.today().isoformat():
            return 0
        return int(self._state.get("trades_today", 0))

    def day_start_value(self) -> float:
        """Today's opening value, or 0 before the first cycle of the day. The risk agent is
        given this as a starting point and refreshes it with get_portfolio itself."""
        if self._state.get("day") != dt.date.today().isoformat():
            return 0.0
        return float(self._state.get("day_start_value") or 0.0)

    def daily_loss_pct(self, portfolio_value: float) -> float:
        start_value = self._state.get("day_start_value") or 0.0
        if not start_value:
            return 0.0
        return (start_value - portfolio_value) / start_value

    def daily_loss_dollars(self, portfolio_value: float) -> float:
        start_value = self._state.get("day_start_value") or 0.0
        return max(0.0, start_value - portfolio_value)

    def circuit_breaker_tripped(self, portfolio_value: float) -> bool:
        """A dollar limit, when set, replaces the percentage rather than adding to it —
        otherwise the tighter percentage would fire first and the dollar figure would be
        decorative."""
        if settings.max_daily_loss_usd > 0:
            return self.daily_loss_dollars(portfolio_value) >= settings.max_daily_loss_usd
        return self.daily_loss_pct(portfolio_value) >= settings.max_daily_loss_pct

    def trade_budget_remaining(self) -> int:
        return max(0, self.limits.max_trades_today - self.trades_today())

    def per_order_ceiling(self, portfolio_value: float) -> float:
        """The agent sizes freely under this. It scales with the account, so it never needs
        raising by hand."""
        return self.limits.per_order_pct * portfolio_value

    def max_order_dollars(
        self, side: str, portfolio_value: float, buying_power: float, position_value: float, sellable_value: float
    ) -> float:
        ceiling = self.per_order_ceiling(portfolio_value)
        if side == "buy":
            limit = min(
                ceiling,
                self.limits.max_position_pct * portfolio_value - position_value,
                buying_power - settings.min_cash_buffer,
            )
        elif side == "sell":
            limit = min(ceiling, sellable_value)
        else:
            return 0.0
        return math.floor(round(max(0.0, limit) * 100, 6)) / 100

    def record_trade(self):
        self._state["trades_today"] = self.trades_today() + 1
        self._save()
