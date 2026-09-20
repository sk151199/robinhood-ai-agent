"""Agent-set risk policy, driven by the operator's growth target.

The trading agent chooses its own risk settings each session — order size, concentration,
trade count, volatility scaling, cooldown, spread tolerance — and must explain them against the
growth target. Code then enforces whatever it chose.

One limit is the operator's and cannot be changed by the agent: the daily loss breaker
(MAX_DAILY_LOSS_PCT). Bounds below are validity checks, not risk appetite: they reject values
that cannot be meant (a negative trade count, a 500% position), nothing more.

`effective()` is the single place both the risk manager and the order gate read their limits
from, so a run either enforces the agent's whole policy or the fixed fallback — never a mix.
"""

import datetime as dt
import json
import os
from dataclasses import dataclass

from config import settings

POLICY_PATH = os.path.join(settings.log_dir, "risk_policy.jsonl")
GROWTH_TARGET_USD = float(os.getenv("GROWTH_TARGET_USD", "5000") or 5000)

# field -> (min, max, description). Validity only.
FIELDS = {
    "per_order_pct": (0.01, 1.0, "largest single order as a fraction of account value"),
    "max_position_pct": (0.05, 1.0, "largest holding in one pair as a fraction of account value"),
    "max_trades_today": (0, 20, "orders allowed today, including any already placed"),
    "vol_target_pct": (0.0, 1.0, "daily volatility above which orders shrink proportionally; 0 disables"),
    "cooldown_hours": (0.0, 168.0, "hours before re-buying the same pair; 0 disables"),
    "max_spread_pct": (0.005, 0.10, "widest bid/ask spread you will pay"),
    "max_theme_pct": (0.1, 1.0, "largest share of the account in one correlated cluster of pairs — the "
                                "per-pair cap does not stop four names that move together becoming the portfolio"),
}


@dataclass(frozen=True)
class Limits:
    """What the gate and risk manager enforce for one run."""
    per_order_pct: float
    max_position_pct: float
    max_trades_today: int
    vol_target_pct: float
    cooldown_hours: float
    max_spread_pct: float
    max_theme_pct: float
    source: str  # "agent" or "fallback", for the log and the prompt

    @property
    def agent_set(self) -> bool:
        return self.source == "agent"


FALLBACK = Limits(
    per_order_pct=settings.max_trade_pct,
    max_position_pct=settings.max_position_pct,
    max_trades_today=settings.max_trades_per_day,
    vol_target_pct=settings.vol_target_pct,
    cooldown_hours=settings.symbol_cooldown_hours,
    max_spread_pct=settings.max_spread_pct,
    max_theme_pct=settings.max_theme_pct,
    source="fallback",
)


def effective(policy: dict = None) -> Limits:
    """The given policy, else today's saved policy, else the operator's fixed limits.

    A missing or unusable policy falls back to config.py rather than to no limits: a failed
    risk agent must never widen what the trader may do.
    """
    policy = policy if policy is not None else today()
    if not policy:
        return FALLBACK
    try:
        return Limits(per_order_pct=float(policy["per_order_pct"]),
                      max_position_pct=float(policy["max_position_pct"]),
                      max_trades_today=int(policy["max_trades_today"]),
                      vol_target_pct=float(policy["vol_target_pct"]),
                      cooldown_hours=float(policy["cooldown_hours"]),
                      max_spread_pct=float(policy["max_spread_pct"]),
                      max_theme_pct=float(policy.get("max_theme_pct") or settings.max_theme_pct),
                      source="agent")
    except (KeyError, TypeError, ValueError):
        return FALLBACK


def validate(policy: dict) -> tuple:
    """(clean policy, list of problems). A policy with problems is rejected whole, not clamped:
    silently adjusting a number the agent chose would enforce something it never decided."""
    problems, clean = [], {}
    for name, (low, high, _desc) in FIELDS.items():
        try:
            value = float(policy[name])
        except (KeyError, TypeError, ValueError):
            problems.append(f"{name} missing or not a number")
            continue
        if not low <= value <= high:
            problems.append(f"{name}={value} outside the valid range {low}-{high}")
        clean[name] = int(value) if name == "max_trades_today" else value
    rationale = str(policy.get("rationale") or "").strip()
    if len(rationale) < 40:
        problems.append("rationale must explain the choice against the growth target (40+ characters)")
    clean["rationale"] = rationale[:1500]
    return clean, problems


def save(policy: dict, account_value: float) -> dict:
    entry = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "account_value": round(account_value, 2),
             "growth_target_usd": GROWTH_TARGET_USD, "daily_loss_limit_pct": settings.max_daily_loss_pct, **policy}
    os.makedirs(settings.log_dir, exist_ok=True)
    with open(POLICY_PATH, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    return entry


def history() -> list:
    try:
        with open(POLICY_PATH, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, json.JSONDecodeError):
        return []


def today() -> dict:
    """The latest policy set today (UTC day), or {} when none has been set."""
    stamp = dt.datetime.now(dt.timezone.utc).date().isoformat()
    todays = [entry for entry in history() if entry["timestamp"].startswith(stamp)]
    return todays[-1] if todays else {}


def as_prompt_section(account_value: float = None, limit: int = 5) -> str:
    lines = [f"Growth target: ${GROWTH_TARGET_USD:,.0f}."]
    if account_value:
        lines[0] += (f" Account ${account_value:,.2f} — {account_value / GROWTH_TARGET_USD:.1%} of the way; "
                     f"reaching it needs {GROWTH_TARGET_USD / account_value:.0f}x, mostly from deposits at this size.")
    lines.append(f"Fixed by the operator, not yours to change: a {settings.max_daily_loss_pct:.0%} daily loss breaker.")
    lines.append("You set everything else each session with set_risk_policy, before any order:")
    lines += [f"  {name}: {desc} (valid {low}-{high})" for name, (low, high, desc) in FIELDS.items()]
    past = history()[-limit:]
    if past:
        lines.append("Your recent policies (compare against what happened to the account since):")
        for entry in past:
            lines.append(f"  {entry['timestamp'][:16]} value ${entry['account_value']}: order {entry['per_order_pct']:.0%}, "
                         f"position {entry['max_position_pct']:.0%}, trades {entry['max_trades_today']}, "
                         f"vol {entry['vol_target_pct']:.1%}, cooldown {entry['cooldown_hours']:.0f}h, "
                         f"spread {entry['max_spread_pct']:.1%}, theme {entry.get('max_theme_pct', 0):.0%}"
                         f" — {entry['rationale'][:140]}")
    return "\n".join(lines)
