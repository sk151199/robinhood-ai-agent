"""Per-position exit plans, set by the sell agent rather than fixed in configuration.

A single take-profit percentage cannot be right for every holding: a pair that moves 6% a day
reaches +25% on noise, while Bitcoin at +25% is a genuine event. Worse, the levels that actually
matter were already being written in prose — "wrong if LINK closes below $11.40" — where no code
could act on them, so a broken thesis waited for the next scheduled review to be noticed.

The sell agent now states, for every holding it rules on, the level at which it would take profit,
the loss it is prepared to sit through, and the price that proves the thesis wrong. Those numbers
are stored here and enforced by exit_watch every five minutes. Configuration keeps the same
thresholds only as a fallback for a position no plan covers yet.
"""

import datetime as dt
import json
import os

from config import settings

PLANS_PATH = os.path.join(settings.log_dir, "exit_plans.json")
# Bounds are sanity checks, not opinions: they reject a number that cannot be meant.
TAKE_PROFIT_RANGE = (0.03, 5.0)
STOP_RANGE = (0.03, 0.9)


def load() -> dict:
    try:
        with open(PLANS_PATH, encoding="utf-8") as handle:
            plans = json.load(handle)
        return plans if isinstance(plans, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        return {}


def _clean(row: dict) -> dict:
    """The whole plan or nothing. Keeping the half of a plan that parsed would enforce a
    combination the agent never chose — the same reason a risk policy is rejected whole."""
    plan = {}
    for name, (low, high) in (("take_profit_pct", TAKE_PROFIT_RANGE), ("stop_pct", STOP_RANGE)):
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError):
            return {}
        if not low <= value <= high:
            return {}
        plan[name] = value
    try:
        price = float(row.get("invalidation_price"))
        if price > 0:
            plan["invalidation_price"] = price
    except (TypeError, ValueError):
        pass
    note = str(row.get("note") or "").strip()
    if note:
        plan["note"] = note[:300]
    return plan


def remember(rulings) -> int:
    """Store the plans from one sell-agent review. Returns how many were usable."""
    plans = load()
    stamped = dt.datetime.now(dt.timezone.utc).isoformat()
    kept = 0
    for row in rulings or []:
        symbol = str((row or {}).get("symbol", "")).upper()
        plan = _clean(row or {})
        if not symbol or not plan:
            continue
        plans[symbol] = {**plan, "at": stamped}
        kept += 1
    if kept:
        try:
            os.makedirs(settings.log_dir, exist_ok=True)
            tmp_path = f"{PLANS_PATH}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as handle:
                json.dump(plans, handle, indent=2, sort_keys=True)
            os.replace(tmp_path, PLANS_PATH)
        except OSError:
            return 0
    return kept


def forget(symbols):
    """Drop plans for positions that no longer exist, so a re-entry starts fresh."""
    plans = load()
    remaining = {symbol: plan for symbol, plan in plans.items()
                 if symbol not in {str(s).upper() for s in symbols}}
    if len(remaining) != len(plans):
        try:
            with open(PLANS_PATH, "w", encoding="utf-8") as handle:
                json.dump(remaining, handle, indent=2, sort_keys=True)
        except OSError:
            pass


def for_symbol(symbol: str) -> dict:
    """The agent's plan for this pair, falling back to the operator's fixed thresholds."""
    plan = load().get(str(symbol).upper()) or {}
    return {
        "take_profit_pct": plan.get("take_profit_pct", settings.take_profit_pct),
        "stop_pct": plan.get("stop_pct", settings.stop_loss_pct),
        "invalidation_price": plan.get("invalidation_price"),
        "note": plan.get("note", ""),
        "at": plan.get("at"),
        "agent_set": bool(plan),
    }


def as_prompt_section(symbols=()) -> str:
    plans = load()
    wanted = [str(s).upper() for s in symbols] or sorted(plans)
    lines = ["Your standing exit plans. You set these; they are enforced between runs, and you may revise any "
             "of them this session. A position with no plan falls back to the operator's fixed "
             f"+{settings.take_profit_pct:.0%} / -{settings.stop_loss_pct:.0%} thresholds, which are a placeholder, "
             "not a judgement."]
    for symbol in wanted:
        plan = for_symbol(symbol)
        source = "yours" if plan["agent_set"] else "fallback"
        detail = (f"take profit +{plan['take_profit_pct']:.0%}, stop -{plan['stop_pct']:.0%}"
                  + (f", invalidated below ${plan['invalidation_price']:,.6g}" if plan["invalidation_price"] else ""))
        note = f" — {plan['note']}" if plan["note"] else ""
        lines.append(f"  {symbol}: {detail} ({source}){note}")
    return "\n".join(lines)
