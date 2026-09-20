"""Did the agent's decisions actually work?

Every approved order is recorded with the price and spread at the moment it was made. Later,
this scores it against what the price actually did, net of the spread it had to cross, and
against the only benchmark that matters on a crypto account: buying and holding BTC.

Prices come from Coinbase's public candles rather than Robinhood, because this runs outside
the agent and has no broker access. That independence is a feature — the agent cannot mark
its own homework.

Run it any time:  python scorecard.py
"""

import datetime as dt
import json
import os

import market_context
import positions_store
from config import settings

# Separate files, like the daily risk state: a simulated buy must never start a live cooldown,
# pose as a live entry price, or be scored as a real decision.
DECISIONS_PATH = os.path.join(settings.log_dir, "decisions_dry_run.jsonl" if settings.dry_run else "decisions.jsonl")
HORIZONS_DAYS = (1, 7, 30)
BENCHMARK = "BTC"


def record(symbol: str, side: str, dollars: float, price: float, spread_pct: float, mode: str,
           reason: str, forced: bool = False):
    """One line per approved order, written at decision time.

    `forced` marks an order the operator demanded via require_trade. Those are not the agent's
    judgement and are excluded from the scored record — otherwise it reasons about its own
    ability from trades it never chose to make.
    """
    if not symbol or price <= 0:
        return
    entry = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "symbol": symbol.upper(),
        "side": side,
        "dollars": round(dollars, 2),
        "price": price,
        "spread_pct": spread_pct,
        "mode": mode,
        "forced": bool(forced),
        "reason": reason[:200],
    }
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(DECISIONS_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
    except OSError:
        pass


LIVE_DECISIONS_PATH = os.path.join(settings.log_dir, "decisions.jsonl")


def decisions(path: str = None) -> list:
    try:
        with open(path or DECISIONS_PATH, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def attach_summary(summary: str, since_iso: str):
    """Store the cycle's own reasoning on the decisions it produced.

    The thesis is written at the end of a cycle, after the orders. Pairing them lets a later
    cycle be shown what it claimed would happen next to what actually did.
    """
    if not summary:
        return
    entries = decisions()
    if not entries:
        return
    changed = False
    for entry in entries:
        if entry.get("timestamp", "") >= since_iso and not entry.get("summary"):
            entry["summary"] = summary.strip()[:600]
            changed = True
    if not changed:
        return
    try:
        tmp_path = f"{DECISIONS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write("".join(json.dumps(entry) + "\n" for entry in entries))
        os.replace(tmp_path, DECISIONS_PATH)
    except OSError:
        pass


PASSES_PATH = os.path.join(settings.log_dir, "passes.jsonl")


def record_passes(candidates: list, mode: str):
    """Candidates the agent considered and rejected.

    Declining a pair that then falls is a genuinely good decision, and until now it left no
    trace at all — only purchases were ever recorded, so the record flattered action over
    restraint.
    """
    if not candidates:
        return
    stamped = dt.datetime.now(dt.timezone.utc).isoformat()
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(PASSES_PATH, "a", encoding="utf-8") as handle:
            for symbol, reason in candidates:
                handle.write(json.dumps({"timestamp": stamped, "symbol": symbol.upper(),
                                         "reason": reason[:200], "mode": mode}) + "\n")
    except OSError:
        pass


def passes() -> list:
    try:
        with open(PASSES_PATH, encoding="utf-8") as handle:
            lines = handle.readlines()
    except FileNotFoundError:
        return []
    entries = []
    for line in lines:
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def passes_review(limit: int = 8) -> str:
    """What happened to the pairs it declined."""
    recent = passes()[-limit:]
    if not recent:
        return "No passes recorded yet."

    lines = []
    for entry in recent:
        prices = market_context.closes(entry["symbol"])
        if not prices:
            continue
        passed_on = dt.date.fromisoformat(entry["timestamp"][:10])
        then = _close_on_or_before(prices, passed_on)
        now = prices[max(prices)]
        if not then:
            continue
        move = (now - then) / then
        verdict = "good pass" if move < 0 else "it ran without you"
        lines.append(f"  {entry['symbol']}: passed {passed_on.isoformat()}, since then {move:+.1%} "
                     f"({verdict}) — you said: {entry['reason'][:90]}")
    if not lines:
        return "No passes with usable price history yet."
    return ("Pairs you declined, and what they did after. Declining correctly is a real skill and this is "
            "the only place it is measured:\n" + "\n".join(lines))


def _last_buy(symbol: str):
    latest = None
    # Holdings are real even in a dry run, so their entries come from the live record too.
    for entry in decisions() + (decisions(LIVE_DECISIONS_PATH) if settings.dry_run else []):
        if entry.get("side") == "buy" and str(entry.get("symbol", "")).upper() == symbol.upper():
            if latest is None or entry.get("timestamp", "") > latest.get("timestamp", ""):
                latest = entry
    return latest


def position_review() -> str:
    """Every open position with entry, current price, unrealized move and its own thesis.

    This is what makes an exit decision possible: without the entry price the agent cannot
    tell a winner from a loser, and without the thesis it cannot tell whether the reason it
    bought still holds.
    """
    positions, updated = positions_store.load()
    live = {symbol: values for symbol, values in positions.items() if values[0] > 0}
    if not live:
        return "No open crypto positions."

    lines = [f"Holdings as Robinhood last reported them ({updated[:16].replace('T', ' ')} UTC):"]
    for symbol, (quantity, _sellable) in sorted(live.items()):
        current = market_context.stats(symbol).get("price")
        entry = _last_buy(symbol)
        entry_price = (entry or {}).get("price")

        detail = f"  {symbol}: {quantity:.8f} units"
        if entry_price and current:
            move = (current - entry_price) / entry_price
            detail += (f", entry ${entry_price:,.4f} on {entry['timestamp'][:10]}, "
                       f"now ${current:,.4f}, unrealized {move:+.1%}")
            if move <= -settings.stop_loss_pct:
                detail += f"  <-- past your -{settings.stop_loss_pct:.0%} stop guideline"
            elif move >= settings.take_profit_pct:
                detail += f"  <-- past your +{settings.take_profit_pct:.0%} target"
        elif current:
            detail += f", now ${current:,.4f}, entry price unknown"
        lines.append(detail)

        thesis = (entry or {}).get("summary")
        if thesis:
            lines.append(f"     you said: {thesis[:220]}")

    lines.append("Rule on each holding before proposing any new buy: hold, trim or exit, and say why. "
                 "The thresholds are guidance, not automatic — nothing sells unless you order it.")
    lines.append("Unrealized figures above are mid-to-mid. You bought at the ask and would exit at the bid, "
                 "so a round trip costs roughly the quoted spread (~1.9% on these pairs). A position needs to "
                 "be up about 2% before selling realizes anything at all; taking a 1% gain is a loss. Selling "
                 "to cut a broken thesis is a different decision and does not have to clear that bar.")
    return "\n".join(lines)


def last_buy_times() -> dict:
    """symbol -> when it was last bought. Feeds the gate's cooldown rule."""
    times = {}
    # A dry run still honours real cooldowns, so it behaves like the live run it rehearses.
    entries = decisions() + (decisions(LIVE_DECISIONS_PATH) if settings.dry_run else [])
    for entry in entries:
        if entry.get("side") != "buy":
            continue
        try:
            stamped = dt.datetime.fromisoformat(entry["timestamp"])
        except (KeyError, TypeError, ValueError):
            continue
        symbol = str(entry.get("symbol", "")).upper()
        if symbol and (symbol not in times or stamped > times[symbol]):
            times[symbol] = stamped
    return times


def closes(symbol: str) -> dict:
    """date -> close, via the shared day-cache in market_context so scoring and the price
    context section do not download the same series twice."""
    return market_context.closes(symbol)


def _close_on_or_before(prices: dict, target: dt.date):
    for offset in range(0, 5):
        candidate = target - dt.timedelta(days=offset)
        if candidate in prices:
            return prices[candidate]
    return None


def score(today: dt.date = None) -> dict:
    """Net return per decision at each horizon, and the excess over holding BTC."""
    today = today or dt.datetime.now(dt.timezone.utc).date()
    # Operator-forced orders are excluded: they measure my instruction, not its judgement.
    entries = [entry for entry in decisions() if not entry.get("forced")]
    if not entries:
        return {"decisions": 0, "horizons": {}}

    price_cache = {BENCHMARK: closes(BENCHMARK)}
    results = {days: [] for days in HORIZONS_DAYS}

    for entry in entries:
        symbol = entry["symbol"]
        if symbol not in price_cache:
            price_cache[symbol] = closes(symbol)
        prices = price_cache[symbol]
        if not prices:
            continue

        decided_on = dt.date.fromisoformat(entry["timestamp"][:10])
        direction = -1.0 if entry.get("side") == "sell" else 1.0

        for days in HORIZONS_DAYS:
            target = decided_on + dt.timedelta(days=days)
            if target > today:
                continue  # horizon has not elapsed yet
            later = _close_on_or_before(prices, target)
            if later is None or not entry.get("price"):
                continue

            gross = direction * (later - entry["price"]) / entry["price"]
            # Crossing the book costs roughly the full quoted spread over a round trip:
            # half on the way in at the ask, half on the way out at the bid.
            net = gross - float(entry.get("spread_pct") or 0.0)

            benchmark_then = _close_on_or_before(price_cache[BENCHMARK], decided_on)
            benchmark_now = _close_on_or_before(price_cache[BENCHMARK], target)
            excess = None
            if benchmark_then and benchmark_now:
                excess = net - (benchmark_now - benchmark_then) / benchmark_then

            results[days].append({"symbol": symbol, "net": net, "excess": excess})

    summary = {}
    for days, scored in results.items():
        if not scored:
            continue
        with_excess = [row["excess"] for row in scored if row["excess"] is not None]
        summary[days] = {
            "n": len(scored),
            "hit_rate": sum(1 for row in scored if row["net"] > 0) / len(scored),
            "mean_net": sum(row["net"] for row in scored) / len(scored),
            "mean_excess": (sum(with_excess) / len(with_excess)) if with_excess else None,
            "beat_btc_rate": (sum(1 for value in with_excess if value > 0) / len(with_excess))
            if with_excess else None,
        }
    return {"decisions": len(entries), "horizons": summary}


def as_report() -> str:
    result = score()
    if not result["decisions"]:
        return "No decisions recorded yet."
    lines = [f"{result['decisions']} decision(s) recorded.", ""]
    if not result["horizons"]:
        return "\n".join(lines + ["None have aged past the 1-day horizon yet — nothing is scorable."])
    lines.append(f"{'horizon':>8}  {'n':>3}  {'profitable':>10}  {'mean net':>9}  {'vs BTC':>8}  {'beat BTC':>8}")
    for days, stats in sorted(result["horizons"].items()):
        excess = f"{stats['mean_excess']:+.2%}" if stats["mean_excess"] is not None else "n/a"
        beat = f"{stats['beat_btc_rate']:.0%}" if stats["beat_btc_rate"] is not None else "n/a"
        lines.append(f"{days:>6}d  {stats['n']:>3}  {stats['hit_rate']:>10.0%}  "
                     f"{stats['mean_net']:>+9.2%}  {excess:>8}  {beat:>8}")
    lines.append("")
    lines.append("Net is after the quoted spread. vs BTC is the same money held in Bitcoin instead.")
    return "\n".join(lines)


def as_prompt_section() -> str:
    """Short enough to sit in every cycle prompt."""
    result = score()
    if not result["decisions"]:
        return "No scored decisions yet — this is early, and nothing here proves the approach works."
    if not result["horizons"]:
        return f"{result['decisions']} decision(s) recorded, none aged enough to score yet."

    parts = []
    for days, stats in sorted(result["horizons"].items()):
        excess = f", {stats['mean_excess']:+.1%} vs BTC" if stats["mean_excess"] is not None else ""
        parts.append(f"{days}d: {stats['n']} trades, {stats['hit_rate']:.0%} profitable, "
                     f"{stats['mean_net']:+.1%} net{excess}")
    return ("Your own track record, scored independently after spread costs:\n"
            + "\n".join(f"  {part}" for part in parts)
            + "\nIf you are not beating BTC, the honest move is fewer trades, not more.")


if __name__ == "__main__":
    print(as_report())
