"""Parsers for robinhood-trading tool results.

Shapes verified against live responses: every payload is {"data": ..., "guide": ...}. Each
function raises ValueError on a shape it does not recognize; the order gate treats that as
unknown state and refuses to size orders from it.
"""

import json


def _data(raw) -> dict:
    payload = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
        raise ValueError("payload has no data object")
    return payload["data"]


def _num(value) -> float:
    if value in (None, ""):
        raise ValueError("missing number")
    return float(value)


def _complete_list(data: dict, key: str) -> list:
    rows = data.get(key)
    if not isinstance(rows, list):
        raise ValueError(f"{key} list missing")
    if data.get("next"):
        # A partial page could hide a holding and understate position size.
        raise ValueError(f"{key} is paginated; refusing a partial view")
    return rows


def crypto_symbol(symbol: str) -> str:
    """'BTC-USD', 'BTCUSD' and 'btc' all become 'BTC'."""
    cleaned = str(symbol).upper().replace("-", "").strip()
    return cleaned[:-3] if cleaned.endswith("USD") and len(cleaned) > 3 else cleaned


def agentic_account(raw) -> dict:
    accounts = _data(raw).get("accounts")
    if not isinstance(accounts, list):
        raise ValueError("accounts list missing")
    matches = [account for account in accounts if account.get("agentic_allowed") is True]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one agentic account, found {len(matches)}")
    return {
        "account_number": str(matches[0]["account_number"]),
        "rhs_account_number": str(matches[0]["rhs_account_number"]),
    }


def portfolio(raw) -> dict:
    data = _data(raw)
    crypto = (data.get("crypto_buying_power") or {}).get("buying_power")
    return {
        "total_value": _num(data.get("total_value")),
        "buying_power": _num((data.get("buying_power") or {}).get("buying_power")),
        # Crypto is cash-only. When Robinhood omits the figure, assume none rather than
        # borrowing the marginable number, which can overstate it.
        "crypto_buying_power": float(crypto) if crypto not in (None, "") else 0.0,
    }


def equity_positions(raw) -> dict:
    held = {}
    for row in _complete_list(_data(raw), "positions"):
        quantity = _num(row.get("quantity"))
        sellable = _num(row.get("shares_available_for_sells", row.get("quantity")))
        held[str(row["symbol"]).upper()] = (quantity, sellable)
    return held


def crypto_positions(raw) -> dict:
    held = {}
    for row in _complete_list(_data(raw), "results"):
        code = (row.get("currency") or {}).get("code")
        if not code:
            raise ValueError("crypto position without currency.code")
        quantity = _num(row.get("quantity"))
        sellable = _num(row.get("quantity_transferable", row.get("quantity")))
        held[crypto_symbol(code)] = (quantity, sellable)
    return held


def _quote(bid, ask, fallback) -> dict:
    """Price plus relative spread. spread_pct is None when the book is one-sided."""
    bid_value, ask_value = float(bid or 0), float(ask or 0)
    if bid_value > 0 and ask_value > 0 and ask_value >= bid_value:
        mid = (bid_value + ask_value) / 2
        return {"price": mid, "spread_pct": (ask_value - bid_value) / mid}
    if fallback not in (None, "") and float(fallback) > 0:
        return {"price": float(fallback), "spread_pct": None}
    raise ValueError("no usable price")


def equity_prices(raw) -> dict:
    prices = {}
    for row in _data(raw).get("results") or []:
        quote = row.get("quote") or {}
        if not quote.get("symbol"):
            continue
        try:
            prices[str(quote["symbol"]).upper()] = _quote(
                quote.get("bid_price"), quote.get("ask_price"), quote.get("last_trade_price")
            )
        except ValueError:
            continue
    return prices


def crypto_prices(raw) -> dict:
    prices = {}
    for row in _data(raw).get("results") or []:
        if not row.get("symbol"):
            continue
        try:
            quote = _quote(row.get("bid_price"), row.get("ask_price"), row.get("mark_price"))
        except ValueError:
            continue
        prices[crypto_symbol(row["symbol"])] = quote
    return prices


def equity_tradability(raw) -> dict:
    """Symbol -> whether a dollar-based order can be placed on this account.

    Dollar-denominated orders are fractional and regular-hours only, so fractional
    eligibility is the binding requirement.
    """
    results = {}
    for row in _data(raw).get("results") or []:
        symbol = row.get("symbol")
        if not symbol:
            continue
        account_types = row.get("account_type_tradabilities") or []
        individual_ok = any(
            entry.get("account_type") == "individual"
            and "untradable" not in str(entry.get("account_type_tradability", ""))
            for entry in account_types
        )
        results[str(symbol).upper()] = {
            "tradable": bool(
                row.get("tradeable") is True
                and row.get("state") == "active"
                and "untradable" not in str(row.get("fractional_tradability", ""))
                and (individual_ok or not account_types)
            ),
            "name": row.get("simple_name") or row.get("name") or "",
        }
    return results


def blocks_orders(pair: dict) -> bool:
    """True when a pair's halt applies to this account.

    Robinhood marks many pairs halted with halted_regions like ["NY"] or ["TX"]: state-level
    restrictions, not trading halts. Previews of NY-halted ONDO and PAXG priced exactly like
    unhalted SOL on this account (2026-09-17), so only a halt naming no region blocks. A
    regional halt that does apply is still rejected by Robinhood at order time.
    """
    return bool(pair.get("halted")) and not pair.get("halted_regions")


def currency_pairs(raw) -> dict:
    """Symbol -> order constraints. Paginated pages are merged by the caller."""
    pairs = {}
    for row in _data(raw).get("results") or []:
        code = (row.get("asset_currency") or {}).get("code") or row.get("symbol")
        if not code:
            continue
        by_account = row.get("tradability_by_account_type") or {}
        pairs[crypto_symbol(code)] = {
            "symbol": row.get("symbol"),
            "tradable": row.get("tradability") == "tradable"
            and by_account.get("individual", "tradable") == "tradable",
            "halted": bool(row.get("halted")),
            "halted_regions": row.get("halted_regions") or [],
            "min_order_size": float(row.get("min_order_size") or 0),
        }
    return pairs
