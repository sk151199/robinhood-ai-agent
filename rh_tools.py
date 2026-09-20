"""Exact-name tool lists for the robinhood-trading MCP server.

Crypto-only configuration: the equity order tools are not exposed at all, so the agent can
neither buy nor sell stocks. Equity reads are kept only where they inform crypto — account
context and macro headlines — because every tool left in the list costs context on every turn.

Never widen these to a prefix or pattern. Robinhood's tool descriptions reference tools the
server does not expose yet (replace_option_order, get_advanced_orders); a pattern would admit
them the day they ship. The PreToolUse hook denies any tool not named here.
"""

from config import settings

SERVER = "robinhood-trading"
PREFIX = f"mcp__{SERVER}__"


def _full(*names: str) -> tuple:
    return tuple(PREFIX + name for name in names)


# Account and position state. get_equity_positions stays for context only — with no equity
# order tools, the agent can see the SPY/AAPL holdings but cannot act on them.
ACCOUNT_TOOLS = _full(
    "get_accounts",
    "get_portfolio",
    "get_crypto_positions",
    "get_crypto_orders",
    "get_equity_positions",
    "get_realized_pnl",
    "get_pnl_trade_history",
)

# Robinhood exposes no crypto historicals and no crypto news, so market_context supplies the
# history from Coinbase and equity news is the only route to a macro headline.
MARKET_TOOLS = _full(
    "get_crypto_quotes",
    "get_currency_pairs",
    "get_equity_news",
    "get_indexes",
    "get_index_quotes",
)

DISCOVERY_TOOLS = _full("search")

READ_TOOLS = ACCOUNT_TOOLS + MARKET_TOOLS + DISCOVERY_TOOLS

PLACE = {"crypto": PREFIX + "place_crypto_order"}
SIMULATE = {"crypto": PREFIX + "preview_crypto_order"}

# Equity and option order tools are blocked outright in this configuration.
NEVER_EXPOSED = _full(
    "place_equity_order",
    "review_equity_order",
    "place_option_order",
    "review_option_order",
    "exercise_option",
    "cancel_option_exercise",
    "get_option_level_upgrade_info",
    "cancel_equity_order",
    "cancel_crypto_order",
    "cancel_option_order",
    "create_alert",
    "update_alert",
    "delete_alert",
    "mark_alerts_read",
    "create_watchlist",
    "update_watchlist",
    "follow_watchlist",
    "unfollow_watchlist",
    "add_to_watchlist",
    "remove_from_watchlist",
    "add_option_to_watchlist",
    "remove_option_from_watchlist",
    "create_scan",
    "update_scan_filters",
    "update_scan_config",
)

BUILTIN_BLOCKED = ("Bash", "Read", "Write", "Edit", "Glob", "Grep", "NotebookEdit", "WebFetch", "WebSearch", "Agent")
HARNESS_TOOLS = ("ToolSearch",)


def gated_order_tools() -> dict:
    """Tool name -> asset class, for the tools that place (or in dry run, simulate) orders."""
    tools = SIMULATE if settings.dry_run else PLACE
    return {name: asset for asset, name in tools.items()}


def passthrough_tools() -> set:
    """Allowed without gating. In live mode the simulate tool is a harmless preview."""
    names = set(HARNESS_TOOLS) | set(READ_TOOLS)
    if not settings.dry_run:
        names |= set(SIMULATE.values())
    return names


def allowed_tools() -> list:
    return sorted(passthrough_tools() | set(gated_order_tools()))


def disallowed_tools() -> list:
    names = list(BUILTIN_BLOCKED) + list(NEVER_EXPOSED)
    if settings.dry_run:
        names += list(PLACE.values())
    return names
