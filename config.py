import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# A blank ANTHROPIC_API_KEY still counts as "set" and outranks the Claude Code login, which
# fails auth instead of falling back. Remove it so the login is actually reached.
if not os.getenv("ANTHROPIC_API_KEY", "").strip():
    os.environ.pop("ANTHROPIC_API_KEY", None)


def _get_urls(name: str, default: str) -> list:
    """Comma-separated URLs. Unlike symbols these must keep their original case."""
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


@dataclass(frozen=True)
class Settings:
    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5")

    dry_run: bool = _get_bool("DRY_RUN", True)
    poll_interval_seconds: int = _get_int("POLL_INTERVAL_SECONDS", 3600)
    market_hours_only: bool = _get_bool("MARKET_HOURS_ONLY", True)
    max_cycle_budget_usd: float = _get_float("MAX_CYCLE_BUDGET_USD", 0.50)
    max_cycle_turns: int = _get_int("MAX_CYCLE_TURNS", 60)
    # Reasoning depth per model call: low, medium, high, xhigh or max.
    claude_effort: str = os.getenv("AGENT_EFFORT", "max").strip().lower() or "max"

    # Research desk: a read-only agent that scans the market before each trading session.
    research_enabled: bool = _get_bool("RESEARCH_ENABLED", True)
    research_effort: str = os.getenv("RESEARCH_EFFORT", "max").strip().lower() or "max"
    research_max_turns: int = _get_int("RESEARCH_MAX_TURNS", 150)
    research_budget_usd: float = _get_float("RESEARCH_BUDGET_USD", 10.0)
    # Past this the trader proceeds without a brief, so a slow desk cannot eat the trading window.
    research_timeout_minutes: int = _get_int("RESEARCH_TIMEOUT_MINUTES", 45)

    # Sell agent: woken by exit_watch when a position breaks a level. Sells only.
    sell_agent_enabled: bool = _get_bool("SELL_AGENT_ENABLED", True)
    sell_effort: str = os.getenv("SELL_EFFORT", "high").strip().lower() or "high"
    sell_max_turns: int = _get_int("SELL_MAX_TURNS", 40)
    sell_budget_usd: float = _get_float("SELL_BUDGET_USD", 5.0)

    # Largest share of the account in one correlated cluster (pairs moving together after BTC is
    # stripped out). Four DeFi names each under the per-pair cap were 78% of the account.
    max_theme_pct: float = _get_float("MAX_THEME_PCT", 0.65)
    theme_correlation: float = _get_float("THEME_CORRELATION", 0.6)

    # Risk agent: sets this run's limits against the growth target. Off = fixed limits from here.
    risk_agent_enabled: bool = _get_bool("RISK_AGENT_ENABLED", True)
    risk_effort: str = os.getenv("RISK_EFFORT", "high").strip().lower() or "high"
    risk_max_turns: int = _get_int("RISK_MAX_TURNS", 40)
    risk_budget_usd: float = _get_float("RISK_BUDGET_USD", 5.0)

    # Live mover monitor thresholds (fractions): what counts as a move worth flagging.
    mover_1h_pct: float = _get_float("MOVER_1H_PCT", 0.04)
    mover_24h_pct: float = _get_float("MOVER_24H_PCT", 0.10)
    mover_volume_surge_pct: float = _get_float("MOVER_VOLUME_SURGE_PCT", 1.50)
    journal_lookback: int = _get_int("JOURNAL_LOOKBACK", 5)

    # Per-order ceiling is a share of account value, so it scales with the account and the
    # agent sizes within it without asking. Cash, the per-symbol cap and the daily loss
    # breaker still bind underneath.
    max_trade_pct: float = _get_float("MAX_TRADE_PCT", 0.50)
    max_position_pct: float = _get_float("MAX_POSITION_PCT", 0.30)
    max_daily_loss_pct: float = _get_float("MAX_DAILY_LOSS_PCT", 0.15)
    # Dollar-denominated daily loss breaker. When above zero it takes precedence over the
    # percentage, so the halt point is a fixed amount rather than a share of the account.
    max_daily_loss_usd: float = _get_float("MAX_DAILY_LOSS_USD", 0.0)
    max_trades_per_day: int = _get_int("MAX_TRADES_PER_DAY", 3)
    min_cash_buffer: float = _get_float("MIN_CASH_BUFFER", 0.0)
    # There is no watchlist; this spread cap is the liquidity guard that replaces one.
    max_spread_pct: float = _get_float("MAX_SPREAD_PCT", 0.025)

    # Anti-churn, derived from measurement rather than taste: every pair quotes ~1.9% wide, so
    # repeatedly topping up one name burns the spread without changing the thesis.
    symbol_cooldown_hours: float = _get_float("SYMBOL_COOLDOWN_HOURS", 24.0)
    # Off by default: forcing a hold can trap the agent in a position that deserves exiting,
    # and there is no evidence yet that it helps.
    min_hold_hours: float = _get_float("MIN_HOLD_HOURS", 0.0)

    # Freshness. Every order must rest on data pulled moments earlier, not on a price the
    # model remembers from its journal or from earlier in a long session.
    max_quote_age_seconds: int = _get_int("MAX_QUOTE_AGE_SECONDS", 300)
    max_state_age_seconds: int = _get_int("MAX_STATE_AGE_SECONDS", 900)

    # Volatility targeting: an order in a pair running hotter than this daily standard
    # deviation is scaled down in proportion. Crypto's variance becomes smaller size rather
    # than a warning the model may ignore. 0 disables it.
    vol_target_pct: float = _get_float("VOL_TARGET_PCT", 0.03)

    # Advisory exit thresholds. Surfaced in the prompt so the agent must justify holding
    # through them; nothing here places a sell on its own.
    stop_loss_pct: float = _get_float("STOP_LOSS_PCT", 0.15)
    take_profit_pct: float = _get_float("TAKE_PROFIT_PCT", 0.25)

    # How many pairs get price-history context in the prompt. Each costs one Coinbase call
    # per day, cached, so this is a context-size limit rather than a network one.
    context_symbols_max: int = _get_int("CONTEXT_SYMBOLS_MAX", 24)

    # Crypto headlines, fetched by our code from publisher RSS feeds and summarized into the
    # prompt. Robinhood exposes no crypto news tool, and the agent has no web access.
    crypto_news_enabled: bool = _get_bool("CRYPTO_NEWS_ENABLED", True)
    crypto_news_feeds: list = field(default_factory=lambda: _get_urls(
        "CRYPTO_NEWS_FEEDS",
        "https://www.coindesk.com/arc/outboundfeeds/rss/,https://cointelegraph.com/rss,https://decrypt.co/feed",
    ))
    crypto_news_lookback_hours: int = _get_int("CRYPTO_NEWS_LOOKBACK_HOURS", 24)
    crypto_news_max_items: int = _get_int("CRYPTO_NEWS_MAX_ITEMS", 20)
    crypto_news_timeout_seconds: int = _get_int("CRYPTO_NEWS_TIMEOUT_SECONDS", 10)

    # Congressional STOCK Act disclosures, fetched by our code and summarized into the
    # prompt. The agent never browses the web itself.
    congress_enabled: bool = _get_bool("CONGRESS_ENABLED", True)
    congress_api_url: str = os.getenv("CONGRESS_API_URL", "https://www.bargo.ai/free-apis/congress/v1/trades")
    congress_lookback_days: int = _get_int("CONGRESS_LOOKBACK_DAYS", 60)
    congress_max_rows: int = _get_int("CONGRESS_MAX_ROWS", 40)
    congress_timeout_seconds: int = _get_int("CONGRESS_TIMEOUT_SECONDS", 20)

    log_dir: str = os.getenv("LOG_DIR", "logs")


settings = Settings()
