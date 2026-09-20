# Robinhood AI Crypto Agent

A long-running agent that trades **crypto only** in your Robinhood Agentic account. Each cycle, Claude
surveys Robinhood's crypto catalog, decides whether anything is worth owning, sizes its own order, and
places it. Every order passes through a code-enforced risk gate first.

## Read this first

- With `DRY_RUN=false` this places **real orders with real money**. Nothing here is financial advice, and
  an LLM's trade calls carry no expectation of profit.
- Trading is confined to the Agentic account. Fund it only with money you can afford to lose.
- **Running the agent costs usage too.** Each cycle is a Claude session, roughly $0.25–0.50 equivalent.
  With no `ANTHROPIC_API_KEY` set it draws on your Claude subscription instead of per-token billing.

### The honest limitation

Robinhood's MCP server exposes **no crypto historicals, no crypto news, and equity-only technical
indicators**. A crypto-only agent therefore decides on a live quote, the previous close, and the spread.
Equity news and index quotes are kept purely as a macro backdrop. Expect thin evidence, and expect the
agent to say so — it is instructed to admit a coin flip rather than dress one up as a thesis.

## How it works

The agent runs on the Claude Agent SDK, which reuses the Robinhood login you completed in Claude Code. No
Robinhood password or token is stored here.

Between cycles, **the mover monitor** (`mover_monitor.py`, task `RobinhoodMoverMonitor`) checks every tradable
pair every 5 minutes with one public Coinbase request. It is plain code, not a model. It records 15m / 1h / 4h
moves from its own snapshots, 24h moves and volume surges, and logs threshold alerts (`MOVER_*` settings) to
`logs/mover_alerts.jsonl`. The research desk reads this feed first.

Each cycle runs two agents in sequence:

0. **Research desk** (`research_agent.py`). A read-only agent with no account or order tools scans every
   tradable pair's price history and the headlines, investigates unusual movers with live quotes and web
   search, and returns a ranked brief: candidates with thesis, catalyst and invalidation, a call on each
   holding, and pairs to avoid. Its picks are logged to `logs/research_picks.jsonl` and scored against BTC,
   and that record is shown to it on the next run. If it fails or exceeds `RESEARCH_TIMEOUT_MINUTES`, the
   trader proceeds without a brief and scans the full market itself.

0a2. **Data feeds** (`mover_monitor.py`, `macro_feed.py`, `fundamentals.py`, `technicals.py`). All keyless, all code, no model
   calls: live movers every 5 minutes; a macro dashboard (yields, dollar, oil, gold, VIX, indices, crypto ETFs);
   volume and price structure computed from Coinbase candles and order book plus OKX open interest (TradingView
   has no free API and its scanner is off-limits under its terms, so the measures are computed from the same
   venue data it reads); and on-chain fundamentals from DefiLlama, CoinGecko, alternative.me and OKX — protocol deposits and their
   1d/7d momentum, aggregate stablecoin supply, sector rotation, perp funding and the Fear & Greed index. Each
   section is cached separately and degrades to stale-but-labelled rather than to silence.

0b. **Risk agent** (`risk_agent.py`). A second read-only agent (no order tools, account reads only) sets this
   run's limits against the `GROWTH_TARGET_USD` target: order size, per-pair cap, per-theme cap, trades today,
   volatility target, cooldown and max spread, each with a rationale. The theme cap exists because the per-pair
   cap does not stop one bet taking over: AAVE + UNI + ONDO + LINK reached ~78% of the account with every pair
   inside its own limit. Pairs sharing the research desk's theme label count as one bet (`themes.py`). It sees the scorecard, its own past policies, the
   macro tape and the research brief. Code validates the policy (`risk_policy.py`), logs it to
   `logs/risk_policy.jsonl` and enforces it. The operator's daily loss breaker is outside its reach, and an
   invalid or missing policy falls back to the fixed limits in `.env` — never to no limits.

0c. **Sell agent** (`sell_agent.py`). Rules on every holding — sell, trim or hold — before the trading agent
   looks for new buys, so realising a profit does not wait on a threshold, and the trader sizes against the cash
   it frees. It runs in a sell-only gate: a buy from this session is rejected in code. Between scheduled runs,
   `exit_watch.py` (task `RobinhoodExitWatch`, every 5 minutes, no model calls) checks live prices against each
   position's stop/take-profit guidelines and fast drops, and wakes the sell agent only when one trips — at most
   4 wakes a day, one per position per 4 hours. Protective exits are exempt from the daily trade budget, which
   exists to curb churn and must not trap a position.

Unscheduled cycles: `opportunity_watch.py` (task `RobinhoodOpportunityWatch`, every 15 minutes, no model calls)
screens the live mover feed for entry setups — a 5% hourly fall, a 3x volume surge with the price following, or a
12% daily fall. When one appears it writes the reason into the one-shot instruction file and runs a full live
cycle. It will not wake one when the day's order budget is spent, when the last known buying power is below $15,
when another cycle holds `logs/cycle.lock`, after two unscheduled cycles in a day, or within 6 hours of waking on
the same pair.

Then the trading agent, which alone can open positions:

1. It reads the brief (when present) and its journal of recent cycles of recent cycles — what it traded, why, and what that cost.
2. It checks the account: portfolio, buying power, crypto positions, realized P&L.
3. It pages through `get_currency_pairs` and prices candidates with `get_crypto_quotes`.
4. A `PostToolUse` hook parses Robinhood's own responses into cycle state. The gate never trusts numbers
   the model reports.
5. When it calls the order tool, a `PreToolUse` hook runs `order_gate.py` on the exact order and rejects
   it, allows it, or reduces `dollar_amount` to what the limits permit.

### What the gate enforces

- **Catalog membership** — the pair must appear in a `get_currency_pairs` page actually fetched and parsed
  this cycle, be tradable on this account, and not be halted.
- **Spread** — the quoted bid/ask spread must be inside `MAX_SPREAD_PCT`. A one-sided book is refused.
- **Minimum order size** — the order must buy at least the pair's minimum units.
- **Freshness** — a quote older than `MAX_QUOTE_AGE_SECONDS`, or an account snapshot older than
  `MAX_STATE_AGE_SECONDS`, is refused. No decision rests on a remembered price.
- **Size** — per-order ceiling as a share of account value, per-symbol concentration cap, buying power and
  cash buffer, counting orders already approved earlier in the same cycle.
- **Daily** — order count and loss breaker, persisted to disk so a restart grants nothing new.

A hook deny holds in every permission mode, and any tool not on the exact-name allowlist in `rh_tools.py`
is denied — including tools Robinhood adds later.

### What is blocked

**Equities and options entirely.** The equity order tools are not exposed, which also means the agent
cannot sell the SPY and AAPL already in the account; those show as context only. Cancels, option exercise,
alerts, watchlist edits and scanners are blocked, as are all built-in file, shell and web tools.

`congress.py` still ships and works, but `CONGRESS_ENABLED=false` — congressional STOCK Act filings are
stock and ETF trades, so they carry no signal for a crypto-only agent. Flip it back on if equities return.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Connect and authenticate Robinhood's MCP server once, in Claude Code:

```powershell
claude mcp add robinhood-trading --transport http https://agent.robinhood.com/mcp/trading
```

Then run `/mcp` inside Claude Code, select `robinhood-trading`, and approve it for your Agentic account.

Leave `ANTHROPIC_API_KEY` blank to run on your Claude login. `config.py` strips a blank value so it cannot
shadow that login.

## Run

Continuous loop:

```powershell
.\.venv\Scripts\python.exe main.py
```

One cycle and exit — this is what the scheduled task runs:

```powershell
.\.venv\Scripts\python.exe run_once.py --live
```

Add `--require-trade` to demand exactly one buy; omit it to let the agent decide. Don't run two cycles at
once, since each reads the daily order count at startup.

Output goes to the console and `logs/agent.log`. Every order attempt, including reductions and denials,
lands in `logs/trades.csv`; each cycle's reasoning and token usage in `logs/journal.jsonl`.

In dry run the gated tool is `preview_crypto_order`, which prices an order server-side without placing it.
`place_crypto_order` is removed from the model's context entirely.

## Settings

| Setting | Meaning |
| --- | --- |
| `DRY_RUN` | `true` simulates orders server-side. `false` trades real money. |
| `MAX_TRADE_PCT` | Per-order ceiling as a share of account value. The agent sizes freely under it. |
| `MAX_POSITION_PCT` | A pair's position may not exceed this share of portfolio value. |
| `MAX_DAILY_LOSS_PCT` | Drawdown from the day's opening value that halts all orders. |
| `MAX_DAILY_LOSS_USD` | Dollar drawdown that halts all orders. Above zero it overrides the percentage. |
| `MAX_TRADES_PER_DAY` | Hard cap on approved orders per calendar day. |
| `MAX_SPREAD_PCT` | Widest bid/ask spread it may trade into. The liquidity guard. |
| `SYMBOL_COOLDOWN_HOURS` | Hours before the same pair may be bought again. Sells are never blocked. |
| `STOP_LOSS_PCT` | Unrealized loss at which a position is flagged for an exit decision. Advisory. |
| `TAKE_PROFIT_PCT` | Unrealized gain at which a position is flagged for an exit decision. Advisory. |
| `MIN_HOLD_HOURS` | Minimum hold before selling. `0` disables it. |
| `MAX_QUOTE_AGE_SECONDS` | Oldest a quote may be when an order is placed. |
| `MAX_STATE_AGE_SECONDS` | Oldest the account snapshot may be when an order is placed. |
| `MIN_CASH_BUFFER` | Cash the agent will never spend below. |
| `JOURNAL_LOOKBACK` | How many past cycles the agent sees as memory. |
| `CRYPTO_NEWS_ENABLED` | Include the crypto headline digest in the prompt. |
| `CRYPTO_NEWS_FEEDS` | Comma-separated RSS feeds. Defaults to CoinDesk, Cointelegraph and Decrypt. |
| `CRYPTO_NEWS_LOOKBACK_HOURS` | How far back headlines are included. |
| `MARKET_HOURS_ONLY` | Leave `false`: crypto trades 24/7, so equity hours are not a meaningful gate. |
| `MAX_CYCLE_BUDGET_USD` | Spend cap per cycle. Still enforced on subscription auth. |
| `CLAUDE_MODEL` | Model for the agent. |

Dry-run and live modes keep separate daily state files, so dry runs never consume the live order budget.

## Scheduled runs

A one-time Windows task runs a single live cycle:

```powershell
schtasks /Query /TN "RobinhoodAgentLiveRun" /FO LIST     # inspect
schtasks /Delete /TN "RobinhoodAgentLiveRun" /F          # cancel
```

It is **Interactive only** — the machine must be awake and logged in, or it silently does not fire.

## Files

| File | Role |
| --- | --- |
| `main.py` | Daemon loop, signal handling |
| `run_once.py` | Single cycle for scheduled runs |
| `agent.py` | Agent SDK session per cycle, prompt, hooks, logging |
| `mover_monitor.py` | 5-minute live mover scan and alerts feeding the research desk (no model calls) |
| `research_agent.py` | Read-only research desk that briefs the trader; scores its own picks |
| `risk_agent.py` | Read-only risk agent that sets each run's limits against the growth target |
| `risk_policy.py` | Policy validation, logging and the fixed fallback limits |
| `sell_agent.py` | Sell-only agent: scheduled holding reviews and between-run exits |
| `opportunity_watch.py` | 15-minute entry screen that wakes a full live cycle when a setup appears (no model calls) |
| `portfolio_store.py` | Last observed account value and buying power, for code deciding whether a wake is affordable |
| `marks_store.py` | Broker marks, used to verify ticker identity across data sources |
| `exit_watch.py` | 5-minute position watch that wakes the sell agent when a level breaks (no model calls) |
| `themes.py` | What each pair is a bet on, so concentration is measured by story, not just by symbol |
| `technicals.py` | Volume and price structure: 6h relative volume, VWAP, order-book depth, open interest, RSI, OBV |
| `fundamentals.py` | On-chain fundamentals and flows: protocol TVL, stablecoin supply, sector rotation, funding, Fear & Greed |
| `macro_feed.py` | Live macro dashboard (yields, dollar, oil, VIX, ETFs) for the research and risk agents |
| `order_gate.py` | Deterministic order checks, universe guard and sizing |
| `rh_parse.py` | Parsers for Robinhood tool responses |
| `rh_tools.py` | Exact-name tool allowlists |
| `risk_manager.py` | Limits, per-order ceiling, persisted daily state |
| `journal.py` | Cycle memory fed back into the next prompt |
| `scorecard.py` | Scores past decisions against price history and a buy-and-hold BTC benchmark |
| `market_context.py` | Coinbase daily candles: moves, 30-day range, volume and BTC-relative performance |
| `positions_store.py` | Last observed holdings, so entry vs current is known without a broker call |
| `congress.py` | Congressional filings digest (disabled in crypto-only mode) |
| `trade_logger.py` | Console, file and CSV logging |
| `.mcp.json` | Robinhood MCP server configuration |
