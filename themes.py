"""What each pair is a bet on, so concentration can be measured by story, not just by symbol.

A per-pair cap does not stop four names in the same trade becoming the whole account: after the
AAVE buy, AAVE + UNI + ONDO + LINK were about 78% of the portfolio, each comfortably under the
45% per-pair limit. The trader flagged it itself.

Price correlation turned out to be the wrong instrument for this. Measured on BTC-excess daily
returns, AAVE-UNI correlates 0.39 while AAVE-DOGE correlates 0.68 — it finds shared beta, not a
shared story. So the label is primary: the research desk names each candidate's theme, and that
label is remembered here. Correlation is kept only as a fallback for a pair nothing has labelled.

Labels are lowercase slugs, e.g. "defi-lending", "rwa-tokenization", "l1-major".
"""

import json
import os
import re

import market_context
from config import settings

THEMES_PATH = os.path.join(settings.log_dir, "themes.json")

# Seeded from what the account already holds, so the cap works before the desk labels anything.
SEED = {
    "BTC": "l1-major", "ETH": "l1-major", "SOL": "l1-major",
    "AAVE": "defi", "UNI": "defi", "MORPHO": "defi", "CRV": "defi", "SYRUP": "defi",
    "ONDO": "rwa-tokenization", "PAXG": "rwa-tokenization",
    "LINK": "oracle-infra", "PYTH": "oracle-infra",
}


def _normalize(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(label or "").strip().lower()).strip("-")
    return slug[:40]


def load() -> dict:
    try:
        with open(THEMES_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        stored = {}
    merged = dict(SEED)
    if isinstance(stored, dict):
        merged.update({str(k).upper(): _normalize(v) for k, v in stored.items() if _normalize(v)})
    return merged


def remember(labels: dict):
    """Persist symbol -> theme labels, newest winning."""
    if not labels:
        return
    try:
        with open(THEMES_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
        stored = stored if isinstance(stored, dict) else {}
    except (OSError, ValueError, json.JSONDecodeError):
        stored = {}
    for symbol, label in labels.items():
        slug = _normalize(label)
        if slug:
            stored[str(symbol).upper()] = slug
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{THEMES_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(stored, handle, indent=2, sort_keys=True)
        os.replace(tmp_path, THEMES_PATH)
    except OSError:
        pass


def remember_brief(brief: dict):
    if not brief:
        return
    remember({row["symbol"]: row.get("theme") for row in brief.get("candidates") or [] if row.get("theme")})


def theme_of(symbol: str) -> str:
    return load().get(str(symbol).upper(), "")


def cluster(symbol: str, candidates) -> list:
    """Which of `candidates` are the same bet as `symbol`, including itself when held.

    Same label first. For a pair nothing has labelled, fall back to correlated movement, which
    at least catches a pair that has been trading as the same thing.
    """
    symbol = str(symbol).upper()
    labels = load()
    mine = labels.get(symbol, "")
    same = []
    for other in candidates:
        other = str(other).upper()
        if other == symbol:
            same.append(other)
            continue
        theirs = labels.get(other, "")
        if mine and theirs:
            if theirs == mine:
                same.append(other)
            continue
        correlation = market_context.correlation(symbol, other)
        if correlation is not None and correlation >= settings.theme_correlation:
            same.append(other)
    return same
