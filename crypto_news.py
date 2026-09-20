"""Crypto headlines, summarized into the cycle prompt.

Robinhood's MCP server has no crypto news tool, so this fills the gap. Fetched here rather
than by the agent on purpose: the agent has no web tools, so nothing it reads from the
internet can steer a tool call. Headlines arrive as bounded, plain-text data.

RSS rather than a news API: publisher feeds need no key, return stable XML the standard
library parses, and three independent sources mean one outage does not blind the agent.
"""

import datetime as dt
import json
import os
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

from config import settings
from trade_logger import logger

CAVEAT = ("Headlines below are third-party reporting, not verified fact, and they are already priced in to "
          "some degree. Weigh them against the quote and the book; never trade on a headline alone.")

CACHE_MAX_AGE_MINUTES = 30
MAX_TITLE_CHARS = 140


def _cache_path() -> str:
    return os.path.join(settings.log_dir, "crypto_news_cache.json")


def _read_cache():
    try:
        with open(_cache_path(), encoding="utf-8") as handle:
            cached = json.load(handle)
        stamped = dt.datetime.fromisoformat(cached["timestamp"])
        if (dt.datetime.now(dt.timezone.utc) - stamped).total_seconds() <= CACHE_MAX_AGE_MINUTES * 60:
            return cached["section"]
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None
    return None


def _write_cache(section: str):
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(_cache_path(), "w", encoding="utf-8") as handle:
            json.dump({"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "section": section}, handle)
    except OSError:
        pass


def _fetch(url: str) -> bytes:
    """Seam for tests."""
    request = urllib.request.Request(url, headers={"User-Agent": "robinhood-ai-agent"})
    with urllib.request.urlopen(request, timeout=settings.crypto_news_timeout_seconds) as response:
        return response.read()


def _published(item) -> dt.datetime:
    raw = item.findtext("pubDate") or item.findtext("{http://purl.org/dc/elements/1.1/}date")
    stamped = parsedate_to_datetime(raw)
    return stamped if stamped.tzinfo else stamped.replace(tzinfo=dt.timezone.utc)


def _feed_items(url: str, cutoff: dt.datetime) -> list:
    root = ET.fromstring(_fetch(url))
    source = (root.findtext(".//channel/title") or url).strip()
    items = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title:
            continue
        try:
            published = _published(item)
        except (TypeError, ValueError):
            continue
        if published < cutoff:
            continue
        items.append({"title": title[:MAX_TITLE_CHARS], "source": source, "published": published})
    return items


def recent_headlines(now: dt.datetime = None) -> list:
    """Headlines from every reachable feed inside the lookback window, newest first.

    One dead feed must not cost the others, so failures are skipped rather than raised; the
    caller only sees an exception when nothing at all could be fetched.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=settings.crypto_news_lookback_hours)

    headlines, failures = [], []
    for url in settings.crypto_news_feeds:
        try:
            headlines.extend(_feed_items(url, cutoff))
        except (urllib.error.URLError, OSError, ET.ParseError, ValueError, TimeoutError) as error:
            failures.append(f"{url}: {error}")

    if failures and not headlines:
        raise OSError("; ".join(failures)[:300])
    for failure in failures:
        logger.warning("Crypto news feed unavailable: %s", failure)

    headlines.sort(key=lambda item: item["published"], reverse=True)
    for item in headlines:
        item["age_hours"] = max(0.0, (now - item["published"]).total_seconds() / 3600)
    return headlines[: settings.crypto_news_max_items]


def as_prompt_section() -> str:
    if not settings.crypto_news_enabled:
        return "Crypto headlines are disabled this cycle."

    try:
        headlines = recent_headlines()
    except (urllib.error.URLError, OSError, ET.ParseError, ValueError, TimeoutError) as error:
        logger.warning("Crypto headlines unavailable: %s", error)
        cached = _read_cache()
        if cached:
            return f"{cached}\n\n(Cached from an earlier fetch; the feeds were unreachable this cycle.)"
        return "Crypto headlines unavailable this cycle (fetch failed)."

    if not headlines:
        return f"No crypto headlines published in the last {settings.crypto_news_lookback_hours} hours."

    lines = [CAVEAT, ""]
    for item in headlines:
        age = f"{item['age_hours']:.0f}h ago" if item["age_hours"] >= 1 else "just now"
        lines.append(f"[{item['source']} · {age}] {item['title']}")
    section = "\n".join(lines)
    _write_cache(section)
    return section
