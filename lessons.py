"""Lessons the weekly review writes for the other agents, and the prompts that read them.

Each run starts cold: the agents see their journal and their scores, but nothing distils what
those records mean. The review agent reads the whole record once a week and writes a small number
of lessons, each tied to evidence, which then appear in the relevant agent's prompt every run.

Three properties keep this from becoming prompt rot. A lesson must cite its evidence, so it can
be checked rather than believed. Lessons expire, so a conclusion drawn from one week's tape does
not govern forever unless the next review restates it. And they are advice only: a lesson cannot
widen a limit, because limits live in the risk policy and the enforcement layer, and an agent
that could write its own constraints would have none.
"""

import datetime as dt
import json
import os

from config import settings

LESSONS_PATH = os.path.join(settings.log_dir, "lessons.json")
AGENTS = ("research", "risk", "sell", "trader")
MAX_PER_AGENT = 4
EXPIRY_DAYS = 21


def load() -> list:
    try:
        with open(LESSONS_PATH, encoding="utf-8") as handle:
            stored = json.load(handle)
        rows = stored if isinstance(stored, list) else []
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=EXPIRY_DAYS)
    return [row for row in rows
            if isinstance(row, dict) and row.get("at") and dt.datetime.fromisoformat(row["at"]) >= cutoff]


def replace(rows) -> int:
    """Store this review's lessons, replacing the previous set. Returns how many were kept."""
    stamped = dt.datetime.now(dt.timezone.utc).isoformat()
    kept, counts = [], {agent: 0 for agent in AGENTS}
    for row in rows or []:
        agent = str((row or {}).get("agent", "")).strip().lower()
        lesson = str((row or {}).get("lesson", "")).strip()
        evidence = str((row or {}).get("evidence", "")).strip()
        if agent not in AGENTS or len(lesson) < 20 or len(evidence) < 10:
            continue
        if counts[agent] >= MAX_PER_AGENT:
            continue
        counts[agent] += 1
        kept.append({"agent": agent, "lesson": lesson[:400], "evidence": evidence[:300], "at": stamped})
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        tmp_path = f"{LESSONS_PATH}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(kept, handle, indent=2)
        os.replace(tmp_path, LESSONS_PATH)
    except OSError:
        return 0
    return len(kept)


def as_prompt_section(agent: str) -> str:
    rows = [row for row in load() if row["agent"] == agent]
    if not rows:
        return ""
    written = rows[0]["at"][:10]
    lines = [f"Lessons from your own record, written by the weekly review on {written} and expiring "
             f"{EXPIRY_DAYS} days after that. Each cites the evidence behind it, so weigh it as evidence rather "
             "than instruction — if this session's data contradicts a lesson, say so and act on what you see. "
             "Lessons never widen a limit; limits come from the risk policy and are enforced in code."]
    for row in rows:
        lines.append(f"  - {row['lesson']}")
        lines.append(f"    evidence: {row['evidence']}")
    return "\n".join(lines)
