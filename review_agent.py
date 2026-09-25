"""Weekly review: reads the whole record and writes lessons the other agents carry into each run.

The other four agents improve only within a session. They see their journal and their scores, but
nobody reads the record as a whole and asks what it says about how they decide. This agent does
that once a week, and its output is the only channel by which one week's experience changes how
the next week is approached.

It holds no tools at all — not even account reads. Everything it needs is measurement the system
already produced, so there is nothing for it to fetch and nothing for it to break. Its output is
advice with citations; limits remain with the risk agent and the enforcement layer.

Run: `python review_agent.py` (scheduled weekly).
"""

import asyncio
import datetime as dt
import json
import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    os.environ.setdefault("DRY_RUN", "false")   # the review reads the live record, never trades

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage  # noqa: E402

import journal  # noqa: E402
import lessons  # noqa: E402
import research_agent  # noqa: E402
import risk_policy  # noqa: E402
import scorecard  # noqa: E402
from config import settings  # noqa: E402
from trade_logger import logger  # noqa: E402

SYSTEM_PROMPT = """You review a small automated crypto trading desk once a week and write the lessons its four
agents will carry into every run until the next review. You do not trade, hold no tools, and set no limits.

The desk is four LLM sessions: a research agent that finds candidates, a risk agent that sets the session's
limits, a sell agent that rules on exits, and a trading agent that places the buys. Each is given its own record.
Your job is the one none of them can do: read the whole record at once and say what it reveals about how they
decide.

What makes a lesson worth writing:
- It names a pattern in the evidence, not a market opinion. "Exits have been early: eight of nine sales kept
  rising" is a lesson. "Altcoins look strong" is not.
- It is actionable by the agent it addresses, in the next session, without new tools or wider limits.
- It cites the specific numbers behind it, so the agent can check it rather than take it on faith.
- It would change a decision. A lesson that merely affirms what the agent already does is noise, and noise in a
  prompt costs attention on every future run.

What to avoid:
- Do not recommend widening any limit. Order size, concentration, trade count, volatility scaling, cooldown and
  spread belong to the risk agent; the daily loss breaker belongs to the operator. A lesson that argues for more
  risk will be ignored, so spend the slot on something usable.
- Do not write lessons from a handful of decisions in one direction of the market. Say plainly when the sample
  is too small to conclude anything — "no lesson yet" is a legitimate and useful review.
- Do not repeat a lesson that is already being followed. Check the journal: if the agent is already doing it,
  the lesson is spent.

Write at most four lessons per agent and fewer where the evidence is thin. Prefer one sharp lesson with numbers
to four vague ones."""

REVIEW_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "lessons"],
    "properties": {
        "summary": {"type": "string",
                    "description": "Two to five sentences on what the record shows this week, for the operator."},
        "lessons": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["agent", "lesson", "evidence"],
                "properties": {
                    "agent": {"type": "string", "enum": list(lessons.AGENTS)},
                    "lesson": {"type": "string", "description": "What to do differently, addressed to that agent."},
                    "evidence": {"type": "string", "description": "The specific numbers this rests on."},
                },
            },
        },
    },
}


def _prompt() -> str:
    policies = risk_policy.history()[-6:]
    policy_lines = [
        f"  {p['timestamp'][:16]} account ${p['account_value']}: {p.get('stance', '?')} — order "
        f"{p['per_order_pct']:.0%}, position {p['max_position_pct']:.0%}, trades {p['max_trades_today']}, "
        f"vol {p['vol_target_pct']:.1%}, cooldown {p['cooldown_hours']:.0f}h" for p in policies] or ["  none yet"]

    return "\n".join([
        f"Review date: {dt.datetime.now().astimezone().date().isoformat()}",
        "",
        "SCORED DECISIONS (net of spread, against holding BTC over the same window):",
        scorecard.as_report(),
        "",
        "EXIT TIMING:",
        scorecard.exit_review(),
        "",
        "RESEARCH VERSUS EXECUTION:",
        research_agent.compare_report(),
        "",
        "CANDIDATES DECLINED, AND WHAT THEY DID NEXT:",
        scorecard.passes_review(10),
        "",
        "RISK POLICIES SET RECENTLY:",
        "\n".join(policy_lines),
        "",
        "RECENT SESSIONS, IN THE AGENTS' OWN WORDS:",
        journal.as_prompt_section(8),
        "",
        "LESSONS CURRENTLY IN FORCE (yours to keep, revise or drop — anything you do not restate expires):",
        "\n".join(f"  [{row['agent']}] {row['lesson']}" for row in lessons.load()) or "  none",
        "",
        "Write this week's lessons.",
    ])


async def run() -> dict:
    options = ClaudeAgentOptions(
        cwd=PROJECT_DIR,
        setting_sources=["project"],
        model=settings.claude_model,
        system_prompt=SYSTEM_PROMPT,
        allowed_tools=["StructuredOutput"],
        disallowed_tools=["Bash", "Read", "Write", "Edit", "WebSearch", "WebFetch"],
        permission_mode="dontAsk",
        max_turns=settings.review_max_turns,
        max_budget_usd=settings.review_budget_usd,
        thinking={"type": "adaptive"},
        effort=settings.review_effort,
        output_format={"type": "json_schema", "schema": REVIEW_SCHEMA},
    )

    raw, text = None, ""
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(_prompt())
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    raw, text = message.structured_output, (message.result or "").strip()
                    logger.info("Weekly review ended: %s after %d turns, cost $%.4f",
                                message.subtype, message.num_turns, message.total_cost_usd or 0.0)
    except Exception:
        logger.exception("Weekly review failed; the previous lessons stand.")
        return {}

    import agent
    if agent.usage_limited(text):
        logger.error("Weekly review stopped by the Claude usage limit; the previous lessons stand.")
        return {}

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        logger.warning("Weekly review returned no structured output; the previous lessons stand.")
        return {}

    kept = lessons.replace(raw.get("lessons"))
    logger.info("Weekly review: %d lesson(s) in force. %s", kept, (raw.get("summary") or "")[:600])
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(os.path.join(settings.log_dir, "reviews.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"at": dt.datetime.now(dt.timezone.utc).isoformat(), **raw}) + "\n")
    except OSError:
        pass
    return raw


if __name__ == "__main__":
    result = asyncio.run(run())
    if result:
        print(result.get("summary", ""))
        for row in result.get("lessons", []):
            print(f"\n[{row['agent']}] {row['lesson']}\n    evidence: {row['evidence']}")
    sys.exit(0 if result else 1)
