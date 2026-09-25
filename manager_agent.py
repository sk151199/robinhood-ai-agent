"""Manager: reads how the desk performs and changes the desk itself, without asking first.

Every other agent decides within a run. This one decides what the runs are — it edits prompts,
limits, thresholds, schedules and code, and writes new agents. The operator chose deliberately
that nothing is off limits to it, including the enforcement layer in order_gate.py, the tool
allowlists in rh_tools.py, and the daily loss breaker. That choice is recorded here because it
inverts the property every other part of this system was built around: elsewhere an agent cannot
widen its own constraints, and here one can.

What that leaves as the only real protection is reversibility, so the design spends everything on
it. Each run commits the tree first and records the SHA. Afterwards every module is imported in a
fresh interpreter, and a tree that no longer imports is reset to the snapshot — a manager that
breaks the desk cannot leave it broken. Changes it makes to the loss breaker, the gate or the
allowlists are applied like any other, and flagged loudly in the log so a human reading it later
can find the moment the rules changed.

Run: `python manager_agent.py` (scheduled), or `--dry` to propose without touching anything.
"""

import asyncio
import datetime as dt
import json
import os
import subprocess
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if __name__ == "__main__":
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, PROJECT_DIR)
    os.environ.setdefault("DRY_RUN", "true")   # the manager never trades; this only guards imports

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient, ResultMessage  # noqa: E402

import journal  # noqa: E402
import lessons  # noqa: E402
import risk_policy  # noqa: E402
import scorecard  # noqa: E402
from config import settings  # noqa: E402
from trade_logger import logger  # noqa: E402

CHANGES_PATH = os.path.join(settings.log_dir, "manager_changes.jsonl")

# Touching one of these changes what the desk is allowed to do, rather than how well it does it.
# The manager may edit them; naming them here is what makes that visible afterwards.
GUARDRAIL_FILES = ("order_gate.py", "rh_tools.py", "risk_policy.py", "risk_manager.py", "config.py", ".env")

SYSTEM_PROMPT = """You manage a small automated crypto trading desk and you change it directly. You are not
advising an operator who will review your work; what you write is what runs next.

The desk is five Claude sessions over a code-enforced order gate: research finds candidates, risk sets the
session's limits, sell rules on exits, trading places buys, and a weekly review writes lessons the others carry.
Code around them gathers market data, watches positions between runs, and adjudicates every order.

You hold file tools over the project. You may edit any file, including the enforcement layer (order_gate.py),
the tool allowlists (rh_tools.py), the risk policy, the configuration and the operator's daily loss breaker.
You may write new agents and wire them in. The operator has explicitly granted this.

Because nothing refuses you, the discipline has to come from you:

- Change what the evidence supports, and nothing else. Every edit names the measurement that motivated it. A
  change you cannot tie to a number in the record is a guess, and guesses here are expensive.
- Prefer the smallest change that tests the idea. One agent's effort level is recoverable; a rewritten gate is
  a new system whose behaviour you cannot predict from the record you just read.
- A limit you widen is a limit that was protecting something. If you widen one, say what it was protecting and
  why that protection is no longer worth its cost. "It would allow more trades" is not a reason; the desk is
  scored against holding BTC, and more trades has so far made that comparison worse.
- The account is small and spreads are near 2%. Anything that increases turnover has to clear roughly 4% of
  price movement per round trip before it earns anything.
- Leave the desk importable. Your changes are validated by importing every module afterwards, and a tree that
  does not import is reset — an over-ambitious edit costs the whole run, including the good parts.
- Doing nothing is a legitimate outcome. If the record is too thin to conclude anything, say so and change
  nothing. The desk has few closed trades and no demonstrated edge; optimising noise makes it worse.

Report every change you made, the evidence, and what you expect to see if you were right. Say plainly which
changes touch what the desk is permitted to do rather than how well it does it."""

REPORT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "changes", "expected_effect"],
    "properties": {
        "summary": {"type": "string", "description": "What you changed this run and why, for the operator."},
        "expected_effect": {"type": "string",
                            "description": "What should show up in the record if these changes were right, and by when."},
        "changes": {
            "type": "array",
            "maxItems": 20,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["file", "what", "evidence", "widens_a_limit"],
                "properties": {
                    "file": {"type": "string", "description": "Path relative to the project."},
                    "what": {"type": "string", "description": "The change, concretely."},
                    "evidence": {"type": "string", "description": "The measurement this rests on."},
                    "widens_a_limit": {"type": "boolean",
                                       "description": "True if this lets the desk take more risk or act more freely."},
                },
            },
        },
    },
}


def _git(*args) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT_DIR, capture_output=True, text=True).stdout.strip()


def snapshot() -> str:
    """Commit the tree so any change this run makes can be undone. Returns the SHA to reset to."""
    _git("add", "-A")
    _git("commit", "-q", "-m", f"Manager snapshot {dt.datetime.now().astimezone().isoformat(timespec='seconds')}",
         "--allow-empty")
    return _git("rev-parse", "HEAD")


def validate() -> tuple:
    """Import every module in a fresh interpreter. Returns (ok, first error)."""
    modules = sorted(f[:-3] for f in os.listdir(PROJECT_DIR)
                     if f.endswith(".py") and f != "manager_agent.py")
    code = "import os;os.environ.setdefault('DRY_RUN','true')\n" + \
           "\n".join(f"import {m}" for m in modules)
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_DIR, capture_output=True, text=True, timeout=180)
    return done.returncode == 0, (done.stderr or "").strip()[-1200:]


def revert(sha: str):
    _git("reset", "--hard", sha)
    logger.warning("Manager changes reverted to %s.", sha[:8])


def changed_files(sha: str) -> list:
    out = _git("diff", "--name-only", sha)
    return [line for line in out.splitlines() if line.strip()]


def _prompt() -> str:
    cycles = journal.recent(12)
    costs = [f"  {c['timestamp'][:16]} {c['mode']}: ${c.get('cost_usd', 0):.4f}, "
             f"{len(c.get('orders') or [])} order(s)" for c in cycles] or ["  none yet"]
    policies = risk_policy.history()[-6:]
    policy_lines = [f"  {p['timestamp'][:16]} account ${p['account_value']}: order {p['per_order_pct']:.0%}, "
                    f"position {p['max_position_pct']:.0%}, trades {p['max_trades_today']}, "
                    f"spread {p.get('max_spread_pct', 0):.2%}" for p in policies] or ["  none yet"]
    settings_lines = [f"  {k} = {v!r}" for k, v in sorted(vars(settings).items())
                      if not k.startswith("_") and isinstance(v, (str, int, float, bool))]

    return "\n".join([
        f"Run date: {dt.datetime.now().astimezone().isoformat(timespec='minutes')}",
        "",
        "SCORED DECISIONS (net of spread, against holding BTC over the same window):",
        scorecard.as_report(),
        "",
        "EXIT TIMING:",
        scorecard.exit_review(),
        "",
        "WHAT EACH CYCLE COST AND PRODUCED:",
        "\n".join(costs),
        "",
        "RISK POLICIES SET RECENTLY:",
        "\n".join(policy_lines),
        "",
        "LESSONS CURRENTLY IN FORCE:",
        "\n".join(f"  [{row['agent']}] {row['lesson']}" for row in lessons.load()) or "  none",
        "",
        "RECENT SESSIONS, IN THE AGENTS' OWN WORDS:",
        journal.as_prompt_section(6),
        "",
        "CURRENT CONFIGURATION:",
        "\n".join(settings_lines),
        "",
        "Read the code you intend to change before changing it. Then make the changes and report them.",
    ])


def allowed_tools() -> list:
    names = ["Read", "Write", "Edit", "Glob", "Grep", "StructuredOutput"]
    if settings.manager_allow_bash:
        names.append("Bash")
    return names


async def run(dry: bool = False) -> dict:
    if not settings.manager_agent_enabled:
        logger.info("Manager agent is disabled.")
        return {}

    before = snapshot()
    logger.info("Manager run starting; tree snapshotted at %s.", before[:8])

    options = ClaudeAgentOptions(
        cwd=PROJECT_DIR,
        setting_sources=["project"],
        model=settings.claude_model,
        system_prompt=SYSTEM_PROMPT + ("\n\nThis run is a rehearsal: describe the changes you would make, but do "
                                       "not write any file." if dry else ""),
        allowed_tools=["Read", "Glob", "Grep", "StructuredOutput"] if dry else allowed_tools(),
        disallowed_tools=["WebSearch", "WebFetch"] + (["Write", "Edit", "Bash"] if dry else []),
        permission_mode="dontAsk",
        max_turns=settings.manager_max_turns,
        max_budget_usd=settings.manager_budget_usd,
        thinking={"type": "adaptive"},
        effort=settings.manager_effort,
        output_format={"type": "json_schema", "schema": REPORT_SCHEMA},
    )

    raw, text = None, ""
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(_prompt())
            async for message in client.receive_response():
                if isinstance(message, ResultMessage):
                    raw, text = message.structured_output, (message.result or "").strip()
                    logger.info("Manager ended: %s after %d turns, cost $%.4f",
                                message.subtype, message.num_turns, message.total_cost_usd or 0.0)
    except Exception:
        logger.exception("Manager run failed.")
        revert(before)
        return {}

    import agent
    if agent.usage_limited(text):
        logger.error("Manager stopped by the Claude usage limit; reverting anything half-written.")
        revert(before)
        return {}

    touched = changed_files(before)
    if dry:
        if touched:
            logger.warning("Rehearsal wrote files despite being told not to: %s. Reverting.", ", ".join(touched))
            revert(before)
        touched = []
    elif touched:
        ok, error = validate()
        if not ok:
            logger.error("Manager left the desk unimportable; reverting. First error:\n%s", error)
            revert(before)
            touched = []
        else:
            guardrails = [f for f in touched if f in GUARDRAIL_FILES]
            if guardrails:
                logger.warning("MANAGER CHANGED WHAT THE DESK IS PERMITTED TO DO: %s. "
                               "Snapshot before the change is %s.", ", ".join(guardrails), before[:8])
            logger.info("Manager changed %d file(s): %s", len(touched), ", ".join(touched))

    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if not isinstance(raw, dict):
        logger.warning("Manager returned no structured report.")
        raw = {}

    record = {
        "at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "dry_run": dry,
        "snapshot_before": before,
        "files_changed": touched,
        "guardrail_files_changed": [f for f in touched if f in GUARDRAIL_FILES],
        **raw,
    }
    try:
        os.makedirs(settings.log_dir, exist_ok=True)
        with open(CHANGES_PATH, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass

    logger.info("Manager: %s", (raw.get("summary") or "no summary")[:800])
    return record


def main() -> int:
    dry = "--dry" in sys.argv
    record = asyncio.run(run(dry=dry))
    if not record:
        return 1
    print(json.dumps({k: record[k] for k in ("summary", "files_changed", "guardrail_files_changed",
                                             "expected_effect") if k in record}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
