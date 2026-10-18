"""Interview Coach: a mock coding interviewer that runs your Python against real tests.

It gives you a LeetCode-style problem, hands out hints one level at a time, and
checks a submission by running it next to its own hidden reference solution on
edge cases plus a large input (see harness.py). The current problem and hint
count live in session state; solved problems are stored under the "user:"
prefix, so they carry over to new sessions.
"""

import ast
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Optional

from google.adk.agents import LlmAgent
from google.adk.agents.readonly_context import ReadonlyContext
from google.adk.tools import ToolContext

from shared import MODEL

HARNESS = Path(__file__).with_name("harness.py")
MARKER = "@@RESULT@@ "
TIME_LIMIT_SECONDS = 10
MAX_HINTS = 3
HINT_LEVELS = {
    1: "A nudge only: point at what to notice, or ask a guiding question. Don't name the technique or data structure.",
    2: "Name the key idea: the technique or data structure that makes this efficient, and why it fits. No steps, no code.",
    3: "Outline the algorithm in plain-English steps, with the target time and space complexity. Still no code.",
}
SOLVED_KEY = "user:solved"  # "user:" state is shared by all of this user's sessions


def start_problem(
    title: str, difficulty: str, pattern: str, function_name: str, tool_context: ToolContext
) -> str:
    """Records a new problem so hints and test runs can be tracked. Call it right before presenting the problem.

    Args:
        title: The problem name, e.g. "Two Sum".
        difficulty: "Easy", "Medium", or "Hard".
        pattern: The core technique, e.g. "hash map", "sliding window", "BFS".
        function_name: The method the candidate implements, e.g. "twoSum".

    Returns:
        A confirmation, plus a note if they've solved this problem before.
    """
    tool_context.state["problem"] = {
        "title": title,
        "difficulty": difficulty,
        "pattern": pattern,
        "function_name": function_name,
        "solved": False,
    }
    tool_context.state["hints_used"] = 0
    tool_context.state["attempts"] = 0
    solved_before = any(
        p["title"].lower() == title.lower() for p in tool_context.state.get(SOLVED_KEY, [])
    )
    if solved_before:
        return "Recorded. They've solved this one before, so mention that and offer a different problem."
    return "Recorded."


def get_hint(tool_context: ToolContext) -> str:
    """Unlocks the next hint level. Call it every time the candidate asks for a hint.

    Returns:
        Which hint level to give and what it may contain.
    """
    if not tool_context.state.get("problem"):
        return "There's no problem in progress. Give them a problem first."
    used = tool_context.state.get("hints_used", 0)
    if used >= MAX_HINTS:
        return (
            f"All {MAX_HINTS} hints are used. Don't give another hint. Offer to walk "
            "through the full solution if they want to stop here."
        )
    used += 1
    tool_context.state["hints_used"] = used
    return f"Give hint {used} of {MAX_HINTS}. {HINT_LEVELS[used]}"


def _parse_cases(text: str) -> list:
    """JSON, or Python literals as a fallback (models sometimes write None/True instead of null/true)."""
    try:
        cases = json.loads(text)
    except json.JSONDecodeError:
        cases = ast.literal_eval(text)
    if not isinstance(cases, list):
        raise ValueError("test_cases must be a list of argument lists")
    return cases


def _child_env() -> dict:
    """The current environment minus anything that looks like a secret (the Gemini key, for one)."""
    hidden = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in hidden)}


def _run_harness(job: dict) -> tuple[str, str, bool]:
    """Runs harness.py in its own process with a time limit. Returns (stdout, stderr, timed_out)."""
    with tempfile.TemporaryDirectory() as workdir:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(HARNESS)],
                input=json.dumps(job),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=TIME_LIMIT_SECONDS,
                cwd=workdir,
                env=_child_env(),
            )
            return proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            def as_text(data) -> str:
                if isinstance(data, bytes):
                    return data.decode("utf-8", "replace")
                return data or ""

            return as_text(e.stdout), as_text(e.stderr), True


def _summarize(events: list[dict], timed_out: bool, stderr: str, total: int) -> tuple[str, bool]:
    """Turns harness events into a report for the model. Returns (report, all_passed)."""
    for event in events:
        if event["kind"] == "load_error":
            if event["who"] == "solution":
                return f"Their code didn't run: {event['error']}", False
            return f"Your reference solution didn't load: {event['error']}. Fix it and call run_tests again.", False

    cases = [e for e in events if e["kind"] == "case"]
    passed = [c["case"] for c in cases if c["status"] == "pass"]
    lines = []
    for c in cases:
        name = "large input" if c["case"] == "large" else f"case {c['case']}"
        if c["status"] == "pass":
            if c["case"] == "large":
                lines.append(f"large input: passed in {c['ms']} ms")
            continue
        if c["status"] == "fail":
            lines.append(f"{name}: FAILED. input {c['input']} | expected {c['expected']} | got {c['got']}")
        elif c["status"] == "error":
            lines.append(f"{name}: CRASHED. {c['error']} | input {c['input']}")
        elif c["status"] == "reference_error":
            lines.append(f"{name}: YOUR REFERENCE crashed ({c['error']}). Not their fault: fix it and rerun.")
        elif c["status"] == "bad_test":
            lines.append(f"{name}: BAD TEST ({c['error']}). Not their fault: fix it and rerun.")
        if c.get("printed"):
            lines.append(f"  their print output: {c['printed']}")

    finished = any(e["kind"] == "done" for e in events)
    if timed_out or not finished:
        progress = [e for e in events if e["kind"] == "progress"]
        stuck = progress[-1] if progress else None
        if stuck and stuck["stage"] == "solution":
            name = "the large input" if stuck["case"] == "large" else f"case {stuck['case']}"
            lines.append(
                f"Their code was stopped on {name} after {TIME_LIMIT_SECONDS}s: an infinite loop, "
                "or too slow for the input size."
            )
        elif stuck:
            lines.append(f"Your reference timed out on case {stuck['case']}. Fix it or shrink the test, then rerun.")
        else:
            lines.append(f"The test run failed to start. stderr: {stderr[-500:]}")

    all_passed = finished and not timed_out and total > 0 and len(passed) == total
    header = f"Passed {len(passed)}/{total} test cases."
    if len(cases) < total:
        header += f" ({total - len(cases)} didn't finish.)"
    return "\n".join([header, *lines]), all_passed


async def run_tests(
    solution_code: str,
    reference_code: str,
    test_cases: str,
    tool_context: ToolContext,
    order: str = "exact",
    arg_kinds: Optional[list[str]] = None,
    in_place: bool = False,
    large_input: Optional[str] = None,
) -> str:
    """Runs the candidate's code and your reference solution on the same inputs and compares the results.

    Args:
        solution_code: The candidate's code, exactly as they wrote it. Never fix it first.
        reference_code: Your own correct solution, never shown to the candidate. Use the same
            function or Solution method name as theirs.
        test_cases: A JSON list where each item is the argument list for one call, e.g.
            '[[[2, 7, 11, 15], 9], [[3, 3], 6]]' for twoSum(nums, target). Expected outputs
            come from running your reference, so don't include them.
        order: "exact" (default); "any" if the order of the returned list doesn't matter
            (Two Sum indices, Top K Frequent); "any_nested" if no list order matters at any
            level (Group Anagrams, 3Sum).
        arg_kinds: One entry per argument, only needed for linked lists or trees: "plain",
            "linked_list" (a list of values, or {"values": [...], "cycle_pos": k} for a cycle),
            "tree" (LeetCode level order with null for gaps), or "node_in_tree" (a value; passes
            the node with that value from the tree argument before it). Returned ListNode and
            TreeNode values are converted back to lists automatically.
        in_place: True when the problem says to modify the input and return nothing (Rotate
            Image, Reorder List). Compares the first argument after the call instead.
        large_input: Optional Python expression that builds one argument list at the
            constraints' max size, to catch solutions that are too slow, e.g.
            "[[random.randint(-10**9, 10**9) for _ in range(10**5)], 7]". random is seeded.

    Returns:
        Which tests passed, with input, expected, and actual output for each failure.
    """
    problem = tool_context.state.get("problem")
    if not problem:
        return "There's no problem in progress. Call start_problem first."
    try:
        cases = _parse_cases(test_cases)
    except (ValueError, SyntaxError) as e:
        return f"test_cases isn't a valid list ({e}). Fix the format and call run_tests again."

    job = {
        "solution": solution_code,
        "reference": reference_code,
        "function_name": problem["function_name"],
        "cases": cases,
        "order": order,
        "arg_kinds": arg_kinds or [],
        "in_place": in_place,
        "large_input": large_input,
    }
    stdout, stderr, timed_out = await asyncio.to_thread(_run_harness, job)
    events = [
        json.loads(line[len(MARKER):]) for line in stdout.splitlines() if line.startswith(MARKER)
    ]
    total = len(cases) + (1 if large_input else 0)
    report, all_passed = _summarize(events, timed_out, stderr, total)

    tool_context.state["attempts"] = tool_context.state.get("attempts", 0) + 1
    if all_passed and not problem.get("solved"):
        tool_context.state["problem"] = {**problem, "solved": True}
        history = [
            p for p in tool_context.state.get(SOLVED_KEY, [])
            if p["title"].lower() != problem["title"].lower()
        ]
        history.append({
            "title": problem["title"],
            "difficulty": problem["difficulty"],
            "pattern": problem["pattern"],
            "hints": tool_context.state.get("hints_used", 0),
            "attempts": tool_context.state["attempts"],
            "date": date.today().isoformat(),
        })
        tool_context.state[SOLVED_KEY] = history
        report += "\nAll passed. Recorded as solved."
    return report


COACH_PROMPT = """You are a coding interviewer helping the candidate practice LeetCode / NeetCode 150 style problems in Python. Be casual, direct, and honest, like a friendly senior engineer running a mock interview. Don't use em dashes.

Starting a problem:
- If they don't say what they want, pick an Easy or Medium problem they haven't solved yet (see the solved list below), or ask for a topic.
- Call start_problem, then present the problem statement, two examples with short explanations, the constraints, and the exact signature to implement, LeetCode style (class Solution with the method and type hints).
- Prefer problems whose inputs and outputs are plain data, linked lists, or binary trees, since run_tests can check those. For design problems (implement a class, like LRU Cache), review their code by reading it instead of running tests.
- Then let them think. Invite them to talk through their approach before coding.

While they work:
- Never reveal the solution, the key technique, or code unless they give up or explicitly ask for the answer.
- When they ask for a hint, call get_hint and give exactly one hint at the level it returns. Don't slip extra information into it.
- When they describe an approach, react like an interviewer: ask about time and space complexity, edge cases, or whether there's something faster. Don't solve it for them.

When they submit code:
- Call run_tests with their code exactly as written and your own correct reference solution, which you never show them.
- Write 6 to 10 test cases: the examples, plus edge cases like empty or minimal input, a single element, duplicates, negatives, and the extreme values allowed by the constraints.
- When efficiency matters, also pass large_input at the constraints' maximum size.
- Then give feedback: which tests failed and why, pointing at the specific line or logic; the time and space complexity of their approach; and one or two concrete improvements. If everything passed, say so, state the complexity, and mention a better approach only if one exists.
- If a failure comes from your reference or a bad test, fix it and rerun instead of blaming them.
- Point them at the bug and let them fix it. Don't rewrite their solution unless they ask.

After they solve it, offer a follow-up variation or the next problem."""


def instruction(ctx: ReadonlyContext) -> str:
    """Rebuilt every turn, so the model always sees the live problem state."""
    state = ctx.state
    problem = state.get("problem")
    if problem:
        status = (
            f"Current problem: {problem['title']} ({problem['difficulty']}, {problem['pattern']}), "
            f"method `{problem['function_name']}`. Hints used: {state.get('hints_used', 0)}/{MAX_HINTS}. "
            f"Test runs so far: {state.get('attempts', 0)}."
        )
        if problem.get("solved"):
            status += " They've solved it."
    else:
        status = "No problem in progress."
    solved = state.get(SOLVED_KEY, [])
    if solved:
        history = "; ".join(f"{p['title']} ({p['difficulty']}, {p['hints']} hints)" for p in solved[-50:])
        status += f"\nSolved in past sessions: {history}"
    return f"{COACH_PROMPT}\n\n{status}"


root_agent = LlmAgent(
    model=MODEL,
    name="interview_coach",
    description="Mock coding interviewer: gives LeetCode-style problems, hands out hints one level at a time, and runs your Python against real tests.",
    instruction=instruction,
    tools=[start_problem, get_hint, run_tests],
)
