"""Research Crew: a multi-agent research pipeline built as an ADK Workflow graph.

Pipeline: 3 parallel researchers -> research_lead -> summarizer -> (critic <-> reviser loop) -> publish_report
"""

import asyncio
from datetime import date

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from google.adk.agents import LlmAgent
from google.adk.agents.context import Context
from google.adk.events.event import Event
from google.adk.models.google_llm import Gemini
from google.adk.workflow import START, JoinNode, Workflow
from google.genai import types
from pydantic import BaseModel

MAX_REVISIONS = 2
SEARCH_ATTEMPTS = 3

# Gemini calls don't retry by default; the free tier often returns 429/503 under load.
MODEL = Gemini(
    model="gemini-flash-lite-latest",
    retry_options=types.HttpRetryOptions(attempts=5, initial_delay=2),
)


# DuckDuckGo instead of ADK's google_search, which needs a separate (flaky) grounding quota.
async def web_search(query: str) -> str:
    """Searches the web and returns the top results.

    Args:
        query: The search query.

    Returns:
        Result titles, snippets, and source URLs.
    """
    error = None
    for attempt in range(SEARCH_ATTEMPTS):
        if attempt:
            await asyncio.sleep(2**attempt)
        try:
            results = await asyncio.to_thread(DDGS().text, query, max_results=8)
        except DDGSException as e:
            error = e
            continue
        return "\n\n".join(
            f"{r['title']}\n{r['body']}\nSource: {r['href']}" for r in results
        )
    return f"Search failed after {SEARCH_ATTEMPTS} attempts ({error})."


def _researcher(name: str, focus: str) -> LlmAgent:
    def instruction(_ctx) -> str:
        return (
            f"You are a research assistant focused on {focus}. Today is "
            f"{date.today():%B %d, %Y}. The user's message is the topic. Use "
            "the web_search tool to find current, credible information about "
            f"{focus} for that topic, and skip results that are about anything "
            "else. Gather 3-5 concrete facts, figures, or quotes as a bullet "
            "list. End every bullet with 'Source: ' followed by the exact URL "
            "from the search result it came from, never a site or "
            "publication name. Output only the bullet list. If your searches "
            "return nothing usable, output exactly 'No findings.'"
        )

    return LlmAgent(
        model=MODEL,
        name=name,
        description=f"Researches {focus}.",
        instruction=instruction,
        tools=[web_search],
    )


developments_researcher = _researcher(
    "developments_researcher", "the latest developments, key facts, and figures"
)
impact_researcher = _researcher(
    "impact_researcher",
    "who is involved and the real-world impact (economic, social, industry)",
)
challenges_researcher = _researcher(
    "challenges_researcher", "limitations, criticism, risks, and open questions"
)

research_lead = LlmAgent(
    model=MODEL,
    name="research_lead",
    description="Merges the parallel researchers' findings into one deduplicated set.",
    instruction=(
        "You are the research lead. The message is a JSON object mapping each "
        "researcher's name to its findings. Merge them into one bullet list "
        "grouped under the headings 'Developments', 'Impact', and "
        "'Challenges'. Remove duplicates, drop any finding without a source, "
        "and keep each source URL next to its finding. If a heading has no "
        "findings left, write 'No findings.' under it."
    ),
    output_key="temp:research_findings",
)

_SOURCES_FORMAT = (
    "Mark claims with numbered citations like [1] and end with a 'Sources' "
    "list where each line is the citation number followed by its URL, e.g. "
    "'[1] https://...'. Give each unique URL one number and reuse that "
    "number whenever you cite it again."
)

summarizer = LlmAgent(
    model=MODEL,
    name="summarizer",
    description="Turns the merged research findings into a draft summary.",
    instruction=(
        "You are a summarizer. The message contains research findings. Write "
        "a clear, well-organized summary (3-5 short paragraphs) for someone "
        "new to the topic, covering developments, impact, and challenges. "
        "State only what the findings support. If a heading has no findings, "
        "say in one sentence that no reliable information was found for it. "
        + _SOURCES_FORMAT
    ),
    output_key="temp:current_draft",
)


class Review(BaseModel):
    issues: list[str]


critic = LlmAgent(
    model=MODEL,
    name="critic",
    description="Fact-checks a draft against the research and lists anything that must be fixed.",
    instruction=(
        "You are a demanding fact-checking editor. The message is a draft "
        "summary. It must be based only on these research findings:\n\n"
        "{temp:research_findings}\n\n"
        "Check the draft sentence by sentence:\n"
        "- Every factual claim must be supported by a finding.\n"
        "- Every citation [n] must match entry [n] in the draft's Sources "
        "list, and that URL must be the source of the finding behind the "
        "claim.\n"
        "- Each URL appears only once in the Sources list.\n"
        "- No important finding may be left out.\n"
        "List each substantive problem in issues as a specific, actionable "
        "fix. Ignore minor style preferences. If there are no substantive "
        "problems, return an empty issues list."
    ),
    output_schema=Review,
)

reviser = LlmAgent(
    model=MODEL,
    name="reviser",
    description="Revises a draft to fix the editor's issues.",
    instruction=(
        "You are a writer revising a draft. The message contains the draft "
        "and the editor's feedback. Fix every point of feedback, using only "
        "these research findings as your factual source:\n\n"
        "{temp:research_findings}\n\n"
        + _SOURCES_FORMAT
        + " Output only the revised summary."
    ),
    output_key="temp:current_draft",
)


def route_review(ctx: Context, node_input: Review) -> str:
    ctx.state["temp:review_log"] = [
        *ctx.state.get("temp:review_log", []),
        node_input.issues,
    ]
    draft = ctx.state["temp:current_draft"]
    revisions = ctx.state.get("temp:revisions", 0)
    if not node_input.issues or revisions >= MAX_REVISIONS:
        ctx.route = "approve"
        return draft
    ctx.state["temp:revisions"] = revisions + 1
    ctx.route = "revise"
    feedback = "\n".join(f"- {issue}" for issue in node_input.issues)
    return f"DRAFT:\n{draft}\n\nEDITOR FEEDBACK:\n{feedback}"


def publish_report(ctx: Context, node_input: str):
    log = ctx.state.get("temp:review_log", [])
    lines = []
    for i, issues in enumerate(log, 1):
        if not issues:
            lines.append(f"- Round {i}: approved")
            continue
        lines.append(f"- Round {i}: {len(issues)} issue(s) flagged")
        lines.extend(f"  - {issue}" for issue in issues)
    if log and log[-1]:
        lines.append(
            f"- Stopped after {MAX_REVISIONS} revisions; the last round's "
            "issues were not fixed."
        )
    report = (
        f"## Final Report\n\n{node_input}\n\n## Editorial Review\n"
        + "\n".join(lines)
    )
    ctx.state["final_report"] = report
    yield Event(message=report, output=report)


root_agent = Workflow(
    name="research_crew",
    description=(
        "Researches a topic from three angles in parallel, drafts a summary, "
        "then revises it with a fact-checking editor until it's approved."
    ),
    edges=[
        (
            START,
            (developments_researcher, impact_researcher, challenges_researcher),
            JoinNode(name="join_research"),
            research_lead,
            summarizer,
            critic,
            route_review,
        ),
        (route_review, {"revise": reviser, "approve": publish_report}),
        (reviser, critic),
    ],
)
