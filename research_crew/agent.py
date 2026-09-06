"""Research Crew: a 3-agent sequential pipeline built with Google ADK.

Pipeline: researcher -> summarizer -> critic
Each agent's output is stashed in session state (via output_key) and the
next agent's instruction pulls it back out with a {state_key} placeholder.
"""

from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.tools import google_search

MODEL = "gemini-flash-latest"

# 1. Researcher: the only tool allowed here is google_search — ADK does not
# let you mix google_search with other tools on the same LlmAgent.
researcher = LlmAgent(
    model=MODEL,
    name="researcher",
    description="Searches the web and gathers raw facts on the user's topic.",
    instruction=(
        "You are a research assistant. Use the google_search tool to find "
        "current, credible information on the user's topic. Gather 5-8 "
        "concrete facts, figures, or quotes with their sources. Output raw "
        "findings as a bullet list — do not summarize or editorialize yet."
    ),
    tools=[google_search],
    output_key="research_findings",
)

# 2. Summarizer: reads the researcher's output from session state.
summarizer = LlmAgent(
    model=MODEL,
    name="summarizer",
    description="Turns raw research findings into a tight summary.",
    instruction=(
        "You are a summarizer. Here are the raw research findings:\n\n"
        "{research_findings}\n\n"
        "Write a clear, well-organized summary (3-5 short paragraphs) of "
        "these findings for someone who knows nothing about the topic yet. "
        "Keep the most important facts and sources; drop redundant ones."
    ),
    output_key="summary",
)

# 3. Critic: reviews and polishes the summary into a final report.
critic = LlmAgent(
    model=MODEL,
    name="critic",
    description="Reviews the summary for accuracy/gaps and produces the final report.",
    instruction=(
        "You are an editor. Here is a draft summary:\n\n"
        "{summary}\n\n"
        "Critique it for unsupported claims, missing context, or unclear "
        "phrasing. Then produce a final, polished version that fixes those "
        "issues. Output only the final version, under a '## Final Report' "
        "heading, followed by a short '## What I Fixed' bullet list."
    ),
    output_key="final_report",
)

root_agent = SequentialAgent(
    name="research_crew",
    description="Researches a topic, summarizes it, then critiques and polishes the result.",
    sub_agents=[researcher, summarizer, critic],
)
