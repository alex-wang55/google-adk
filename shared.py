"""Model config and tools shared by the agents in this repo."""

import asyncio

from ddgs import DDGS
from ddgs.exceptions import DDGSException
from google.adk.models.google_llm import Gemini
from google.genai import types

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
