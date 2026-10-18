"""Code Explainer: a conversational agent that walks through code, pasted or fetched from a link."""

import re

import httpx
from google.adk.agents import LlmAgent

from shared import MODEL

MAX_CODE_CHARS = 60_000
_GITHUB_FILE_PAGE = re.compile(r"https://github\.com/([^/]+)/([^/]+)/blob/(.+)")


async def fetch_code(url: str) -> str:
    """Downloads a source file from a public https:// link.

    GitHub file page links are converted to their raw file automatically.

    Args:
        url: An https:// link to a source file, e.g. a GitHub file page.

    Returns:
        The file's contents, or an error message.
    """
    if not url.startswith("https://"):
        return "Only https:// links are supported."
    url = _GITHUB_FILE_PAGE.sub(r"https://raw.githubusercontent.com/\1/\2/\3", url)
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as e:
        return f"Couldn't fetch {url}: {e}"
    if len(resp.text) > MAX_CODE_CHARS:
        return (
            resp.text[:MAX_CODE_CHARS]
            + f"\n\n[Truncated: showing the first {MAX_CODE_CHARS} of "
            f"{len(resp.text)} characters.]"
        )
    return resp.text


root_agent = LlmAgent(
    model=MODEL,
    name="code_explainer",
    description="Explains what a piece of code does and how it works, from pasted code or a link.",
    instruction="""You are a patient senior engineer who explains code clearly.

The user will paste code or share a link to a file. If they share an https link, call fetch_code to get the file. If they mention their experience level, match it. Otherwise, assume they know basic programming but not this particular language or library.

Explain the code in these sections:
1. What it does: two or three sentences on its purpose and what it produces.
2. How it works: walk through it in execution order, naming the specific functions, classes, and lines you're describing. Quote short snippets when that helps.
3. Key concepts: language features, libraries, or patterns a reader might not know, a sentence or two each.
4. Watch out for: real bugs, edge cases, and security or performance problems. Leave this section out if there aren't any. Never invent problems.

Describe only what the code actually does. If it depends on code or configuration you can't see, say what you're assuming. Answer follow-up questions about specific lines directly, without repeating the whole explanation.""",
    tools=[fetch_code],
)
