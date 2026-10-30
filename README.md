# Google ADK agents

Five agents built with Google's [Agent Development Kit](https://google.github.io/adk-docs/) on Gemini. Each folder is a standalone agent you can pick in `adk web`.

## The agents

**`research_crew`** is a multi-agent research pipeline built as an ADK `Workflow` graph. Three researchers search the web in parallel (latest developments, real-world impact, and risks and criticism), a research lead merges and dedupes what they found, and a summarizer writes a cited draft. A fact-checking critic then checks every claim and citation against the findings and returns a structured list of issues, and a reviser fixes them, for up to 2 rounds. The final report ends with the editorial review log.

**`interview_coach`** is a mock coding interviewer. It gives you a LeetCode-style problem, hands out up to 3 hints that each go a step further (a nudge, then the key technique, then the algorithm outline), and checks your Python by running it next to its own hidden reference solution on edge cases and a large input. Submissions run in a separate process with a 10-second limit (`harness.py`), so an infinite loop or a crash can't take down the agent. Solved problems are kept in user-scoped state, so they carry over to new sessions.

**`trip_planner`** builds and refines day-by-day itineraries. It uses web search for sights, prices and events, and the Open-Meteo API for real forecasts when the trip starts within 16 days. When you ask for changes it updates only the parts that changed.

**`code_explainer`** explains pasted code or a file from a link (GitHub file pages are converted to raw links). It covers what the code does, how it works in execution order, concepts a reader might not know, and any real bugs or edge cases.

**`spotify_curator`** builds playlists from your top tracks and liked songs and cleans up the ones you have: finding duplicates, renaming, and removing tracks. It signs in with Spotify's Authorization Code with PKCE flow, refreshes tokens on its own and backs off on rate limits. Spotify no longer gives development-mode apps recommendations, so the model picks songs from its own music knowledge and every pick is checked with search before it's added. Removing tracks asks for your confirmation first.

`shared.py` holds what the agents share: the model (`gemini-flash-lite-latest`, with retries because the free tier often returns 429s and 503s) and a DuckDuckGo `web_search` tool with backoff, used instead of ADK's built-in `google_search`, which needs a separate grounding quota.

## Running it

```bash
python -m venv .venv
./.venv/Scripts/activate   # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
cp .env.example .env       # then paste in your Gemini API key
adk web
```

Run `adk web` from the repo root so the agents can import `shared.py`, then open http://localhost:8000 and pick an agent.

### Spotify setup

The Spotify curator needs a one-time sign-in:

1. Create an app at the [Spotify developer dashboard](https://developer.spotify.com/dashboard) (the owner needs Spotify Premium). Add `http://127.0.0.1:8899/callback` as a redirect URI and pick Web API.
2. Put the app's Client ID in `.env` as `SPOTIFY_CLIENT_ID`.
3. Run `python -m spotify_curator.auth` from the repo root and sign in. Tokens are saved to `spotify_curator/.spotify_token.json`, which is gitignored.
