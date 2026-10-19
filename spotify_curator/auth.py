"""Spotify sign-in for the curator agent, using the Authorization Code with PKCE flow.

Run once from the repo root:

    python -m spotify_curator.auth

It opens your browser to sign in, then saves the tokens next to this file in
.spotify_token.json (gitignored). The agent refreshes them on its own after that.
"""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse

import httpx
from dotenv import load_dotenv

TOKEN_FILE = Path(__file__).with_name(".spotify_token.json")
# Must match a redirect URI on the app in the Spotify dashboard. Spotify only allows
# plain http for 127.0.0.1, not localhost.
REDIRECT_URI = "http://127.0.0.1:8899/callback"
AUTHORIZE_URL = "https://accounts.spotify.com/authorize"
TOKEN_URL = "https://accounts.spotify.com/api/token"
SCOPES = " ".join([
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-top-read",
    "user-library-read",
])
SIGN_IN_HINT = "Run `python -m spotify_curator.auth` from the repo root to sign in."

_refresh_lock = asyncio.Lock()


class SpotifyError(Exception):
    """Something went wrong talking to Spotify. The message is safe to show the user."""


def _client_id() -> str:
    client_id = os.environ.get("SPOTIFY_CLIENT_ID")
    if not client_id:
        raise SpotifyError("SPOTIFY_CLIENT_ID isn't set. Add it to .env (see .env.example).")
    return client_id


def _save(token: dict, old_refresh_token: Optional[str] = None) -> dict:
    data = {
        "access_token": token["access_token"],
        # Spotify sometimes rotates the refresh token. Keep the old one if it didn't.
        "refresh_token": token.get("refresh_token") or old_refresh_token,
        "expires_at": time.time() + token["expires_in"],
    }
    TOKEN_FILE.write_text(json.dumps(data))
    os.chmod(TOKEN_FILE, 0o600)
    return data


async def get_access_token(force_refresh: bool = False) -> str:
    """Returns a valid access token, refreshing it first if it's about to expire."""
    if not TOKEN_FILE.exists():
        raise SpotifyError(f"Not signed in to Spotify yet. {SIGN_IN_HINT}")
    async with _refresh_lock:
        data = json.loads(TOKEN_FILE.read_text())
        if not force_refresh and data["expires_at"] - 60 > time.time():
            return data["access_token"]
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(TOKEN_URL, data={
                    "grant_type": "refresh_token",
                    "refresh_token": data["refresh_token"],
                    "client_id": _client_id(),
                })
        except httpx.HTTPError as e:
            raise SpotifyError(f"Couldn't reach Spotify to refresh the sign-in: {e}")
        if resp.status_code != 200:
            raise SpotifyError(f"The Spotify sign-in expired or was revoked. {SIGN_IN_HINT}")
        return _save(resp.json(), data["refresh_token"])["access_token"]


def login() -> None:
    """Signs in through the browser and saves the tokens."""
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
    client_id = _client_id()
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(16)
    url = AUTHORIZE_URL + "?" + urlencode({
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": REDIRECT_URI,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "scope": SCOPES,
        "state": state,
    })

    redirect = urlparse(REDIRECT_URI)
    result: dict = {}

    class CallbackHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            request = urlparse(self.path)
            if request.path != redirect.path:
                self.send_response(404)
                self.end_headers()
                return
            result.update({k: v[0] for k, v in parse_qs(request.query).items()})
            ok = "code" in result and result.get("state") == state
            body = "Signed in to Spotify. You can close this tab." if ok else "Sign-in failed. Check the terminal."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode())

        def log_message(self, *args):
            pass

    server = HTTPServer((redirect.hostname, redirect.port), CallbackHandler)
    print(f"Opening your browser to sign in to Spotify. If it doesn't open, go to:\n\n{url}\n")
    webbrowser.open(url)
    try:
        while "code" not in result and "error" not in result:
            server.handle_request()
    finally:
        server.server_close()

    if "error" in result:
        raise SystemExit(f"Spotify sign-in failed: {result['error']}")
    if result.get("state") != state:
        raise SystemExit("The sign-in response didn't match this request. Try again.")

    resp = httpx.post(TOKEN_URL, timeout=15, data={
        "grant_type": "authorization_code",
        "code": result["code"],
        "redirect_uri": REDIRECT_URI,
        "client_id": client_id,
        "code_verifier": verifier,
    })
    if resp.status_code != 200:
        raise SystemExit(f"Couldn't get a token from Spotify: {resp.text}")
    _save(resp.json())
    print(f"Signed in. Tokens saved to {TOKEN_FILE}")


if __name__ == "__main__":
    login()
