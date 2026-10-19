"""Spotify Curator: builds new playlists from your taste and cleans up the ones you have.

Setup, once:
1. Create an app at https://developer.spotify.com/dashboard (the owner needs Spotify
   Premium). Add http://127.0.0.1:8899/callback as a redirect URI and pick Web API.
2. Put the app's Client ID in .env as SPOTIFY_CLIENT_ID.
3. Run `python -m spotify_curator.auth` from the repo root and sign in.

Spotify's API no longer gives development-mode apps recommendations or audio
features, and only returns the tracks of playlists you own or collaborate on. So
the model picks songs from its own music knowledge and every pick goes through
search before it's added. Removing tracks asks for approval in adk web first.
"""

import asyncio
import re
from typing import Optional

import httpx
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from shared import MODEL

from .auth import SIGN_IN_HINT, SpotifyError, get_access_token

API = "https://api.spotify.com/v1"
PAGE_SIZE = 50
TRACKS_PER_CALL = 200  # keeps a big playlist from flooding the model's context
MAX_SCAN = 2000  # tracks read when looking for duplicates
MAX_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 30
_URI = re.compile(r"^spotify:(track|episode):[0-9A-Za-z]{22}$")


async def _api(method: str, path: str, *, params: Optional[dict] = None, json: Optional[dict] = None) -> dict:
    """Calls the Web API. Refreshes the token once on a 401 and backs off on 429s."""
    url = path if path.startswith("https://") else API + path
    force_refresh = False
    for attempt in range(MAX_RETRIES + 1):
        token = await get_access_token(force_refresh)
        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.request(
                    method, url, params=params, json=json,
                    headers={"Authorization": f"Bearer {token}"},
                )
        except httpx.HTTPError as e:
            raise SpotifyError(f"Couldn't reach Spotify: {e}")
        if resp.status_code == 401 and not force_refresh:
            force_refresh = True
            continue
        if resp.status_code == 429 and attempt < MAX_RETRIES:
            wait = float(resp.headers.get("Retry-After", 2))
            if wait > MAX_RETRY_WAIT_SECONDS:
                raise SpotifyError(f"Hit Spotify's rate limit. It says to wait {wait:.0f}s, so try again later.")
            await asyncio.sleep(wait)
            continue
        if resp.status_code == 401:
            raise SpotifyError(f"Spotify rejected the sign-in. {SIGN_IN_HINT}")
        if resp.status_code >= 400:
            try:
                message = resp.json()["error"]["message"]
            except (ValueError, KeyError, TypeError):
                message = resp.text[:200]
            raise SpotifyError(f"Spotify returned {resp.status_code}: {message}")
        return resp.json() if resp.content else {}
    raise SpotifyError("Spotify kept rate limiting the request. Try again in a bit.")


def _track(item: dict) -> dict:
    album = item.get("album") or {}
    return {
        "uri": item["uri"],
        "name": item["name"],
        "artists": ", ".join(a["name"] for a in item.get("artists") or []),
        "album": album.get("name"),
        "year": (album.get("release_date") or "")[:4] or None,
    }


def _entry_item(entry: dict) -> Optional[dict]:
    # Development-mode apps get "item"; older and extended-quota apps still get "track".
    return entry.get("item") or entry.get("track")


async def _read_playlist(playlist_id: str, offset: int, limit: int) -> tuple[list[dict], int, int]:
    """Returns (tracks with positions, playlist total, entries read)."""
    tracks, read, total = [], 0, 0
    while read < limit:
        data = await _api(
            "GET", f"/playlists/{playlist_id}/items",
            params={"limit": PAGE_SIZE, "offset": offset + read},
        )
        total = data.get("total", 0)
        page = data.get("items") or []
        for i, entry in enumerate(page):
            item = _entry_item(entry)
            if item and item.get("uri"):
                tracks.append({"position": offset + read + i, **_track(item)})
        read += len(page)
        if not page or not data.get("next"):
            break
    return tracks, total, read


async def list_my_playlists() -> dict:
    """Lists the user's playlists: ones they made, collaborate on, or follow.

    Returns:
        Each playlist's id, name, track count, and whether the user can edit it.
        Only editable playlists can have their tracks read or changed.
    """
    try:
        me = (await _api("GET", "/me"))["id"]
        playlists, url, params = [], "/me/playlists", {"limit": PAGE_SIZE}
        while url:
            data = await _api("GET", url, params=params)
            for p in data.get("items") or []:
                if not p:
                    continue
                counts = p.get("items") or p.get("tracks") or {}
                playlists.append({
                    "id": p["id"],
                    "name": p["name"],
                    "tracks": counts.get("total"),
                    "editable": p["owner"]["id"] == me or bool(p.get("collaborative")),
                    "public": p.get("public"),
                })
            url, params = data.get("next"), None
    except SpotifyError as e:
        return {"error": str(e)}
    return {"playlists": playlists}


async def get_playlist_tracks(playlist_id: str, offset: int = 0) -> dict:
    """Gets tracks from a playlist the user owns or collaborates on, up to 200 at a time.

    Args:
        playlist_id: The playlist id from list_my_playlists.
        offset: Where to start reading, for playlists with more than 200 tracks.

    Returns:
        The tracks (position, uri, name, artists, album, year), the playlist's
        total track count, and next_offset if there are more to read.
    """
    try:
        tracks, total, read = await _read_playlist(playlist_id, offset, TRACKS_PER_CALL)
    except SpotifyError as e:
        return {"error": f"{e} (Spotify only shares the tracks of playlists the user owns or collaborates on.)"}
    result = {"total": total, "tracks": tracks}
    if offset + read < total:
        result["next_offset"] = offset + read
    return result


async def get_top_tracks(time_range: str = "medium_term") -> dict:
    """Gets the user's 50 most-played tracks, to learn their taste.

    Args:
        time_range: "short_term" (about 4 weeks), "medium_term" (about 6 months),
            or "long_term" (about a year).

    Returns:
        The tracks, most played first.
    """
    if time_range not in ("short_term", "medium_term", "long_term"):
        return {"error": "time_range must be short_term, medium_term, or long_term."}
    try:
        data = await _api("GET", "/me/top/tracks", params={"limit": 50, "time_range": time_range})
    except SpotifyError as e:
        return {"error": str(e)}
    return {"tracks": [_track(t) for t in data.get("items") or []]}


async def get_liked_songs(offset: int = 0) -> dict:
    """Gets the user's Liked Songs, newest first, up to 200 at a time.

    Args:
        offset: Where to start reading, to page through a long library.

    Returns:
        The tracks, the total number of liked songs, and next_offset if there are more.
    """
    tracks, read, total = [], 0, 0
    try:
        while read < TRACKS_PER_CALL:
            data = await _api("GET", "/me/tracks", params={"limit": PAGE_SIZE, "offset": offset + read})
            total = data.get("total", 0)
            page = data.get("items") or []
            tracks += [_track(item) for item in map(_entry_item, page) if item]
            read += len(page)
            if not page or not data.get("next"):
                break
    except SpotifyError as e:
        return {"error": str(e)}
    result = {"total": total, "tracks": tracks}
    if offset + read < total:
        result["next_offset"] = offset + read
    return result


async def search_tracks(title: str, artist: str = "") -> dict:
    """Searches Spotify for a song. Use it to get the URI of every song before adding it.

    Args:
        title: The song title, e.g. "Redbone".
        artist: The artist, e.g. "Childish Gambino". Leave empty if unknown.

    Returns:
        Up to 5 matches with uri, name, artists, album, and year. Check that the
        title and artist match the song you meant before using one.
    """
    queries = [f"track:{title} artist:{artist}", f"{title} {artist}"] if artist else [title]
    try:
        for query in queries:
            data = await _api("GET", "/search", params={"q": query, "type": "track", "limit": 5})
            items = (data.get("tracks") or {}).get("items") or []
            if items:
                return {"tracks": [_track(t) for t in items if t]}
    except SpotifyError as e:
        return {"error": str(e)}
    return {"tracks": [], "note": "No matches. Skip this song or try a different spelling."}


def _song_key(track: dict) -> tuple[str, str]:
    """Same song across releases: 'Song (Remastered 2011)' and 'Song - Live' both become 'song'."""
    name = re.sub(r"[\(\[].*?[\)\]]", "", track["name"].lower())
    name = name.split(" - ")[0].strip() or track["name"].lower()
    first_artist = track["artists"].split(", ")[0].lower()
    return name, first_artist


async def find_duplicates(playlist_id: str) -> dict:
    """Finds repeated songs in a playlist the user owns or collaborates on.

    Args:
        playlist_id: The playlist id from list_my_playlists.

    Returns:
        exact: the same track added more than once (same uri), with positions.
        other_versions: the same song from different releases (remaster, live,
            single vs album), each version with its uri, album, year, and positions.
    """
    try:
        tracks, total, _ = await _read_playlist(playlist_id, 0, MAX_SCAN)
    except SpotifyError as e:
        return {"error": str(e)}

    by_uri: dict[str, list[dict]] = {}
    for t in tracks:
        by_uri.setdefault(t["uri"], []).append(t)
    exact = [
        {"uri": uri, "name": ts[0]["name"], "artists": ts[0]["artists"], "positions": [t["position"] for t in ts]}
        for uri, ts in by_uri.items() if len(ts) > 1
    ]

    by_song: dict[tuple[str, str], dict[str, list[dict]]] = {}
    for uri, ts in by_uri.items():
        by_song.setdefault(_song_key(ts[0]), {})[uri] = ts
    other_versions = [
        {
            "name": next(iter(versions.values()))[0]["name"],
            "artists": next(iter(versions.values()))[0]["artists"],
            "versions": [
                {"uri": uri, "album": ts[0]["album"], "year": ts[0]["year"], "positions": [t["position"] for t in ts]}
                for uri, ts in versions.items()
            ],
        }
        for versions in by_song.values() if len(versions) > 1
    ]

    result = {"exact": exact, "other_versions": other_versions}
    if total > MAX_SCAN:
        result["note"] = f"Only the first {MAX_SCAN} of {total} tracks were checked."
    return result


async def create_playlist(name: str, description: str = "", public: bool = False) -> dict:
    """Creates a new, empty playlist on the user's account. Only call it after they've approved the plan.

    Args:
        name: The playlist name.
        description: A short description.
        public: Whether it shows on their profile. Private by default.

    Returns:
        The new playlist's id and link.
    """
    try:
        p = await _api("POST", "/me/playlists", json={"name": name, "description": description, "public": public})
    except SpotifyError as e:
        return {"error": str(e)}
    return {"id": p["id"], "url": (p.get("external_urls") or {}).get("spotify")}


async def add_tracks_to_playlist(playlist_id: str, track_uris: list[str]) -> dict:
    """Adds tracks to the end of a playlist the user owns or collaborates on, in the given order.

    Args:
        playlist_id: The playlist id.
        track_uris: Track URIs from search_tracks or the user's library, like
            "spotify:track:4iV5W9uYEdYUVa79Axb7Rh". Never make one up.

    Returns:
        How many tracks were added, and any URIs that were skipped as invalid.
    """
    uris = list(dict.fromkeys(track_uris))  # drop repeats, keep order
    invalid = [u for u in uris if not _URI.match(u)]
    uris = [u for u in uris if _URI.match(u)]
    added = 0
    try:
        for i in range(0, len(uris), 100):
            await _api("POST", f"/playlists/{playlist_id}/items", json={"uris": uris[i:i + 100]})
            added += len(uris[i:i + 100])
    except SpotifyError as e:
        return {"error": str(e), "added_before_error": added, "invalid_uris": invalid}
    return {"added": added, "invalid_uris": invalid}


async def remove_tracks_from_playlist(playlist_id: str, track_uris: list[str], keep_one: bool = False) -> dict:
    """Removes tracks from a playlist the user owns or collaborates on. The user approves this call in the app.

    Every copy of each URI is removed. With keep_one, one copy of each is put
    back where it first appeared, which is how to remove exact duplicates.

    Args:
        playlist_id: The playlist id.
        track_uris: The URIs to remove.
        keep_one: True to leave one copy of each track in place.

    Returns:
        How many tracks were removed, or which step failed.
    """
    uris = list(dict.fromkeys(track_uris))
    path = f"/playlists/{playlist_id}/items"
    if not keep_one:
        removed = 0
        try:
            for i in range(0, len(uris), 100):
                await _api("DELETE", path, json={"items": [{"uri": u} for u in uris[i:i + 100]]})
                removed += len(uris[i:i + 100])
        except SpotifyError as e:
            return {"error": str(e), "removed_before_error": removed}
        return {"removed": removed}

    # Removing a URI drops every copy of it, so put one back at its first position.
    try:
        tracks, _, _ = await _read_playlist(playlist_id, 0, MAX_SCAN)
    except SpotifyError as e:
        return {"error": str(e)}
    order = [t["uri"] for t in tracks]
    deduped = []
    for uri in uris:
        if uri not in order:
            continue
        first = order.index(uri)
        try:
            await _api("DELETE", path, json={"items": [{"uri": uri}]})
        except SpotifyError as e:
            return {"error": str(e), "deduped_before_error": deduped}
        order = [u for u in order if u != uri]
        try:
            await _api("POST", path, json={"uris": [uri], "position": first})
        except SpotifyError as e:
            return {
                "error": f"Removed every copy of {uri} but couldn't add one back ({e}). Add it back at position {first}.",
                "deduped_before_error": deduped,
            }
        order.insert(first, uri)
        deduped.append(uri)
    return {"deduped": deduped}


async def update_playlist_details(
    playlist_id: str,
    name: Optional[str] = None,
    description: Optional[str] = None,
    public: Optional[bool] = None,
) -> dict:
    """Renames a playlist or changes its description or visibility. Only call it after the user approves.

    Args:
        playlist_id: The playlist id.
        name: The new name, if changing it.
        description: The new description, if changing it.
        public: True or False to change visibility.

    Returns:
        A confirmation or an error.
    """
    changes = {k: v for k, v in {"name": name, "description": description, "public": public}.items() if v is not None}
    if not changes:
        return {"error": "Nothing to change."}
    try:
        await _api("PUT", f"/playlists/{playlist_id}", json=changes)
    except SpotifyError as e:
        return {"error": str(e)}
    return {"updated": changes}


INSTRUCTION = """You help the user build new Spotify playlists and organize the ones they have. Be casual and direct. Don't use em dashes.

What you can and can't see:
- You can only read and change playlists the user owns or collaborates on (editable in list_my_playlists). Spotify-made playlists like Discover Weekly and other people's playlists only show their names.
- Spotify doesn't give recommendations or audio info like tempo, energy, or mood. You pick songs yourself from what you know about music, guided by the user's listening.

Building a playlist:
1. If the request is vague, ask one short question (mood, genres, length). Default to about 20 songs.
2. Unless they asked for something unrelated to their taste, check get_top_tracks (and get_liked_songs if it helps) to see what they like. Mix in songs they already love with new ones that fit.
3. Call search_tracks for every song you pick. Use a result only if its title and artist match the song you meant. If nothing matches, drop that song. Never make up a URI.
4. Show the plan: the playlist name, a one-line description, and a numbered list of "Title - Artist". Ask if they want any changes.
5. Once they say yes, call create_playlist, then add_tracks_to_playlist with the URIs in order, and share the link.

Organizing:
- Use list_my_playlists, get_playlist_tracks, and find_duplicates to see what's there, then suggest changes: removing duplicates, splitting a playlist by vibe or era into new ones, moving songs, or renaming.
- Before creating, adding, or renaming anything, say exactly what will change and wait for a yes.
- For removals, list the tracks by name, then call remove_tracks_from_playlist. The app asks the user to approve or reject that call, so you don't need a separate yes first.
- For exact duplicates (same URI more than once), call remove_tracks_from_playlist with keep_one=True so one copy stays. For other versions of the same song, ask which version to keep, then remove the others normally.
- Moving a song means adding it to the new playlist first, then removing it from the old one.

If a tool returns an error about signing in, tell the user to run `python -m spotify_curator.auth` from the repo root. For any other error, explain it plainly and don't retry more than once."""


root_agent = LlmAgent(
    model=MODEL,
    name="spotify_curator",
    description="Builds Spotify playlists from your taste and cleans up the ones you already have.",
    instruction=INSTRUCTION,
    tools=[
        list_my_playlists,
        get_playlist_tracks,
        get_top_tracks,
        get_liked_songs,
        search_tracks,
        find_duplicates,
        create_playlist,
        add_tracks_to_playlist,
        FunctionTool(remove_tracks_from_playlist, require_confirmation=True),
        update_playlist_details,
    ],
)
