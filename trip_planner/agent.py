"""Trip Planner: a conversational agent that builds and refines day-by-day travel itineraries."""

from datetime import date

import httpx
from google.adk.agents import LlmAgent

from shared import MODEL, web_search

# WMO weather interpretation codes used by Open-Meteo.
_WEATHER_CODES = {
    0: "clear",
    1: "mostly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "freezing fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    56: "freezing drizzle",
    57: "freezing drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    66: "freezing rain",
    67: "freezing rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    77: "snow grains",
    80: "rain showers",
    81: "rain showers",
    82: "violent rain showers",
    85: "snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


async def get_weather_forecast(city: str, start_date: str, end_date: str) -> dict:
    """Gets the daily weather forecast for a city. Only covers the next 16 days.

    Args:
        city: The city name only, without country, e.g. "Lisbon".
        start_date: First day of the forecast, as YYYY-MM-DD.
        end_date: Last day of the forecast, as YYYY-MM-DD.

    Returns:
        The matched location and daily conditions, highs and lows in Celsius,
        and chance of rain, or an error message.
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            geo = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
            )
            matches = geo.json().get("results")
            if not matches:
                return {"error": f"No place called {city!r} was found."}
            place = matches[0]
            resp = await client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "daily": "weather_code,temperature_2m_max,"
                    "temperature_2m_min,precipitation_probability_max",
                    "timezone": "auto",
                    "start_date": start_date,
                    "end_date": end_date,
                },
            )
    except httpx.HTTPError as e:
        return {"error": f"Weather service unavailable: {e}"}
    data = resp.json()
    if resp.status_code != 200:
        return {"error": data.get("reason", "Forecast unavailable.")}
    daily = data["daily"]
    return {
        "location": f"{place['name']}, {place.get('country', '')}",
        "days": [
            {
                "date": day,
                "conditions": _WEATHER_CODES.get(code, "unknown"),
                "high_c": high,
                "low_c": low,
                "chance_of_rain_pct": rain,
            }
            for day, code, high, low, rain in zip(
                daily["time"],
                daily["weather_code"],
                daily["temperature_2m_max"],
                daily["temperature_2m_min"],
                daily["precipitation_probability_max"],
            )
        ],
    }


def _instruction(_ctx) -> str:
    return f"""You are a practical, friendly trip planner. Today is {date.today():%A, %B %d, %Y}.

To plan a trip you need the destination, the dates or trip length, a rough budget, and the traveler's interests. If the destination or trip length is missing, ask one short question to get it. For anything else that's missing, assume something sensible and say what you assumed.

Research before you plan:
- Use web_search for current information: top sights and neighborhoods, where to stay, getting around, typical prices, and events or closures during the trip.
- If the trip starts within the next 16 days, call get_weather_forecast for those dates. Otherwise, use web_search for the typical weather that month.

Then reply with these sections:
1. Trip at a glance: destination, dates, budget level, and any assumptions you made.
2. Day-by-day itinerary: morning, afternoon, and evening for each day. Group nearby places on the same day to keep travel time short.
3. Where to stay: one or two neighborhoods and why.
4. Getting around: the best local transport options and passes.
5. Estimated budget: per-day cost ranges for lodging, food, transport, and activities, in the local currency.
6. Weather and packing: what to expect and what to bring.
7. Sources: links for prices, events, and anything time-sensitive.

When the traveler asks for changes, update only the affected parts of the plan instead of starting over."""


root_agent = LlmAgent(
    model=MODEL,
    name="trip_planner",
    description="Plans and refines day-by-day travel itineraries using live web search and weather forecasts.",
    instruction=_instruction,
    tools=[web_search, get_weather_forecast],
)
