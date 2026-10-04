"""Built-in current-location and weather tools (internet required)."""

import asyncio
import json
import re
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Any, Optional

from langchain_core.tools import tool

from ..config import NETWORK_REQUEST_TIMEOUT
from . import has_internet_connection
from .network import _get_limited_response, _public_only_async_client
from .schemas import GetWeatherInput

MAX_API_RESPONSE_BYTES = 1_048_576
MAX_WEATHER_DAYS = 16
FORECAST_LIMIT_DAYS = 16
FORECAST_API_PAST_DAYS = 90
NO_INTERNET_MESSAGE = (
    "Error: Network unavailable. Please check your internet connection and try again."
)
IP_GEOLOCATION_SERVICES = (
    ("https://ipwho.is/", "ipwho.is"),
    ("https://ipapi.co/json/", "ipapi.co"),
)
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle",
    55: "dense drizzle", 56: "freezing drizzle", 57: "dense freezing drizzle",
    61: "light rain", 63: "rain", 65: "heavy rain", 66: "freezing rain",
    67: "heavy freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "light showers", 81: "showers", 82: "violent showers",
    85: "light snow showers", 86: "heavy snow showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "severe thunderstorm with hail",
}


def _run(coroutine):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coroutine).result()
    return asyncio.run(coroutine)


async def _get_json_async(url: str, params: Optional[dict] = None) -> Any:
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    async with _public_only_async_client(
        headers={"User-Agent": "private-agent"},
        timeout=NETWORK_REQUEST_TIMEOUT,
    ) as client:
        body, encoding = await _get_limited_response(
            client, url, MAX_API_RESPONSE_BYTES
        )
    return json.loads(body.decode(encoding, errors="replace"))


def _get_json(url: str, params: Optional[dict] = None) -> Any:
    return _run(_get_json_async(url, params))


def _number(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _parse_ip_location(payload: Any, service: str) -> Optional[dict]:
    if not isinstance(payload, dict) or payload.get("success") is False or payload.get("error"):
        return None
    latitude, longitude = _number(payload.get("latitude")), _number(payload.get("longitude"))
    if latitude is None or longitude is None:
        return None
    timezone = payload.get("timezone")
    if isinstance(timezone, dict):
        timezone = timezone.get("id")
    return {
        "city": payload.get("city") or "",
        "region": payload.get("region") or "",
        "country": payload.get("country") or payload.get("country_name") or "",
        "latitude": latitude,
        "longitude": longitude,
        "timezone": timezone if isinstance(timezone, str) else "",
        "source": service,
    }


def locate_current_position() -> dict:
    """Approximate this machine's location from its public IP address."""
    failures = []
    for url, service in IP_GEOLOCATION_SERVICES:
        try:
            location = _parse_ip_location(_get_json(url), service)
        except Exception as exc:
            failures.append(f"{service}: {type(exc).__name__}")
            continue
        if location:
            return location
        failures.append(f"{service}: no usable location")
    raise RuntimeError("Could not determine location (" + "; ".join(failures) + ").")


def _describe_location(location: dict) -> str:
    name = ", ".join(
        part for part in (location.get("city"), location.get("region"), location.get("country")) if part
    )
    return f"{name or 'Unnamed place'} ({location['latitude']:.4f}, {location['longitude']:.4f})"


@tool
def fetch_current_location() -> str:
    """Get the approximate current location (city, region, country, coordinates, timezone) of this machine from its public IP address. Local sessions only."""
    if not has_internet_connection():
        return NO_INTERNET_MESSAGE
    try:
        location = locate_current_position()
    except Exception as exc:
        return f"Location lookup failed: {exc}"
    return (
        f"Approximate current location: {_describe_location(location)}\n"
        f"Timezone: {location['timezone'] or 'unknown'}\n"
        f"Accuracy: IP-based estimate (city level); source: {location['source']}."
    )


def _geocode(place: str) -> dict:
    payload = _get_json(GEOCODING_URL, {"name": place, "count": 1, "format": "json"})
    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        raise ValueError(f"Could not find a place named '{place}'.")
    top = results[0]
    return {
        "city": top.get("name") or place,
        "region": top.get("admin1") or "",
        "country": top.get("country") or "",
        "latitude": float(top["latitude"]),
        "longitude": float(top["longitude"]),
        "timezone": top.get("timezone") or "",
        "source": "open-meteo geocoding",
    }


def resolve_weather_dates(
    start: Optional[str], days: int, *, today: Optional[date] = None
) -> tuple[date, date]:
    """Validate the requested date range; raises ValueError with a clear reason."""
    today = today or date.today()
    try:
        first = date.fromisoformat(start) if start else today
    except ValueError:
        raise ValueError("date must use YYYY-MM-DD format.") from None
    if not 1 <= days <= MAX_WEATHER_DAYS:
        raise ValueError(f"days must be between 1 and {MAX_WEATHER_DAYS}.")
    last = first + timedelta(days=days - 1)
    if last > today + timedelta(days=FORECAST_LIMIT_DAYS):
        raise ValueError(
            f"Forecasts are available only up to {FORECAST_LIMIT_DAYS} days ahead "
            f"(until {today + timedelta(days=FORECAST_LIMIT_DAYS)})."
        )
    if first < date(1940, 1, 1):
        raise ValueError("Historical weather starts at 1940-01-01.")
    return first, last


def _parse_time(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    match = re.fullmatch(r"([01]?\d|2[0-3])(?::([0-5]\d))?", value.strip())
    if not match:
        raise ValueError("time must use HH:MM (24-hour) format.")
    hour, minute = int(match.group(1)), int(match.group(2) or 0)
    return min(hour + (1 if minute >= 30 else 0), 23)


def _fmt(value: Any, unit: str = "") -> str:
    return "n/a" if value is None else f"{value}{unit}"


def format_weather(
    location: dict, payload: dict, first: date, last: date, hour: Optional[int], today: date
) -> str:
    daily = payload.get("daily") or {}
    times = daily.get("time") or []
    if not times:
        return "No weather data was returned for that date range."
    kind = "history" if last < today else "forecast" if first > today else "weather"
    lines = [
        f"Weather {kind} for {_describe_location(location)}"
        + (f", timezone {payload.get('timezone')}" if payload.get("timezone") else "")
    ]
    for index, day in enumerate(times):
        def column(name):
            values = daily.get(name) or []
            return values[index] if index < len(values) else None

        code = column("weather_code")
        lines.append(
            f"- {day}: {WEATHER_CODES.get(code, 'unknown conditions')}; "
            f"high {_fmt(column('temperature_2m_max'), '°C')}, "
            f"low {_fmt(column('temperature_2m_min'), '°C')}; "
            f"precipitation {_fmt(column('precipitation_sum'), ' mm')} "
            f"(chance {_fmt(column('precipitation_probability_max'), '%')}); "
            f"max wind {_fmt(column('wind_speed_10m_max'), ' km/h')}"
        )
    if hour is not None:
        hourly = payload.get("hourly") or {}
        stamp = f"{first.isoformat()}T{hour:02d}:00"
        hourly_times = hourly.get("time") or []
        if stamp in hourly_times:
            position = hourly_times.index(stamp)

            def at(name):
                values = hourly.get(name) or []
                return values[position] if position < len(values) else None

            lines.append(
                f"At {stamp.replace('T', ' ')}: "
                f"{WEATHER_CODES.get(at('weather_code'), 'unknown conditions')}; "
                f"{_fmt(at('temperature_2m'), '°C')}, "
                f"humidity {_fmt(at('relative_humidity_2m'), '%')}, "
                f"precipitation {_fmt(at('precipitation'), ' mm')}, "
                f"wind {_fmt(at('wind_speed_10m'), ' km/h')}"
            )
        else:
            lines.append(f"No hourly data was available for {stamp}.")
    lines.append("Source: Open-Meteo (open-meteo.com).")
    return "\n".join(lines)


def fetch_weather(
    location: dict, first: date, last: date, hour: Optional[int], today: date
) -> str:
    daily = [
        "weather_code", "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "wind_speed_10m_max",
    ]
    hourly = [
        "weather_code", "temperature_2m", "relative_humidity_2m",
        "precipitation", "wind_speed_10m",
    ]
    historical = first < today - timedelta(days=FORECAST_API_PAST_DAYS)
    if not historical:
        daily.append("precipitation_probability_max")
    params = {
        "latitude": location["latitude"],
        "longitude": location["longitude"],
        "start_date": first.isoformat(),
        "end_date": last.isoformat(),
        "daily": ",".join(daily),
        "timezone": "auto",
    }
    if hour is not None:
        params["hourly"] = ",".join(hourly)
    payload = _get_json(ARCHIVE_URL if historical else FORECAST_URL, params)
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(str(payload.get("reason") or "weather service error"))
    return format_weather(location, payload, first, last, hour, today)


@tool(args_schema=GetWeatherInput)
def get_weather(
    location: Optional[str] = None,
    date: Optional[str] = None,
    time: Optional[str] = None,
    days: int = 1,
) -> str:
    """Get weather for a place on a date: history (past days/weeks), today, or a forecast up to 16 days ahead. Omit location to use the current location (Local sessions only)."""
    if not has_internet_connection():
        return NO_INTERNET_MESSAGE
    today = datetime.now().astimezone().date()
    try:
        first, last = resolve_weather_dates(date, days, today=today)
        hour = _parse_time(time)
        place = _geocode(location) if location else locate_current_position()
        return fetch_weather(place, first, last, hour, today)
    except (ValueError, KeyError) as exc:
        return f"Weather lookup failed: {exc}"
    except Exception as exc:
        return f"Weather lookup failed: {type(exc).__name__}: {exc}"
