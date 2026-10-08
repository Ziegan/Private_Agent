from datetime import date
from unittest.mock import patch

import pytest

from private_agent.agent import runtime
from private_agent.tools import (
    AVAILABLE_TOOLS,
    LOCAL_ONLY_NETWORK_TOOL_NAMES,
    NETWORK_TOOL_NAMES,
    location,
)

TODAY = date(2026, 10, 4)


def test_tools_are_registered_as_network_tools():
    assert {"fetch_current_location", "get_weather"} <= set(AVAILABLE_TOOLS)
    assert {"fetch_current_location", "get_weather"} <= NETWORK_TOOL_NAMES
    assert LOCAL_ONLY_NETWORK_TOOL_NAMES == {"fetch_current_location"}


def test_location_tool_hidden_from_online_sessions(monkeypatch):
    monkeypatch.setattr(runtime, "ONLINE_PERMISSION_OVERRIDE", False)
    kwargs = {"capabilities": {}, "isolation_warning": None}
    assert runtime._tool_unavailable_reason(
        "fetch_current_location", provider_type="online", **kwargs
    )
    assert runtime._tool_unavailable_reason(
        "fetch_current_location", provider_type="local", **kwargs
    ) is None
    assert runtime._tool_unavailable_reason(
        "get_weather", provider_type="online", **kwargs
    ) is None


def test_online_weather_requires_explicit_place():
    assert runtime._local_location_refusal("get_weather", {}, False)
    assert runtime._local_location_refusal("get_weather", {"location": "Paris"}, False) is None
    assert runtime._local_location_refusal("get_weather", {}, True) is None
    assert runtime._local_location_refusal("fetch_current_location", {}, False)


@pytest.mark.parametrize("tool_name", ["fetch_current_location", "get_weather"])
def test_tools_check_internet_before_any_request(tool_name):
    args = {} if tool_name == "fetch_current_location" else {"location": "Paris"}
    with patch.object(location, "has_internet_connection", return_value=False), \
         patch.object(location, "_get_json") as request:
        result = AVAILABLE_TOOLS[tool_name].invoke(args)
    assert "Network unavailable" in result
    request.assert_not_called()


def test_current_location_parses_and_falls_back():
    calls = []

    def fake(url, params=None):
        calls.append(url)
        if "ipwho.is" in url:
            return {"success": False}
        return {
            "city": "Chennai", "region": "Tamil Nadu", "country_name": "India",
            "latitude": 13.08, "longitude": 80.27, "timezone": "Asia/Kolkata",
        }

    with patch.object(location, "has_internet_connection", return_value=True), \
         patch.object(location, "_get_json", side_effect=fake):
        result = AVAILABLE_TOOLS["fetch_current_location"].invoke({})
    assert "Chennai, Tamil Nadu, India" in result and "Asia/Kolkata" in result
    assert len(calls) == 2


def test_date_resolution_limits():
    assert location.resolve_weather_dates(None, 1, today=TODAY) == (TODAY, TODAY)
    assert location.resolve_weather_dates("2026-10-03", 7, today=TODAY)[1] == date(2026, 10, 9)
    with pytest.raises(ValueError):
        location.resolve_weather_dates("04-10-2026", 1, today=TODAY)
    with pytest.raises(ValueError):
        location.resolve_weather_dates("2026-10-30", 1, today=TODAY)
    with pytest.raises(ValueError):
        location.resolve_weather_dates(None, 17, today=TODAY)
    assert location._parse_time("14:40") == 15
    assert location._parse_time("23:45") == 23
    with pytest.raises(ValueError):
        location._parse_time("25:00")


def test_weather_uses_archive_for_old_history_and_formats_hour():
    seen = {}

    def fake(url, params=None):
        seen["url"], seen["params"] = url, params
        return {
            "timezone": "Asia/Kolkata",
            "daily": {
                "time": ["2026-01-01"], "weather_code": [61],
                "temperature_2m_max": [30], "temperature_2m_min": [24],
                "precipitation_sum": [2.5], "wind_speed_10m_max": [12],
            },
            "hourly": {
                "time": ["2026-01-01T09:00"], "weather_code": [3],
                "temperature_2m": [26], "relative_humidity_2m": [80],
                "precipitation": [0.1], "wind_speed_10m": [8],
            },
        }

    place = {"city": "Chennai", "region": "", "country": "India",
             "latitude": 13.08, "longitude": 80.27, "timezone": "", "source": "t"}
    with patch.object(location, "_get_json", side_effect=fake):
        text = location.fetch_weather(
            place, date(2026, 1, 1), date(2026, 1, 1), 9, TODAY
        )
    assert seen["url"] == location.ARCHIVE_URL
    assert "precipitation_probability_max" not in seen["params"]["daily"]
    assert "history for Chennai" in text and "light rain" in text
    assert "At 2026-01-01 09:00: overcast; 26°C" in text


def test_weather_forecast_uses_forecast_api():
    seen = {}

    def fake(url, params=None):
        seen["url"] = url
        return {"daily": {"time": ["2026-10-05"], "weather_code": [0]}}

    place = {"city": "X", "region": "", "country": "", "latitude": 1.0,
             "longitude": 2.0, "timezone": "", "source": "t"}
    with patch.object(location, "_get_json", side_effect=fake):
        text = location.fetch_weather(
            place, date(2026, 10, 5), date(2026, 10, 5), None, TODAY
        )
    assert seen["url"] == location.FORECAST_URL
    assert "forecast for X" in text and "clear sky" in text


def test_online_permission_override_enables_device_tools(monkeypatch):
    monkeypatch.setattr(runtime, "ONLINE_PERMISSION_OVERRIDE", False)
    kwargs = {"capabilities": {"vision": True}, "isolation_warning": None}
    names = ("fetch_current_location", "capture_webcam_image")
    for name in names:
        assert runtime._tool_unavailable_reason(name, provider_type="online", **kwargs)
    monkeypatch.setattr(runtime, "ONLINE_PERMISSION_OVERRIDE", True)
    for name in names:
        assert runtime._tool_unavailable_reason(name, provider_type="online", **kwargs) is None
    assert runtime._device_tools_allowed("online")
    assert runtime._local_location_refusal("get_weather", {}, runtime._device_tools_allowed("online")) is None
