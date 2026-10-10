"""Hourly water statistics and the supplier's final-day request-window quirk."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.components.recorder.statistics import async_add_external_statistics

from custom_components.severn_trent.api import APIError, WATER_TIME_ZONE
from custom_components.severn_trent.statistics import (
    HourlyWaterStatistics, WaterStatistics, build_hourly_statistics,
)
from tests.conftest import _make_response
from tests.test_review import FrozenDateTime, complete_days, readings_response


def hourly_nodes(day, count=None, value="0.001", unit="m3"):
    """Hourly nodes for one UK calendar day, with local offsets as the supplier sends them."""
    start = datetime.fromisoformat(day).replace(tzinfo=WATER_TIME_ZONE).astimezone(timezone.utc)
    end = (datetime.fromisoformat(day).replace(tzinfo=WATER_TIME_ZONE) + timedelta(days=1)).astimezone(timezone.utc)
    nodes = []
    while start < end and (count is None or len(nodes) < count):
        nodes.append({"node": {
            "startAt": start.astimezone(WATER_TIME_ZONE).isoformat(),
            "endAt": (start + timedelta(hours=1)).astimezone(WATER_TIME_ZONE).isoformat(),
            "value": value, "unit": unit}})
        start += timedelta(hours=1)
    return nodes


def hourly_response(edges, has_next_page=False):
    return _make_response({"data": {"account": {"properties": [{"measurements": {
        "pageInfo": {"hasNextPage": has_next_page}, "edges": edges}}]}}})


def fetch_hours(api, edges, start, end, **kwargs):
    with patch.object(api.session, "post", return_value=hourly_response(edges, **kwargs)) as post:
        return api.get_hourly_history(start, end), post


# --- Daily sensors: the supplier omits the final hour of the window's last day ---

def daily_request(api, days):
    responses = [_make_response(readings_response(days)), _make_response(readings_response({}))]
    with (
        patch.object(api, "authenticate", return_value=True),
        patch.object(api.session, "post", side_effect=responses) as post,
        patch("custom_components.severn_trent.api.datetime", FrozenDateTime),
    ):
        return api.get_meter_readings(), post.call_args_list[0].kwargs["json"]["variables"]


def test_daily_sensor_request_extends_a_day_beyond_yesterday(authenticated_api):
    _, variables = daily_request(authenticated_api, complete_days())
    # Frozen at 13 September 2026 (BST): the window must end at midnight starting the 14th.
    assert variables["endAt"] == "2026-09-13T23:00:00Z"


def test_daily_sensors_ignore_todays_partial_bucket(authenticated_api):
    result, _ = daily_request(authenticated_api, complete_days() | {"2026-09-13": 0.009})
    assert result["latest_daily_date"] == "2026-09-12"
    assert result["yesterday_usage"] == 1
    assert all(reading["date"] < "2026-09-13" for reading in result["all_readings"])


# --- Hourly history from the API ---

@pytest.mark.parametrize("day,first_hour", [
    ("2026-09-12", "2026-09-11T23:00:00+00:00"),   # BST
    ("2026-01-15", "2026-01-15T00:00:00+00:00"),   # GMT
])
def test_hourly_history_is_keyed_by_utc_hour(authenticated_api, day, first_hour):
    next_day = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    hours, post = fetch_hours(authenticated_api, hourly_nodes(day), day, next_day)
    assert len(hours) == 24
    assert min(hours) == first_hour
    assert hours[first_hour] == 0.001
    variables = post.call_args.kwargs["json"]["variables"]
    assert variables["utilityFilters"][0]["waterFilters"]["readingFrequencyType"] == "HOUR_INTERVAL"
    assert variables["startAt"] == first_hour.replace("+00:00", "Z")


@pytest.mark.parametrize("day,expected", [("2026-03-29", 23), ("2026-10-25", 25)])
def test_hourly_history_keeps_every_hour_of_a_clock_change_day(authenticated_api, day, expected):
    next_day = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    hours, _ = fetch_hours(authenticated_api, hourly_nodes(day), day, next_day)
    assert len(hours) == expected
    assert round(sum(hours.values()), 6) == round(expected * 0.001, 6)


def test_hourly_history_returns_only_published_hours_of_a_partial_day(authenticated_api):
    hours, _ = fetch_hours(authenticated_api, hourly_nodes("2026-09-12", count=5), "2026-09-12", "2026-09-13")
    assert sorted(hours) == [f"2026-09-{d}T{h:02}:00:00+00:00" for d, h in [(11, 23), (12, 0), (12, 1), (12, 2), (12, 3)]]


def test_hourly_history_converts_litres(authenticated_api):
    hours, _ = fetch_hours(authenticated_api, hourly_nodes("2026-09-12", count=1, value="24", unit="L"),
                           "2026-09-12", "2026-09-13")
    assert hours == {"2026-09-11T23:00:00+00:00": 0.024}


def test_hourly_history_rejects_truncated_pages(authenticated_api):
    with pytest.raises(APIError, match="truncated"):
        fetch_hours(authenticated_api, hourly_nodes("2026-09-12"), "2026-09-12", "2026-09-13", has_next_page=True)


def test_hourly_history_rejects_non_hourly_intervals(authenticated_api):
    edges = [{"node": {"startAt": "2026-09-12T00:00:00+01:00", "endAt": "2026-09-13T00:00:00+01:00",
                       "value": "0.341", "unit": "m3"}}]
    with pytest.raises(APIError, match="non-hourly"):
        fetch_hours(authenticated_api, edges, "2026-09-12", "2026-09-13")


def test_hourly_history_rejects_conflicting_readings(authenticated_api):
    edges = hourly_nodes("2026-09-12", count=1) + hourly_nodes("2026-09-12", count=1, value="0.002")
    with pytest.raises(APIError, match="Conflicting"):
        fetch_hours(authenticated_api, edges, "2026-09-12", "2026-09-13")


# --- Hourly statistic rows ---

def utc(text):
    return datetime.fromisoformat(text)


def test_hourly_statistics_put_each_reading_at_its_own_hour():
    rows = build_hourly_statistics({
        "2026-09-11T23:00:00+00:00": 0.007, "2026-09-12T00:00:00+00:00": 0.0, "2026-09-12T01:00:00+00:00": 0.024})
    assert [(row["start"], row["sum"]) for row in rows] == [
        (utc("2026-09-11T22:00:00+00:00"), 0.0),      # baseline before the first reading
        (utc("2026-09-11T23:00:00+00:00"), 0.007),
        (utc("2026-09-12T00:00:00+00:00"), 0.007),
        (utc("2026-09-12T01:00:00+00:00"), 0.031),
    ]
    assert build_hourly_statistics({}) == []


def test_hourly_statistics_carry_the_sum_across_a_gap_without_filling_it():
    rows = build_hourly_statistics({"2026-09-12T01:00:00+00:00": 0.002, "2026-09-12T05:00:00+00:00": 0.003})
    # Only the hour before each run of readings is added, carrying the preceding sum.
    assert [(row["start"].hour, row["sum"]) for row in rows] == [(0, 0.0), (1, 0.002), (4, 0.002), (5, 0.005)]


@pytest.mark.parametrize("day", ["2026-03-29", "2026-10-25", "2026-09-12", "2026-01-15"])
def test_hourly_statistics_pass_the_recorder_validator(authenticated_api, day):
    from homeassistant.components.recorder.models import StatisticMeanType
    next_day = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
    hours, _ = fetch_hours(authenticated_api, hourly_nodes(day), day, next_day)
    rows = build_hourly_statistics(hours)
    assert len({row["start"] for row in rows}) == len(hours) + 1
    assert round(rows[-1]["sum"], 6) == round(len(hours) * 0.001, 6)
    with patch("homeassistant.components.recorder.statistics.get_instance") as recorder:
        async_add_external_statistics(MagicMock(), {
            "statistic_id": "severn_trent:water_hourly_test", "source": "severn_trent",
            "name": "Water", "unit_of_measurement": "m³", "unit_class": "volume",
            "has_sum": True, "mean_type": StatisticMeanType.NONE}, rows)
        recorder.return_value.async_import_statistics.assert_called_once()


# --- Hourly importer ---

@pytest.fixture
def importer_setup():
    hass = MagicMock()
    hass.config.components = {"recorder"}

    async def executor(func, *args):
        return func(*args)
    hass.async_add_executor_job = executor
    api = MagicMock()
    api.device_id, api.market_supply_point_id = "meter", "supply"
    entry = SimpleNamespace(data={"account_number": "account"})
    store = MagicMock()
    store.async_load = AsyncMock(return_value=None)
    store.async_save = AsyncMock()
    with (
        patch("custom_components.severn_trent.statistics.Store", return_value=store),
        patch("custom_components.severn_trent.statistics.async_add_external_statistics") as importer,
    ):
        yield hass, entry, api, store, importer


async def test_hourly_importer_is_a_separate_statistic(importer_setup):
    hass, entry, api, _, importer = importer_setup
    api.get_hourly_history.return_value = {"2026-09-12T01:00:00+00:00": 0.002}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    metadata = importer.call_args.args[1]
    assert metadata["statistic_id"] == hourly.statistic_id
    assert hourly.statistic_id.startswith("severn_trent:water_hourly_")
    assert hourly.statistic_id != WaterStatistics(hass, entry, api).statistic_id
    assert metadata["name"] == "Severn Trent hourly water (meter)"
    assert metadata["unit_of_measurement"] == "m³"
    api.get_daily_history.assert_not_called()
    assert hourly.status["status"] == "ready"
    assert hourly.status["hours_imported"] == 1
    assert hourly.status["latest_hour"] == "2026-09-12T01:00:00+00:00"


async def test_hourly_importer_only_reimports_when_readings_change(importer_setup):
    hass, entry, api, _, importer = importer_setup
    api.get_hourly_history.return_value = {"2026-09-12T01:00:00+00:00": 0.002}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    await hourly.async_update()
    assert importer.call_count == 1


async def test_hourly_importer_adds_late_hours_of_a_partial_day(importer_setup):
    hass, entry, api, _, importer = importer_setup
    api.get_hourly_history.return_value = {"2026-09-12T01:00:00+00:00": 0.002}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    api.get_hourly_history.return_value = {
        "2026-09-12T01:00:00+00:00": 0.002, "2026-09-12T02:00:00+00:00": 0.005}
    await hourly.async_update()
    rows = importer.call_args.args[2]
    assert rows[-1] == {"start": utc("2026-09-12T02:00:00+00:00"), "sum": 0.007, "state": 0.007}
    assert hourly.status["hours_imported"] == 2


async def test_hourly_importer_replays_from_a_corrected_hour(importer_setup):
    hass, entry, api, _, importer = importer_setup
    readings = {f"2026-09-12T{hour:02}:00:00+00:00": 0.001 for hour in range(4)}
    api.get_hourly_history.return_value = dict(readings)
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    api.get_hourly_history.return_value = readings | {"2026-09-12T02:00:00+00:00": 0.010}
    await hourly.async_update()
    rows = importer.call_args.args[2]
    assert [(row["start"].hour, row["sum"]) for row in rows] == [(2, 0.012), (3, 0.013)]


async def test_hourly_importer_backfill_overwrites_the_earlier_zero_baseline(importer_setup):
    hass, entry, api, _, importer = importer_setup
    api.get_hourly_history.return_value = {"2026-09-12T05:00:00+00:00": 0.003}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    assert importer.call_args.args[2][0] == {"start": utc("2026-09-12T04:00:00+00:00"), "sum": 0.0, "state": 0.0}
    api.get_hourly_history.return_value = {"2026-09-12T01:00:00+00:00": 0.002}
    await hourly.async_update()
    rows = importer.call_args.args[2]
    assert [(row["start"].hour, row["sum"]) for row in rows] == [(0, 0.0), (1, 0.002), (4, 0.002), (5, 0.005)]


async def test_hourly_importer_keeps_cache_when_the_supplier_fails(importer_setup):
    hass, entry, api, store, importer = importer_setup
    api.get_hourly_history.return_value = {"2026-09-12T01:00:00+00:00": 0.002}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    saved = dict(hourly.cache)
    api.get_hourly_history.side_effect = APIError("temporary")
    await hourly.async_update()
    assert hourly.cache == saved
    assert hourly.status["status"] == "retrying"
    # A restart replays the saved readings in full.
    api.get_hourly_history.side_effect = None
    store.async_load.return_value = saved
    restarted = HourlyWaterStatistics(hass, entry, api)
    await restarted.async_update()
    assert importer.call_count == 2
    assert len(importer.call_args.args[2]) == 2


async def test_hourly_importer_reports_meters_without_hourly_data(importer_setup):
    hass, entry, api, _, importer = importer_setup
    api.get_hourly_history.return_value = {}
    hourly = HourlyWaterStatistics(hass, entry, api)
    await hourly.async_update()
    importer.assert_not_called()
    assert hourly.status["status"] == "no_hourly_data"


# --- Wiring: setup, status sensor and diagnostics ---

async def test_setup_updates_both_statistics_for_smart_meters():
    from custom_components.severn_trent import async_setup_entry
    api = MagicMock()
    api.authenticate.return_value = True
    for name in ("get_manual_meter_readings", "get_balance", "get_rate_limit_info",
                 "get_current_active_payment_schedule", "get_meter_details",
                 "get_outstanding_payment", "get_next_payment_forecast"):
        getattr(api, name).return_value = {}
    api.get_meter_readings.return_value = {"yesterday_usage": 0.3}
    api.capability_type = "AMI"
    hass = MagicMock()
    hass.data = {}
    hass.config_entries.async_forward_entry_setups = AsyncMock()

    async def executor(method, *args):
        return method(*args)
    hass.async_add_executor_job = executor
    entry = SimpleNamespace(entry_id="entry", data={
        "api_key": "test", "account_number": "test", "capability_type": "AMI"})
    coordinator = MagicMock()
    captured = {}

    async def refresh():
        captured["data"] = await factory.call_args.kwargs["update_method"]()
    coordinator.async_config_entry_first_refresh = refresh
    daily = MagicMock(async_update=AsyncMock(), status={"status": "ready"}, cache={"days": {}})
    hourly = MagicMock(async_update=AsyncMock(), status={"status": "ready", "hours_imported": 5})
    with (
        patch("custom_components.severn_trent.SevernTrentAPI", return_value=api),
        patch("custom_components.severn_trent.DataUpdateCoordinator", return_value=coordinator) as factory,
        patch("custom_components.severn_trent.WaterStatistics", return_value=daily),
        patch("custom_components.severn_trent.HourlyWaterStatistics", return_value=hourly),
    ):
        assert await async_setup_entry(hass, entry)
    daily.async_update.assert_awaited_once()
    hourly.async_update.assert_awaited_once()
    assert captured["data"]["hourly_history"] == {"status": "ready", "hours_imported": 5}


def test_history_status_sensor_exposes_the_hourly_statistic():
    from custom_components.severn_trent.sensor import SevernTrentHistorySensor
    coordinator = SimpleNamespace(last_update_success=True, async_request_refresh=AsyncMock(), data={
        "history": {"status": "ready", "statistic_id": "severn_trent:water_abc", "days_imported": 35},
        "hourly_history": {"status": "ready", "statistic_id": "severn_trent:water_hourly_abc",
                           "hours_imported": 336, "latest_hour": "2026-09-12T03:00:00+00:00"}})
    sensor = SevernTrentHistorySensor(coordinator, "test-account")
    sensor.async_write_ha_state = MagicMock()
    sensor._handle_coordinator_update()
    attributes = sensor.extra_state_attributes
    assert sensor.native_value == "ready"
    assert attributes["statistic_id"] == "severn_trent:water_abc"
    assert attributes["hourly_status"] == "ready"
    assert attributes["hourly_statistic_id"] == "severn_trent:water_hourly_abc"
    assert attributes["hours_imported"] == 336
    assert attributes["latest_hour"] == "2026-09-12T03:00:00+00:00"


async def test_diagnostics_report_hourly_progress_without_the_statistic_id():
    import json
    from custom_components.severn_trent.diagnostics import async_get_config_entry_diagnostics
    coordinator = SimpleNamespace(last_update_success=True, data={
        "hourly_history": {"status": "ready", "statistic_id": "private-hourly-id", "hours_imported": 336,
                           "latest_hour": "2026-09-12T03:00:00+00:00"}})
    hass = SimpleNamespace(data={"severn_trent": {"entry": {"coordinator": coordinator}}})
    result = await async_get_config_entry_diagnostics(hass, SimpleNamespace(entry_id="entry", data={}))
    assert result["hourly_history"]["hours_imported"] == 336
    assert result["hourly_history"]["latest_hour"] == "2026-09-12T03:00:00+00:00"
    assert "private-hourly-id" not in json.dumps(result)
