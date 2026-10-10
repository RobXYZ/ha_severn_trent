"""Persistent, correction-aware daily water history for the Energy Dashboard."""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
import hashlib
import logging

from homeassistant.components.recorder.statistics import async_add_external_statistics
from homeassistant.components.recorder.models import StatisticMeanType
from homeassistant.helpers.storage import Store

from .api import APIError, WATER_TIME_ZONE
from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)


def build_statistics(days: dict[str, float]) -> list[dict]:
    """Put each known day's total in its final hour; never invent hourly usage.

    Midnight carries the preceding sum. The preceding day's final hour seeds
    the baseline so HA can calculate the first day's difference. Replaying
    from a changed date also repairs every subsequent cumulative sum.
    """
    if not days:
        return []
    ordered = sorted(days)
    first = datetime.fromisoformat(ordered[0]).replace(tzinfo=WATER_TIME_ZONE).astimezone(timezone.utc)
    rows = [{"start": first - timedelta(hours=1), "sum": 0.0, "state": 0.0}]
    total = 0.0
    for day in ordered:
        local = datetime.fromisoformat(day).replace(tzinfo=WATER_TIME_ZONE)
        rows.append({"start": local.astimezone(timezone.utc), "sum": round(total, 6), "state": round(total, 6)})
        total += days[day]
        last_hour = (local + timedelta(days=1)).astimezone(timezone.utc) - timedelta(hours=1)
        rows.append({"start": last_hour, "sum": round(total, 6), "state": round(total, 6)})
    return rows


def build_hourly_statistics(hours: dict[str, float]) -> list[dict]:
    """Put each hourly reading at its own UTC hour; missing hours are not filled.

    The hour before each run of readings carries the preceding sum. That seeds
    the baseline for the first reading and overwrites an earlier baseline when
    older history is backfilled in front of it.
    """
    rows = []
    total = 0.0
    previous = None
    for hour in sorted(hours):
        start = datetime.fromisoformat(hour)
        if previous is None or start - previous != timedelta(hours=1):
            rows.append({"start": start - timedelta(hours=1), "sum": round(total, 6), "state": round(total, 6)})
        total += hours[hour]
        rows.append({"start": start, "sum": round(total, 6), "state": round(total, 6)})
        previous = start
    return rows


class WaterStatistics:
    """Cache dated volumes, backfill a year, and revisit older corrections."""

    label = "daily"
    id_prefix = "water"
    store_prefix = "history"
    cache_key = "days"
    fetch_method = "get_daily_history"
    recent_days = 35

    def __init__(self, hass, entry, api):
        self.hass, self.api = hass, api
        identity = f"{entry.data['account_number']}|{api.market_supply_point_id}|{api.device_id}"
        digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
        self.statistic_id = f"{DOMAIN}:{self.id_prefix}_{digest}"
        self.store = Store(hass, 1, f"{DOMAIN}.{self.store_prefix}_{digest}")
        self.cache = None
        self.replayed = False
        self.status = {}

    def _rows(self, values: dict[str, float], previous: dict[str, float]) -> list[dict]:
        return build_statistics(values)

    def _status(self, values: dict[str, float], today: date) -> dict:
        recent_expected = {(today - timedelta(days=n)).isoformat() for n in range(1, 36)}
        return {"days_imported": len(values), "latest_day": max(values) if values else None,
                "missing_recent_days": len(recent_expected - values.keys())}

    async def async_update(self):
        if 'recorder' not in self.hass.config.components:
            self.status = {"status": "recorder_disabled"}
            return
        today = datetime.now(WATER_TIME_ZONE).date()
        if self.cache is None:
            self.cache = await self.store.async_load() or {
                self.cache_key: {}, "start": (today - timedelta(days=365)).isoformat(),
                "cursor": (today - timedelta(days=365)).isoformat(),
            }
        previous = self.cache[self.cache_key]
        values = dict(previous)
        cursor = date.fromisoformat(self.cache["cursor"])
        if cursor >= today:
            cursor = date.fromisoformat(self.cache["start"])
        historical_end = min(cursor + timedelta(days=31), today)
        fetch = getattr(self.api, self.fetch_method)
        try:
            # Recent late readings are checked hourly; older history rotates in
            # month-sized requests, keeping each refresh bounded to two calls.
            recent = await self.hass.async_add_executor_job(
                fetch, (today - timedelta(days=self.recent_days)).isoformat(), self._recent_end(today).isoformat())
            older = await self.hass.async_add_executor_job(
                fetch, cursor.isoformat(), historical_end.isoformat())
        except (APIError, ValueError):
            self.status = {"status": "retrying", "statistic_id": self.statistic_id}
            _LOGGER.warning("Water %s history unavailable; retaining the last successful import", self.label)
            return
        values.update(older)
        values.update(recent)
        if values and (values != previous or not self.replayed):
            async_add_external_statistics(self.hass, {
                "statistic_id": self.statistic_id, "source": DOMAIN,
                "name": f"Severn Trent {self.label} water ({self.api.device_id})",
                "unit_of_measurement": "m³", "unit_class": "volume",
                "mean_type": StatisticMeanType.NONE, "has_sum": True,
            }, self._rows(values, previous if self.replayed else {}))
            self.replayed = True
        self.cache.update({self.cache_key: values, "cursor": historical_end.isoformat()})
        await self.store.async_save(self.cache)
        self.status = {"status": "ready" if values else f"no_{self.label}_data",
                       "statistic_id": self.statistic_id, **self._status(values, today),
                       "backfill_start": self.cache["start"], "backfill_cursor": self.cache["cursor"]}

    def _recent_end(self, today: date) -> date:
        return today


class HourlyWaterStatistics(WaterStatistics):
    """Import each hourly reading at its true hour, as a separate statistic."""

    label = "hourly"
    id_prefix = "water_hourly"
    store_prefix = "hourly"
    cache_key = "hours"
    fetch_method = "get_hourly_history"
    # 14 days of late readings and 31-day backfill windows stay under the 1000-row page.
    recent_days = 14

    def _recent_end(self, today: date) -> date:
        # Hours are published as they arrive, so include today's so far.
        return today + timedelta(days=1)

    def _rows(self, values: dict[str, float], previous: dict[str, float]) -> list[dict]:
        rows = build_hourly_statistics(values)
        if not previous:
            return rows
        # Replay only from the earliest new or corrected hour; later sums follow it.
        changed = datetime.fromisoformat(min(hour for hour in values if previous.get(hour) != values[hour]))
        if (changed - timedelta(hours=1)).isoformat() not in values:
            changed -= timedelta(hours=1)  # include the row carrying the preceding sum
        return [row for row in rows if row["start"] >= changed]

    def _status(self, values: dict[str, float], today: date) -> dict:
        return {"hours_imported": len(values), "latest_hour": max(values) if values else None}


def usage_pattern(days: dict[str, float], today: date | None = None) -> dict:
    """Flag three consecutive high days against the preceding 28-day mean."""
    today = today or datetime.now(WATER_TIME_ZONE).date()
    dates = [(today - timedelta(days=n)).isoformat() for n in range(1, 32)]
    if any(day not in days for day in dates):
        return {"status": "insufficient_data"}
    baseline = sum(days[d] for d in dates[3:]) / 28
    threshold = max(0.5, baseline * 2)
    elevated = all(days[d] > threshold for d in dates[:3])
    return {"status": "elevated" if elevated else "normal",
            "baseline_m3": round(baseline, 3), "threshold_m3": round(threshold, 3),
            "comparison_days": 3, "baseline_days": 28,
            "latest_day": dates[0]}
