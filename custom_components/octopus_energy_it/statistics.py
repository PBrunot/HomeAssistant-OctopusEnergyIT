"""Long-term statistics with the real reading dates.

Meter readings reach the API days after the energy was used, so an entity
state changes on the day of the poll, not of the consumption. To give the
Energy dashboard the right days, the cumulative readings are imported as
external statistics. Daily consumption is booked at local midnight of the
day it belongs to. Longer intervals are estimated across their local days,
in proportion to elapsed time; they are not actual daily measurements.

Downloaded periods replace previously imported periods, including estimates.
Older totals are preserved and subsequent sums are adjusted when necessary.
Home Assistant updates a statistic written again for the same start.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import itertools
import logging
import math
import re
from typing import Any

from homeassistant.components.recorder import get_instance
from homeassistant.components.recorder.models import (
    StatisticData,
    StatisticMeanType,
    StatisticMetaData,
)
from homeassistant.components.recorder.statistics import (
    async_add_external_statistics,
    statistics_during_period,
)
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

ELECTRICITY = "electricity_consumption"
ELECTRICITY_COST = "electricity_cost"
GAS = "gas_consumption"

_NAMES = {
    ELECTRICITY: "Octopus {account} electricity consumption",
    ELECTRICITY_COST: "Octopus {account} electricity cost",
    GAS: "Octopus {account} gas consumption",
}


@dataclass(frozen=True)
class Cursor:
    """Last stored statistic of a series: its start and cumulative sum."""

    start: datetime | None = None
    sum: float = 0.0


def statistic_id(account_number: str, kind: str) -> str:
    """Return the external statistic ID, e.g. ``octopus_energy_it:a_1234_gas_consumption``."""
    account = re.sub(r"[^a-z0-9_]+", "_", account_number.lower()).strip("_")
    return f"{DOMAIN}:{account}_{kind}"


def period_rows(
    points: Iterable[tuple[datetime, float]],
    cursor: Cursor,
    *,
    multiplier: float = 1.0,
) -> tuple[list[dict[str, Any]], Cursor]:
    """Turn cumulative readings into statistic rows after *cursor*.

    *points* are ``(read_at, register_value)`` pairs. Each consecutive pair
    contributes its delta to the local days it overlaps, with ``sum``
    continuing from the cursor. Longer intervals have an estimated daily
    allocation, weighted by elapsed time (including 23/25-hour DST days).
    A register that goes backwards contributes zero. A zero baseline before
    the first day lets the Energy dashboard count that day's consumption.
    """
    # The provider can return revised readings at an identical timestamp.
    # Keep the last one; subtracting both can count the same interval twice.
    ordered = sorted(dict(points).items())
    daily: dict[datetime, float] = {}
    for (prev_at, prev_value), (read_at, value) in itertools.pairwise(ordered):
        duration = (read_at - prev_at).total_seconds()
        consumption = max(0.0, value - prev_value) * multiplier
        part_start = prev_at
        while part_start < read_at:
            day = dt_util.as_local(part_start).date()
            start = dt_util.as_utc(dt_util.start_of_local_day(day))
            next_day = dt_util.as_utc(
                dt_util.start_of_local_day(day + timedelta(days=1))
            )
            part_end = min(read_at, next_day)
            daily[start] = daily.get(start, 0.0) + (
                consumption * (part_end - part_start).total_seconds() / duration
            )
            part_start = part_end

    rows: list[dict[str, Any]] = []
    running = cursor.sum
    if daily and cursor.start is None:
        rows.append({"start": min(daily) - timedelta(hours=1), "sum": running})
    for start, consumption in sorted(daily.items()):
        if cursor.start is not None and start <= cursor.start:
            continue
        running += consumption
        # Only sum is needed: a daily allocation is not a register snapshot.
        rows.append({"start": start, "sum": round(running, 6)})
    last_start = rows[-1]["start"] if rows else cursor.start
    return rows, Cursor(last_start, running)


def readings_to_points(
    readings: Iterable[dict[str, Any]], *, date_key: str
) -> list[tuple[datetime, float]]:
    """Return ``(read_at, value)`` pairs from processed reading dicts."""
    points = []
    for reading in readings:
        raw_date, value = reading.get(date_key), reading.get("value")
        if not isinstance(raw_date, str) or value is None:
            continue
        try:
            numeric_value = float(value)
            if "T" in raw_date:
                read_at = dt_util.parse_datetime(raw_date)
            else:  # date only (gas readings): local midnight of that day
                parsed_date = dt_util.parse_date(raw_date)
                read_at = (
                    dt_util.start_of_local_day(parsed_date) if parsed_date else None
                )
        except (TypeError, ValueError, OverflowError):
            continue
        if read_at is None or not math.isfinite(numeric_value) or numeric_value < 0:
            continue
        points.append((dt_util.as_utc(read_at), numeric_value))
    return points


def reconcile_rows(
    downloaded: list[dict[str, Any]], stored: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Replace downloaded days and propagate corrections to later sums.

    The downloaded baseline is relative to zero. Anchor it to the last
    recorder sum before the first downloaded day, then recalculate every
    downloaded day. A changed total also shifts later stored sums, retaining
    those days' consumption even if this response ends before they do.
    """
    if len(downloaded) < 2:
        return []
    first_day = downloaded[1]["start"]
    last_day = downloaded[-1]["start"]
    previous = sorted(stored, key=lambda row: row["start"])
    preceding = [row for row in previous if row["start"] < first_day]
    offset = preceding[-1]["sum"] if preceding else 0.0
    replacement = [
        {"start": row["start"], "sum": round(row["sum"] + offset, 6)}
        for row in downloaded[1:]
    ]
    if not preceding:
        replacement.insert(0, downloaded[0])
    # An older import may have left a baseline within the newly covered
    # range. Flatten such intermediate rows rather than leaving a dip.
    new_starts = [row["start"] for row in replacement]
    new_values = [row["sum"] for row in replacement]
    replacement.extend(
        {
            "start": row["start"],
            "sum": new_values[bisect_right(new_starts, row["start"]) - 1],
        }
        for row in previous
        if first_day <= row["start"] <= last_day and row["start"] not in new_starts
    )
    earlier = [row for row in previous if row["start"] <= last_day]
    old_total = earlier[-1]["sum"] if earlier else 0.0
    adjustment = downloaded[-1]["sum"] + offset - old_total
    replacement.extend(
        {"start": row["start"], "sum": round(row["sum"] + adjustment, 6)}
        for row in previous
        if row["start"] > last_day
    )
    existing = {row["start"]: row["sum"] for row in previous}
    return [
        row
        for row in sorted(replacement, key=lambda row: row["start"])
        if row["start"] not in existing
        or not math.isclose(
            row["sum"], existing[row["start"]], abs_tol=0.0000005, rel_tol=0
        )
    ]


class StatisticsImporter:
    """Import and reconcile the downloaded periods of one config entry."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Remember unchanged downloads to avoid recorder work on every poll."""
        self._hass = hass
        self._downloads: dict[str, tuple[Any, ...]] = {}

    async def async_import(
        self,
        account_number: str,
        kind: str,
        points: list[tuple[datetime, float]],
        *,
        unit: str,
        unit_class: str | None,
        multiplier: float = 1.0,
    ) -> int:
        """Overwrite changed downloaded days; return how many rows were written."""
        stat_id = statistic_id(account_number, kind)
        fingerprint = (
            tuple(sorted(dict(points).items())),
            unit,
            unit_class,
            multiplier,
        )
        if self._downloads.get(stat_id) == fingerprint:
            return 0
        downloaded, _ = period_rows(points, Cursor(), multiplier=multiplier)
        if not downloaded:
            return 0
        # A sliding response starting/ending within a day cannot replace
        # that whole day's total. Reconcile only fully covered local days.
        first_read = min(point[0] for point in points)
        first_local_day = dt_util.as_local(first_read).date()
        first_day = dt_util.as_utc(dt_util.start_of_local_day(first_local_day))
        if first_read > first_day:
            first_day = dt_util.as_utc(
                dt_util.start_of_local_day(first_local_day + timedelta(days=1))
            )
        end_day = dt_util.as_utc(
            dt_util.start_of_local_day(dt_util.as_local(max(t for t, _ in points)))
        )
        complete = [
            row for row in downloaded[1:] if first_day <= row["start"] < end_day
        ]
        if not complete:
            return 0
        base = next(
            row["sum"] for row in reversed(downloaded) if row["start"] < first_day
        )
        downloaded = [{"start": first_day - timedelta(hours=1), "sum": 0.0}] + [
            {"start": row["start"], "sum": round(row["sum"] - base, 6)}
            for row in complete
        ]
        stored = await get_instance(self._hass).async_add_executor_job(
            statistics_during_period,
            self._hass,
            datetime(1970, 1, 1, tzinfo=UTC),
            None,
            {stat_id},
            "hour",
            {unit_class: unit} if unit_class else None,
            {"sum"},
        )
        rows = reconcile_rows(
            downloaded,
            [
                {"start": dt_util.utc_from_timestamp(row["start"]), "sum": row["sum"]}
                for row in (stored or {}).get(stat_id, [])
                if row.get("sum") is not None
            ],
        )
        if not rows:
            self._downloads[stat_id] = fingerprint
            return 0
        metadata = StatisticMetaData(
            mean_type=StatisticMeanType.NONE,
            has_sum=True,
            name=_NAMES[kind].format(account=account_number),
            source=DOMAIN,
            statistic_id=stat_id,
            unit_class=unit_class,
            unit_of_measurement=unit,
        )
        async_add_external_statistics(
            self._hass, metadata, [StatisticData(**row) for row in rows]
        )
        self._downloads[stat_id] = fingerprint
        _LOGGER.debug("Imported %d rows into %s", len(rows), stat_id)
        return len(rows)

    async def async_import_account(
        self, account_number: str, account: dict[str, Any]
    ) -> int:
        """Import an account's statistics; return the added or corrected row count."""
        written = 0
        electricity = readings_to_points(
            account.get("electricity_readings") or [], date_key="readAt"
        )
        if electricity:
            written += await self.async_import(
                account_number,
                ELECTRICITY,
                electricity,
                unit="kWh",
                unit_class="energy",
            )
            price = _flat_electricity_price(account)
            if price is not None:
                written += await self.async_import(
                    account_number,
                    ELECTRICITY_COST,
                    electricity,
                    unit=self._hass.config.currency,
                    unit_class=None,
                    multiplier=price,
                )

        gas = readings_to_points(
            account.get("gas_readings") or [], date_key="readingDate"
        )
        if gas:
            written += await self.async_import(
                account_number, GAS, gas, unit="m³", unit_class="volume"
            )
        return written


def _flat_electricity_price(account: dict[str, Any]) -> float | None:
    """Return the €/kWh price of a single-rate tariff, else None.

    Daily readings cannot be split by time band, so no cost is imported for
    time-of-use (F1/F2/F3) tariffs rather than an inaccurate one.
    """
    product = account.get("current_electricity_product") or {}
    if product.get("isTimeOfUse"):
        return None
    price = account.get("electricity_consumption_charge")
    try:
        return float(price) if price is not None else None
    except (TypeError, ValueError):
        return None
