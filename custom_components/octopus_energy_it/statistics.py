"""Long-term statistics with the real reading dates.

Meter readings reach the API days after the energy was used, so an entity
state changes on the day of the poll, not of the consumption. To give the
Energy dashboard the right days, the cumulative readings are imported as
external statistics. Daily consumption is booked at local midnight of the
day it belongs to. Longer intervals are estimated across their local days,
in proportion to elapsed time; they are not actual daily measurements.

Imports are incremental and idempotent: only periods after the last stored
statistic are written, continuing from its ``sum``. Home Assistant updates,
rather than duplicates, a statistic written again for the same start.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    get_last_statistics,
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


class StatisticsImporter:
    """Import the statistics of one config entry, remembering each cursor."""

    def __init__(self, hass: HomeAssistant) -> None:
        """Start with no cached cursors; they are read from the recorder once."""
        self._hass = hass
        self._cursors: dict[str, Cursor] = {}

    async def _cursor(self, stat_id: str) -> Cursor:
        if (cursor := self._cursors.get(stat_id)) is not None:
            return cursor
        last = await get_instance(self._hass).async_add_executor_job(
            get_last_statistics, self._hass, 1, stat_id, True, {"sum"}
        )
        rows = (last or {}).get(stat_id) or []
        cursor = Cursor()
        if rows:
            row = rows[0]
            cursor = Cursor(dt_util.utc_from_timestamp(row["start"]), row["sum"] or 0.0)
        self._cursors[stat_id] = cursor
        return cursor

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
        """Import new rows of one series; return how many were written."""
        stat_id = statistic_id(account_number, kind)
        rows, cursor = period_rows(
            points, await self._cursor(stat_id), multiplier=multiplier
        )
        if not rows:
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
        self._cursors[stat_id] = cursor
        _LOGGER.debug("Imported %d rows into %s", len(rows), stat_id)
        return len(rows)

    async def async_import_account(
        self, account_number: str, account: dict[str, Any]
    ) -> None:
        """Import electricity, electricity cost and gas statistics of an account."""
        electricity = readings_to_points(
            account.get("electricity_readings") or [], date_key="readAt"
        )
        if electricity:
            await self.async_import(
                account_number,
                ELECTRICITY,
                electricity,
                unit="kWh",
                unit_class="energy",
            )
            price = _flat_electricity_price(account)
            if price is not None:
                await self.async_import(
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
            await self.async_import(
                account_number, GAS, gas, unit="m³", unit_class="volume"
            )


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
