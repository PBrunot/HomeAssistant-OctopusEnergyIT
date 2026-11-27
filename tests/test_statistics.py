"""Tests for the external statistics import (issues #19 and #21)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.octopus_energy_it import statistics as stats_mod
from custom_components.octopus_energy_it.statistics import (
    Cursor,
    StatisticsImporter,
    period_rows,
    readings_to_points,
    reconcile_rows,
    statistic_id,
)

HOUR = timedelta(hours=1)

ROME = ZoneInfo("Europe/Rome")


def _local_midnight(year, month, day):
    return datetime(year, month, day, tzinfo=ROME).astimezone(UTC)


def _points(*values, start_day=1, month=11):
    """Daily midnight snapshots: values[i] read at local midnight of day start+i."""
    return [
        (_local_midnight(2026, month, start_day + i), value)
        for i, value in enumerate(values)
    ]


class TestStatisticId:
    def test_sanitises_account_number(self):
        assert (
            statistic_id("A-1234ABCD", "gas_consumption")
            == "octopus_energy_it:a_1234abcd_gas_consumption"
        )


def _start(row):
    start = row["start"]
    return start if isinstance(start, datetime) else datetime.fromtimestamp(start, UTC)


def _day_ends(rows):
    """Baseline sum followed by the running total at the end of each local day."""
    return [round(rows[0]["sum"], 6)] + [
        round(row["sum"], 6)
        for row in rows[1:]
        if (_start(row) + HOUR).astimezone(ROME).hour == 0
    ]


class TestPeriodRows:
    def test_daily_consumption_is_spread_over_its_hours(self):
        rows, cursor = period_rows(_points(100.0, 110.0, 125.0), Cursor())

        first = _local_midnight(2026, 11, 1)
        assert [row["start"] for row in rows] == [
            first + HOUR * i for i in range(-1, 48)
        ]
        hourly = [b["sum"] - a["sum"] for a, b in itertools.pairwise(rows)]
        assert hourly[:24] == pytest.approx([10 / 24] * 24, abs=1e-5)
        assert hourly[24:] == pytest.approx([15 / 24] * 24, abs=1e-5)
        assert _day_ends(rows) == [0.0, 10.0, 25.0]
        assert cursor.start == _local_midnight(2026, 11, 3) - HOUR
        assert cursor.sum == pytest.approx(25.0)

    def test_rows_start_on_utc_hours(self):
        rows, _ = period_rows(_points(100.0, 110.0), Cursor())
        assert all(row["start"].tzinfo == UTC for row in rows)
        assert all((row["start"].minute, row["start"].second) == (0, 0) for row in rows)

    def test_continues_after_cursor_without_rewriting_old_hours(self):
        cursor = Cursor(_local_midnight(2026, 11, 2) - HOUR, 500.0)
        rows, new_cursor = period_rows(_points(100.0, 110.0, 125.0), cursor)

        assert rows[0]["start"] == _local_midnight(2026, 11, 2)
        assert len(rows) == 24
        assert rows[-1]["sum"] == pytest.approx(515.0)
        assert new_cursor.sum == pytest.approx(515.0)

    def test_nothing_new_returns_same_cursor(self):
        cursor = Cursor(_local_midnight(2026, 11, 3) - HOUR, 25.0)
        rows, new_cursor = period_rows(_points(100.0, 110.0, 125.0), cursor)

        assert rows == []
        assert new_cursor == cursor

    def test_register_going_backwards_counts_as_zero(self):
        rows, _ = period_rows(_points(100.0, 5.0, 8.0), Cursor())

        assert _day_ends(rows) == [0.0, 0.0, 3.0]

    def test_unsorted_input_is_sorted(self):
        points = _points(100.0, 110.0, 125.0)
        rows, _ = period_rows(list(reversed(points)), Cursor())

        assert _day_ends(rows) == [0.0, 10.0, 25.0]

    def test_multiplier_converts_to_cost_and_state_is_the_total(self):
        rows, _ = period_rows(_points(100.0, 110.0), Cursor(), multiplier=0.25)

        assert rows[-1]["sum"] == pytest.approx(2.5)

    def test_intermediate_reading_shapes_the_hourly_profile(self):
        points = [
            (_local_midnight(2026, 11, 1), 100.0),
            (datetime(2026, 11, 1, 12, tzinfo=ROME).astimezone(UTC), 104.0),
            (_local_midnight(2026, 11, 2), 110.0),
        ]
        rows, _ = period_rows(points, Cursor())

        assert len(rows) == 25  # baseline followed by 24 hours
        hourly = [b["sum"] - a["sum"] for a, b in itertools.pairwise(rows)]
        assert hourly == pytest.approx([4 / 12] * 12 + [6 / 12] * 12, abs=1e-5)
        assert rows[-1]["sum"] == pytest.approx(10.0)

    def test_dst_day_has_twenty_five_hourly_rows(self):
        # 25 Oct 2026: Italy leaves summer time; the day is 25 hours long.
        rows, _ = period_rows(_points(100.0, 110.0, start_day=25, month=10), Cursor())

        assert len(rows) == 26
        assert rows[-1]["start"] + HOUR == _local_midnight(2026, 10, 26)
        hourly = [b["sum"] - a["sum"] for a, b in itertools.pairwise(rows)]
        assert hourly == pytest.approx([10 / 25] * 25, abs=1e-5)

    def test_reported_example_counts_first_hour_with_a_baseline(self):
        rows, _ = period_rows(_points(13329.56, 13345.26, start_day=23), Cursor())
        assert rows[0]["start"] == _local_midnight(2026, 11, 23) - HOUR
        assert rows[0]["sum"] == 0
        assert rows[1]["start"] == _local_midnight(2026, 11, 23)
        assert rows[1]["sum"] == pytest.approx(15.70 / 24, abs=1e-6)
        assert rows[-1]["sum"] == pytest.approx(15.70)

    def test_duplicate_timestamp_keeps_last_revision(self):
        points = _points(100.0, 110.0, 125.0)
        points.insert(1, (points[0][0], 90.0))
        rows, _ = period_rows(points, Cursor())
        assert _day_ends(rows) == [0.0, 20.0, 35.0]

    def test_multi_day_interval_does_not_book_everything_on_first_day(self):
        points = [
            (_local_midnight(2026, 11, 1), 100),
            (_local_midnight(2026, 11, 4), 130),
        ]
        rows, _ = period_rows(points, Cursor())
        assert _day_ends(rows) == [0, 10, 20, 30]
        assert len(rows) == 1 + 72

    @pytest.mark.parametrize(
        ("month", "first_day", "hours"), [(3, 28, [24, 23, 24]), (10, 24, [24, 25, 24])]
    )
    def test_multi_day_estimate_respects_dst(self, month, first_day, hours):
        points = [
            (_local_midnight(2026, month, first_day), 100),
            (_local_midnight(2026, month, first_day + 3), 100 + sum(hours)),
        ]
        rows, _ = period_rows(points, Cursor())
        ends = _day_ends(rows)
        assert [b - a for a, b in itertools.pairwise(ends)] == pytest.approx(hours)

    def test_insufficient_readings_does_not_create_a_baseline(self):
        assert period_rows(_points(100.0), Cursor()) == ([], Cursor())


class TestReadingsToPoints:
    @pytest.mark.parametrize(
        ("month", "utc_hour", "offset"), [(11, 23, "+01:00"), (7, 22, "+02:00")]
    )
    def test_graphql_offset_and_utc_timestamps_book_same_italian_day(
        self, month, utc_hour, offset
    ):
        local = [
            {"readAt": f"2026-{month:02d}-{day:02d}T00:00:00{offset}", "value": value}
            for day, value in [(2, 100), (3, 115.7)]
        ]
        utc = [
            {"readAt": f"2026-{month:02d}-{day:02d}T{utc_hour}:00:00Z", "value": value}
            for day, value in [(1, 100), (2, 115.7)]
        ]
        local_rows, _ = period_rows(
            readings_to_points(local, date_key="readAt"), Cursor()
        )
        utc_rows, _ = period_rows(readings_to_points(utc, date_key="readAt"), Cursor())
        assert utc_rows == local_rows
        assert utc_rows[1]["start"] == _local_midnight(2026, month, 2)
        assert utc_rows[-1]["start"] == _local_midnight(2026, month, 3) - HOUR
        assert utc_rows[-1]["sum"] == pytest.approx(15.7)
        assert utc_rows[-1]["start"].tzinfo == UTC

    @pytest.mark.parametrize("value", ["invalid", "nan", "inf", "-inf", -1, {}, []])
    def test_invalid_numeric_values_are_skipped(self, value):
        assert (
            readings_to_points(
                [{"readAt": "2026-11-01T00:00:00Z", "value": value}], date_key="readAt"
            )
            == []
        )

    def test_datetime_and_date_only_readings(self):
        points = readings_to_points(
            [
                {"readAt": "2026-11-01T23:00:00+00:00", "value": 10.0},
                {"readingDate": "2026-11-03", "value": 3},
            ],
            date_key="readAt",
        ) + readings_to_points(
            [{"readingDate": "2026-11-03", "value": 3}], date_key="readingDate"
        )

        assert points == [
            (datetime(2026, 11, 1, 23, tzinfo=UTC), 10.0),
            (_local_midnight(2026, 11, 3), 3.0),
        ]

    def test_invalid_entries_are_skipped(self):
        assert (
            readings_to_points(
                [{"readAt": None, "value": 1}, {"readAt": "x", "value": None}],
                date_key="readAt",
            )
            == []
        )


# ---------------------------------------------------------------------------
# StatisticsImporter
# ---------------------------------------------------------------------------


def _hass():
    return SimpleNamespace(config=SimpleNamespace(currency="EUR"))


def _electricity(*values):
    return [
        {"readAt": point.isoformat(), "value": value}
        for point, value in _points(*values)
    ]


@pytest.fixture
def recorder():
    """Model recorder reads and upserts by statistic ID and UTC timestamp."""
    state = SimpleNamespace(last={}, added=[])
    instance = MagicMock()

    async def _executor(func, *args):
        return state.last

    def _add(hass, meta, rows):
        state.added.append((meta, rows))
        stat_id = meta["statistic_id"]
        existing = {row["start"]: row for row in state.last.get(stat_id, [])}
        existing.update(
            (row["start"].timestamp(), {**row, "start": row["start"].timestamp()})
            for row in rows
        )
        state.last[stat_id] = sorted(existing.values(), key=lambda row: row["start"])

    instance.async_add_executor_job = AsyncMock(side_effect=_executor)
    with (
        patch.object(stats_mod, "get_instance", return_value=instance),
        patch.object(
            stats_mod,
            "async_add_external_statistics",
            side_effect=_add,
        ),
    ):
        yield state


class TestStatisticsImporter:
    async def test_imports_electricity_and_cost_for_flat_tariff(self, recorder):
        account = {
            "electricity_readings": _electricity(100.0, 110.0),
            "electricity_consumption_charge": 0.2,
            "current_electricity_product": {"isTimeOfUse": False},
        }
        written = await StatisticsImporter(_hass()).async_import_account("A-1", account)

        (energy_meta, energy_rows), (cost_meta, cost_rows) = recorder.added
        assert written == 50  # baseline + 24 hourly rows for energy and cost
        assert (
            energy_meta["statistic_id"]
            == "octopus_energy_it:a_1_electricity_consumption"
        )
        assert energy_meta["unit_of_measurement"] == "kWh"
        assert energy_meta["unit_class"] == "energy"
        assert energy_meta["has_sum"] is True
        assert _day_ends(energy_rows) == [0.0, 10.0]
        assert cost_meta["unit_of_measurement"] == "EUR"
        assert cost_meta["unit_class"] is None
        assert cost_rows[-1]["sum"] == pytest.approx(2.0)

    async def test_no_cost_for_time_of_use_tariff(self, recorder):
        account = {
            "electricity_readings": _electricity(100.0, 110.0),
            "electricity_consumption_charge": 0.2,
            "current_electricity_product": {"isTimeOfUse": True},
        }
        await StatisticsImporter(_hass()).async_import_account("A-1", account)

        assert [meta["statistic_id"] for meta, _ in recorder.added] == [
            "octopus_energy_it:a_1_electricity_consumption"
        ]

    async def test_gas_statistics_in_cubic_metres(self, recorder):
        account = {
            "gas_readings": [
                {"readingDate": "2026-09-01", "value": 1000.0},
                {"readingDate": "2026-10-01", "value": 1040.5},
            ]
        }
        await StatisticsImporter(_hass()).async_import_account("A-1", account)

        ((meta, rows),) = recorder.added
        assert meta["unit_of_measurement"] == "m³"
        assert meta["unit_class"] == "volume"
        assert rows[1]["start"] == _local_midnight(2026, 9, 1)
        assert rows[-1]["sum"] == pytest.approx(40.5)
        assert len(rows) == 1 + 30 * 24  # baseline plus estimated hourly totals

    async def test_resumes_from_recorder_and_caches_cursor(self, recorder):
        stat_id = "octopus_energy_it:a_1_electricity_consumption"
        day_one, _ = period_rows(_points(100.0, 110.0), Cursor())
        recorder.last = {
            stat_id: [
                {"start": row["start"].timestamp(), "sum": round(row["sum"] + 890, 6)}
                for row in day_one
            ]
        }
        importer = StatisticsImporter(_hass())
        account = {"electricity_readings": _electricity(100.0, 110.0, 125.0)}

        await importer.async_import_account("A-1", account)
        await importer.async_import_account("A-1", account)  # nothing new

        ((_meta, rows),) = recorder.added
        assert len(rows) == 24  # only the hours of the new day
        assert rows[0]["start"] == _local_midnight(2026, 11, 2)
        assert rows[-1]["sum"] == pytest.approx(915.0)
        # The recorder is queried once; unchanged downloads are skipped.
        assert (
            stats_mod.get_instance.return_value.async_add_executor_job.await_count == 1
        )

    async def test_nothing_to_import_without_readings(self, recorder):
        await StatisticsImporter(_hass()).async_import_account("A-1", {})
        assert recorder.added == []

    async def test_delayed_daily_reading_overwrites_estimates(self, recorder):
        importer = StatisticsImporter(_hass())
        readings = _electricity(100.0, 106.0, 130.0)
        stat_id = statistic_id("A-1", stats_mod.ELECTRICITY)
        # The first download has a two-day gap: estimated 15 kWh each day.
        await importer.async_import_account(
            "A-1", {"electricity_readings": [readings[0], readings[2]]}
        )
        assert _day_ends(recorder.last[stat_id]) == [0, 15, 30]
        # GraphQL later supplies the intermediate reading: actual 6 then 24.
        await importer.async_import_account("A-1", {"electricity_readings": readings})
        assert _day_ends(recorder.last[stat_id]) == [0, 6, 30]
        assert len(recorder.last[stat_id]) == 1 + 48

    async def test_correction_survives_restart_without_duplicate_totals(self, recorder):
        stat_id = statistic_id("A-1", stats_mod.ELECTRICITY)
        await StatisticsImporter(_hass()).async_import_account(
            "A-1", {"electricity_readings": _electricity(100, 110, 125)}
        )
        # Recreate the importer (HA restart), and correct the middle reading.
        importer = StatisticsImporter(_hass())
        await importer.async_import_account(
            "A-1", {"electricity_readings": _electricity(100, 108, 125)}
        )
        assert _day_ends(recorder.last[stat_id]) == [0, 8, 25]
        count = len(recorder.added)
        await StatisticsImporter(_hass()).async_import_account(
            "A-1", {"electricity_readings": _electricity(100, 108, 125)}
        )
        assert len(recorder.added) == count

    async def test_sliding_window_preserves_older_consumption(self, recorder):
        stat_id = statistic_id("A-1", stats_mod.ELECTRICITY)
        importer = StatisticsImporter(_hass())
        await importer.async_import_account(
            "A-1", {"electricity_readings": _electricity(100, 110, 125)}
        )
        readings = [
            {"readAt": t.isoformat(), "value": v}
            for t, v in _points(110, 125, 145, start_day=2)
        ]
        await importer.async_import_account("A-1", {"electricity_readings": readings})
        assert _day_ends(recorder.last[stat_id]) == [0, 10, 25, 45]

    async def test_partial_boundary_days_do_not_replace_complete_days(self, recorder):
        stat_id = statistic_id("A-1", stats_mod.ELECTRICITY)
        importer = StatisticsImporter(_hass())
        await importer.async_import_account(
            "A-1", {"electricity_readings": _electricity(100, 110, 125, 145)}
        )
        readings = [
            {"readAt": "2026-11-01T12:00:00+01:00", "value": 105},
            {"readAt": "2026-11-02T00:00:00+01:00", "value": 110},
            {"readAt": "2026-11-03T00:00:00+01:00", "value": 120},
            {"readAt": "2026-11-03T12:00:00+01:00", "value": 130},
        ]
        await importer.async_import_account("A-1", {"electricity_readings": readings})
        # Only day 2 is completely covered and corrected (15 -> 10).
        assert _day_ends(recorder.last[stat_id]) == [0, 10, 20, 40]

    async def test_download_with_no_complete_days_leaves_recorder_untouched(
        self, recorder
    ):
        readings = [
            {"readAt": "2026-11-01T12:00:00+01:00", "value": 105},
            {"readAt": "2026-11-01T18:00:00+01:00", "value": 110},
        ]
        await StatisticsImporter(_hass()).async_import_account(
            "A-1", {"electricity_readings": readings}
        )
        assert recorder.added == []

    async def test_failed_import_is_retried(self, recorder):
        importer = StatisticsImporter(_hass())
        readings = {"electricity_readings": _electricity(100, 110)}
        with (
            patch.object(
                stats_mod,
                "async_add_external_statistics",
                side_effect=RuntimeError("recorder unavailable"),
            ),
            pytest.raises(RuntimeError),
        ):
            await importer.async_import_account("A-1", readings)
        await importer.async_import_account("A-1", readings)
        assert len(recorder.added) == 1


class TestReconcileRows:
    def test_correction_adjusts_later_sums_without_changing_later_usage(self):
        stored, _ = period_rows(_points(100, 110, 125, 140), Cursor())
        downloaded, _ = period_rows(_points(100, 108, 120), Cursor())
        merged = {r["start"]: r["sum"] for r in stored}
        merged.update(
            (r["start"], r["sum"]) for r in reconcile_rows(downloaded, stored)
        )
        rows = [{"start": k, "sum": v} for k, v in sorted(merged.items())]
        assert _day_ends(rows) == [0, 8, 20, 35]

    def test_identical_download_is_idempotent(self):
        rows, _ = period_rows(_points(100, 110, 125), Cursor())
        assert reconcile_rows(rows, rows) == []

    def test_earlier_history_rebases_existing_later_totals(self):
        stored, _ = period_rows(_points(110, 125, start_day=2), Cursor())
        downloaded, _ = period_rows(_points(100, 110, 125), Cursor())
        merged = {r["start"]: r["sum"] for r in stored}
        merged.update(
            (r["start"], r["sum"]) for r in reconcile_rows(downloaded, stored)
        )
        assert merged[_local_midnight(2026, 11, 2) - HOUR] == pytest.approx(10)
        assert merged[_local_midnight(2026, 11, 3) - HOUR] == pytest.approx(25)
        assert all(
            b >= a for a, b in itertools.pairwise(v for _, v in sorted(merged.items()))
        )
