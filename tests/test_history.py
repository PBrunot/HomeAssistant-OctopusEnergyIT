"""History pagination, checkpoints, refresh windows and recovery."""

from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pytest

from custom_components.octopus_energy_it import (
    history as history_mod,
    statistics as statistics_mod,
)
from custom_components.octopus_energy_it.history import (
    HistorySync,
    merge_readings,
    year_ago,
)
from custom_components.octopus_energy_it.octopus_energy_it import (
    OctopusEnergyIT,
    OctopusEnergyITConnectionError,
)

ACCOUNT = "A-EXAMPLE"
POD = "example-supply"
ACCOUNTS = {ACCOUNT: {"electricity_pod": POD, "electricity_property_id": "property"}}


def _reading(day, value, register="total"):
    return {
        "readAt": f"2026-11-{day:02d}T00:00:00+01:00",
        "value": value,
        "registerId": register,
        "supplyPointId": POD,
    }


@pytest.fixture
def env():
    state = SimpleNamespace(data=None, now=datetime(2026, 11, 25, 12, tzinfo=UTC))

    async def load():
        return deepcopy(state.data)

    async def save(data):
        state.data = deepcopy(data)

    state.store = SimpleNamespace(
        async_load=AsyncMock(side_effect=load), async_save=AsyncMock(side_effect=save)
    )
    state.hass = SimpleNamespace(
        is_running=True, config=SimpleNamespace(language="en", currency="EUR")
    )
    state.api = SimpleNamespace(
        fetch_readings_page=AsyncMock(
            return_value=([_reading(23, 100), _reading(24, 115)], None)
        )
    )
    state.importer = SimpleNamespace(async_import_account=AsyncMock(return_value=2))
    state.recorder = SimpleNamespace(async_block_till_done=AsyncMock())
    strings = json.loads(
        (
            Path(__file__).resolve().parent.parent
            / "custom_components/octopus_energy_it/strings.json"
        ).read_text(encoding="utf-8")
    )
    messages = {
        f"component.octopus_energy_it.exceptions.{key}.message": item["message"]
        for key, item in strings["exceptions"].items()
    }
    with (
        patch.object(history_mod, "Store", return_value=state.store),
        patch.object(history_mod.dt_util, "utcnow", side_effect=lambda: state.now),
        patch.object(history_mod, "get_instance", return_value=state.recorder),
        patch.object(
            history_mod, "async_get_translations", new=AsyncMock(return_value=messages)
        ),
        patch.object(history_mod.persistent_notification, "async_create") as notify,
    ):
        state.notify = notify
        state.new_sync = lambda: HistorySync(
            state.hass, "entry", state.api, state.importer
        )
        yield state


def test_twelve_month_boundary_handles_leap_day():
    assert year_ago(date(2024, 2, 29)) == date(2023, 2, 28)
    assert year_ago(date(2025, 3, 1)) == date(2024, 3, 1)


def test_merge_corrects_timestamp_without_mixing_registers():
    result = merge_readings(
        [_reading(23, 100), _reading(23, 30, "other")],
        [_reading(23, 110)],
        "electricity",
    )
    assert sorted(r["value"] for r in result) == [30, 110]


def test_merge_normalizes_equivalent_timestamps_and_skips_unstorable_values():
    updated = {**_reading(23, 110), "readAt": "2026-11-22T23:00:00Z"}
    invalid = [
        _reading(24, float("nan")),
        _reading(24, float("inf")),
        {**_reading(24, 130), "readAt": "invalid-date"},
    ]
    result = merge_readings([_reading(23, 100)], [updated, *invalid], "electricity")
    assert len(result) == 1
    assert result[0]["value"] == 110
    assert result[0]["readAt"] == "2026-11-22T23:00:00+00:00"


async def test_failed_recorder_commit_does_not_mark_backfill_complete(env):
    env.recorder.async_block_till_done.side_effect = RuntimeError("commit failed")
    await env.new_sync().async_refresh(ACCOUNTS)
    env.importer.async_import_account.assert_awaited_once()
    assert env.data is None
    assert "will retry automatically" in env.notify.call_args.args[1]


async def test_daily_layout_series_is_downloaded_again_for_hourly_rows(env):
    # Series completed before hourly statistics carry no layout marker.
    env.data = {
        f"{ACCOUNT}:electricity": {
            "identity": [POD, "property"],
            "last_start": "2026-10-24",
            "last_end": "2026-11-24",
            "next_refresh": env.now.timestamp() + 3600,
            "readings": [_reading(23, 100)],
        }
    }
    await env.new_sync().async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_args.kwargs["start_on"] == "2025-11-24"
    state = env.data[f"{ACCOUNT}:electricity"]
    assert state["layout"] == history_mod.STATISTICS_LAYOUT
    assert state["last_end"] == "2026-11-25"


async def test_initial_history_is_twelve_months_with_boundary_day(env):
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    env.api.fetch_readings_page.assert_awaited_once_with(
        "electricity",
        ACCOUNT,
        POD,
        property_id="property",
        start_on="2025-11-24",
        end_on="2026-11-25",
        after=None,
    )
    env.importer.async_import_account.assert_awaited_once()
    env.recorder.async_block_till_done.assert_awaited_once()
    assert sync.status(ACCOUNT)["electricity"]["state"] == "ready"
    assert env.data[f"{ACCOUNT}:electricity"]["last_end"] == "2026-11-25"


async def test_no_history_work_before_home_assistant_started(env):
    env.hass.is_running = False
    await env.new_sync().async_refresh(ACCOUNTS)
    env.store.async_load.assert_not_awaited()
    env.api.fetch_readings_page.assert_not_awaited()


async def test_initial_notifications_show_progress_and_complete_without_duplicates(
    env, caplog
):
    caplog.set_level(logging.INFO, logger=history_mod.__name__)
    env.api.fetch_readings_page.side_effect = [
        ([_reading(23, 100)], "next"),
        ([_reading(24, 115)], None),
    ]
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    assert "Importing historical" in env.notify.call_args_list[0].args[1]
    assert "Downloaded 1 pages" in env.notify.call_args_list[1].args[1]
    env.now += timedelta(seconds=30)
    await sync.async_refresh(ACCOUNTS)
    assert "Historical import completed" in env.notify.call_args.args[1]
    assert "2 statistic records" in env.notify.call_args.args[1]
    assert (
        len({call.kwargs["notification_id"] for call in env.notify.call_args_list}) == 1
    )
    assert all(
        POD not in call.args[1] and ACCOUNT not in call.args[1]
        for call in env.notify.call_args_list
    )
    assert "Starting historical electricity download" in caplog.text
    assert "Downloaded historical electricity page 1" in caplog.text
    assert "Historical electricity import complete" in caplog.text


async def test_unchanged_rolling_refresh_does_not_create_notifications(env):
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    count = env.notify.call_count
    env.now += timedelta(minutes=30)
    env.importer.async_import_account.return_value = 0
    await sync.async_refresh(ACCOUNTS)
    assert env.notify.call_count == count


async def test_rolling_correction_notifies_about_written_records(env):
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    count = env.notify.call_count
    env.now += timedelta(minutes=30)
    env.importer.async_import_account.return_value = 3
    await sync.async_refresh(ACCOUNTS)
    assert env.notify.call_count == count + 1
    assert "3 statistic records" in env.notify.call_args.args[1]


async def test_notification_failure_does_not_interrupt_history(env):
    with patch.object(
        history_mod,
        "async_get_translations",
        new=AsyncMock(side_effect=RuntimeError("translation unavailable")),
    ):
        await env.new_sync().async_refresh(ACCOUNTS)
    assert env.data[f"{ACCOUNT}:electricity"]["last_end"] == "2026-11-25"
    env.importer.async_import_account.assert_awaited_once()


@pytest.mark.parametrize("language", ["en", "it"])
def test_history_notification_translations_have_compatible_placeholders(language):
    root = (
        Path(__file__).resolve().parent.parent / "custom_components/octopus_energy_it"
    )
    strings = json.loads(
        (root / "translations" / f"{language}.json").read_text(encoding="utf-8")
    )
    for key, value in strings["exceptions"].items():
        if key.startswith("history_"):
            assert value["message"].format(
                start="2025-11-25", end="2026-11-25", pages=1, readings=2, rows=3
            )


async def test_pages_are_checkpointed_and_imported_only_when_complete(env):
    env.api.fetch_readings_page.side_effect = [
        ([_reading(23, 100)], "page-2"),
        ([_reading(24, 115)], None),
    ]
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    env.importer.async_import_account.assert_not_awaited()
    assert env.data[f"{ACCOUNT}:electricity"]["pending"]["cursor"] == "page-2"
    assert sync.status(ACCOUNT)["electricity"]["readings_downloaded"] == 1
    # Calls before the next coordinator interval do not spend API budget.
    env.now += timedelta(seconds=29)
    await sync.async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_count == 1
    env.now += timedelta(seconds=1)
    await sync.async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_args.kwargs["after"] == "page-2"
    payload = env.importer.async_import_account.await_args.args[1]
    assert len(payload["electricity_readings"]) == 2


async def test_restart_resumes_saved_page_instead_of_restarting_backfill(env):
    env.api.fetch_readings_page.side_effect = [
        ([_reading(23, 100)], "page-2"),
        ([_reading(24, 115)], None),
    ]
    await env.new_sync().async_refresh(ACCOUNTS)
    await env.new_sync().async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_args.kwargs["after"] == "page-2"
    assert (
        len(
            env.importer.async_import_account.await_args.args[1]["electricity_readings"]
        )
        == 2
    )


async def test_completed_backfill_refreshes_thirty_days_after_half_hour(env):
    await env.new_sync().async_refresh(ACCOUNTS)
    sync = env.new_sync()
    env.now += timedelta(minutes=29)
    await sync.async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_count == 1
    env.now += timedelta(minutes=1)
    await sync.async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_args.kwargs["start_on"] == "2026-10-25"


async def test_long_outage_extends_refresh_to_last_successful_end(env):
    await env.new_sync().async_refresh(ACCOUNTS)
    env.now = datetime(2027, 3, 1, 12, tzinfo=UTC)
    await env.new_sync().async_refresh(ACCOUNTS)
    request = env.api.fetch_readings_page.await_args.kwargs
    assert request["start_on"] == "2026-11-23"
    assert request["end_on"] == "2027-03-01"


async def test_failed_page_keeps_checkpoint_and_retries_same_cursor(env):
    env.api.fetch_readings_page.side_effect = [
        ([_reading(23, 100)], "page-2"),
        OctopusEnergyITConnectionError("rate limited"),
        ([_reading(24, 115)], None),
    ]
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    checkpoint = deepcopy(env.data)
    env.now += timedelta(seconds=30)
    await sync.async_refresh(ACCOUNTS)
    assert env.data == checkpoint
    env.importer.async_import_account.assert_not_awaited()
    assert sync.status(ACCOUNT)["electricity"]["state"] == "retrying"
    assert "will retry automatically" in env.notify.call_args.args[1]
    env.now += timedelta(seconds=30)
    await sync.async_refresh(ACCOUNTS)
    assert env.api.fetch_readings_page.await_args.kwargs["after"] == "page-2"
    env.importer.async_import_account.assert_awaited_once()
    assert "Historical import completed" in env.notify.call_args.args[1]


async def test_repeated_cursor_does_not_finish_or_advance_checkpoint(env):
    env.api.fetch_readings_page.side_effect = [
        ([_reading(23, 100)], "page-2"),
        ([_reading(24, 115)], "page-2"),
    ]
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    checkpoint = deepcopy(env.data)
    env.now += timedelta(seconds=30)
    await sync.async_refresh(ACCOUNTS)
    assert env.data == checkpoint
    env.importer.async_import_account.assert_not_awaited()


async def test_failed_recorder_does_not_checkpoint_completion(env):
    env.importer.async_import_account.side_effect = RuntimeError("recorder failed")
    await env.new_sync().async_refresh(ACCOUNTS)
    assert env.data is None


async def test_failed_checkpoint_does_not_advance_in_memory(env):
    env.store.async_save.side_effect = RuntimeError("disk failed")
    sync = env.new_sync()
    await sync.async_refresh(ACCOUNTS)
    assert sync._series == {}
    assert env.data is None


async def test_only_one_page_across_all_accounts_and_utilities_per_cycle(env):
    accounts = deepcopy(ACCOUNTS)
    accounts[ACCOUNT]["gas_pdr"] = "example-gas"
    accounts["A-SECOND"] = {"gas_pdr": "example-other-gas"}
    env.api.fetch_readings_page.return_value = ([], "next")
    sync = env.new_sync()
    for expected in [("electricity", ACCOUNT), ("gas", ACCOUNT), ("gas", "A-SECOND")]:
        await sync.async_refresh(accounts)
        assert env.api.fetch_readings_page.await_args.args[:2] == expected
        env.now += timedelta(seconds=30)
    assert env.api.fetch_readings_page.await_count == 3


async def test_supply_change_starts_new_initial_download(env):
    await env.new_sync().async_refresh(ACCOUNTS)
    env.now += timedelta(minutes=1)
    accounts = {
        ACCOUNT: {
            "electricity_pod": "replacement-supply",
            "electricity_property_id": "new-property",
        }
    }
    await env.new_sync().async_refresh(accounts)
    request = env.api.fetch_readings_page.await_args
    assert request.args[2] == "replacement-supply"
    assert request.kwargs["start_on"] == "2025-11-24"
    assert request.kwargs["after"] is None


async def test_sparse_gas_refresh_retains_preceding_reading(env):
    accounts = {ACCOUNT: {"gas_pdr": "example-gas"}}
    env.api.fetch_readings_page.return_value = (
        [
            {"readingDate": "2026-09-01", "value": 100},
            {"readingDate": "2026-11-01", "value": 120},
        ],
        None,
    )
    sync = env.new_sync()
    await sync.async_refresh(accounts)
    env.now += timedelta(minutes=30)
    env.api.fetch_readings_page.return_value = (
        [{"readingDate": "2026-11-01", "value": 125}],
        None,
    )
    await sync.async_refresh(accounts)
    assert env.importer.async_import_account.await_args.args[1]["gas_readings"] == [
        {"readingDate": "2026-09-01", "value": 100},
        {"readingDate": "2026-11-01", "value": 125},
    ]


def _connection(nodes=(), *, more=False, cursor=None):
    return {
        "edges": [{"node": n} for n in nodes],
        "pageInfo": {"hasNextPage": more, "endCursor": cursor},
    }


async def test_api_electricity_page_uses_date_range_and_pagination():
    api = OctopusEnergyIT("a@example.com", "pw")
    node = {
        "readAt": "2026-11-23T23:00:00Z",
        "value": "115.7",
        "unit": "KWH",
        "metaData": {
            "utilityFilters": {"registerId": "total", "marketSupplyPointId": POD}
        },
    }
    response = {
        "data": {
            "property": {"measurements": _connection([node], more=True, cursor="next")}
        }
    }
    with patch.object(
        api, "_execute_graphql", new=AsyncMock(return_value=response)
    ) as execute:
        readings, cursor = await api.fetch_readings_page(
            "electricity",
            ACCOUNT,
            POD,
            property_id="property",
            start_on="2025-11-24",
            end_on="2026-11-25",
            after="prev",
        )
    assert cursor == "next"
    assert readings[0]["value"] == 115.7
    assert readings[0]["unit"] == "kWh"
    assert "raw" not in readings[0]
    assert execute.await_args.kwargs["variables"] == {
        "first": 50,
        "after": "prev",
        "propertyId": "property",
        "pod": POD,
        "startOn": "2025-11-24",
        "endOn": "2026-11-25",
    }


async def test_api_gas_page_normalizes_readings():
    api = OctopusEnergyIT("a@example.com", "pw")
    response = {
        "data": {
            "gasMeterReadings": _connection(
                [{"readingDate": "2026-11-01", "consumptionValue": "125"}]
            )
        }
    }
    with patch.object(
        api, "_execute_graphql", new=AsyncMock(return_value=response)
    ) as execute:
        readings, cursor = await api.fetch_readings_page(
            "gas",
            ACCOUNT,
            "example-gas",
            property_id=None,
            start_on="2025-11-24",
            end_on="2026-11-25",
        )
    assert cursor is None
    assert readings[0]["value"] == 125
    assert execute.await_args.kwargs["variables"]["dateFrom"] == "2025-11-24"


@pytest.mark.parametrize(
    "connection",
    [
        None,
        {},
        {"edges": []},
        {"edges": [], "pageInfo": {"hasNextPage": True, "endCursor": None}},
        {"edges": [], "pageInfo": {"hasNextPage": True, "endCursor": "same"}},
    ],
)
async def test_api_rejects_incomplete_or_nonadvancing_pages(connection):
    api = OctopusEnergyIT("a@example.com", "pw")
    with (
        patch.object(
            api,
            "_execute_graphql",
            new=AsyncMock(
                return_value={"data": {"property": {"measurements": connection}}}
            ),
        ),
        pytest.raises(OctopusEnergyITConnectionError),
    ):
        await api.fetch_readings_page(
            "electricity",
            ACCOUNT,
            POD,
            property_id="property",
            start_on="2025-11-24",
            end_on="2026-11-25",
            after="same",
        )


async def test_api_graphql_error_is_not_a_successful_empty_page():
    api = OctopusEnergyIT("a@example.com", "pw")
    with (
        patch.object(
            api,
            "_execute_graphql",
            new=AsyncMock(return_value={"errors": [{"message": "rate limited"}]}),
        ),
        pytest.raises(OctopusEnergyITConnectionError),
    ):
        await api.fetch_readings_page(
            "electricity",
            ACCOUNT,
            POD,
            property_id="property",
            start_on="2025-11-24",
            end_on="2026-11-25",
        )


async def test_backfill_and_rolling_correction_reconcile_real_statistics_importer(env):
    env.importer = statistics_mod.StatisticsImporter(env.hass)
    stored = {}

    async def read(*_args):
        return deepcopy(stored)

    def write(_hass, metadata, rows):
        stat_id = metadata["statistic_id"]
        existing = {r["start"]: r for r in stored.get(stat_id, [])}
        existing.update(
            (r["start"].timestamp(), {**r, "start": r["start"].timestamp()})
            for r in rows
        )
        stored[stat_id] = sorted(existing.values(), key=lambda r: r["start"])

    env.recorder.async_add_executor_job = AsyncMock(side_effect=read)
    env.api.fetch_readings_page.side_effect = [
        ([_reading(1, 100), _reading(2, 110)], "next"),
        ([_reading(23, 200), _reading(24, 215)], None),
        (
            [_reading(1, 100), _reading(2, 110), _reading(23, 205), _reading(24, 215)],
            None,
        ),
    ]
    stat_id = statistics_mod.statistic_id(ACCOUNT, statistics_mod.ELECTRICITY)
    rome = ZoneInfo("Europe/Rome")

    def daily(day):
        """Running total at the end of *day* (the row of its last hour)."""
        timestamp = datetime(2026, 11, day + 1, tzinfo=rome).timestamp() - 3600
        return next(
            round(r["sum"], 3) for r in stored[stat_id] if r["start"] == timestamp
        )

    with (
        patch.object(statistics_mod, "get_instance", return_value=env.recorder),
        patch.object(
            statistics_mod, "async_add_external_statistics", side_effect=write
        ),
    ):
        sync = env.new_sync()
        await sync.async_refresh(ACCOUNTS)
        assert stored == {}  # an incomplete page cannot overwrite history
        env.now += timedelta(seconds=30)
        await sync.async_refresh(ACCOUNTS)
        assert daily(1) == 10  # source history older than the old ten-reading window
        assert daily(23) - daily(22) == 15
        assert daily(23) == 115
        # The next complete 30-day download corrects the intermediate register.
        env.now += timedelta(minutes=30)
        await sync.async_refresh(ACCOUNTS)
        assert daily(23) - daily(22) == 10
        assert daily(23) == 115  # correction changes allocation, not the total
        assert daily(1) == 10
