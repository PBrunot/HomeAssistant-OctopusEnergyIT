"""Resumable reading history: twelve months initially, then thirty-day refreshes."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from datetime import date, timedelta
from hashlib import sha256
import logging
from typing import Any

from homeassistant.components import persistent_notification
from homeassistant.components.recorder import get_instance
from homeassistant.core import HomeAssistant
from homeassistant.helpers.storage import Store
from homeassistant.helpers.translation import async_get_translations
from homeassistant.util import dt as dt_util

from .const import DOMAIN
from .data_processor import electricity_register_series
from .octopus_energy_it import OctopusEnergyIT
from .statistics import StatisticsImporter, readings_to_points

_LOGGER = logging.getLogger(__name__)

HISTORY_REFRESH = 30 * 60
HISTORY_REQUEST_INTERVAL = 30
CORRECTION_DAYS = 30
# Statistics layout of the stored series. A change re-downloads twelve
# months so previously imported rows are rewritten in the new layout.
STATISTICS_LAYOUT = "hourly"


def year_ago(day: date) -> date:
    """Return twelve calendar months earlier, including leap-day handling."""
    try:
        return day.replace(year=day.year - 1)
    except ValueError:
        return day.replace(year=day.year - 1, day=28)


def merge_readings(
    previous: list[dict[str, Any]], downloaded: list[dict[str, Any]], kind: str
) -> list[dict[str, Any]]:
    """Merge corrections by timestamp and register, retaining gas boundary readings."""
    date_key = "readAt" if kind == "electricity" else "readingDate"
    merged = {}
    for reading in previous + downloaded:
        points = readings_to_points([reading], date_key=date_key)
        if not points:
            continue
        timestamp, value = points[0]
        normalized = {**reading, "value": value}
        normalized[date_key] = (
            timestamp.isoformat()
            if kind == "electricity"
            else dt_util.as_local(timestamp).date().isoformat()
        )
        merged[timestamp, reading.get("registerId")] = normalized
    return list(merged.values())


class HistorySync:
    """Fetch at most one page per coordinator cycle across all accounts.

    Page buffers and completed source readings are stored locally so a
    restart resumes the download and sparse gas readings retain an anchor.
    Statistics are reconciled only after the complete range was downloaded.
    """

    def __init__(
        self,
        hass: HomeAssistant,
        entry_id: str,
        api: OctopusEnergyIT,
        importer: StatisticsImporter,
    ) -> None:
        self._hass = hass
        self._api = api
        self._importer = importer
        self._store = Store(hass, 1, f"{DOMAIN}.{entry_id}.history")
        self._notification_prefix = f"{DOMAIN}_history_{entry_id}"
        self._series: dict[str, Any] | None = None
        self._next_request = 0.0
        self._position = 0
        self._errors: set[str] = set()

    async def _async_notify(
        self, key: str, kind: str, message_key: str, **values: Any
    ) -> None:
        """Update one translated HA notification per supply instead of adding alerts."""
        try:
            messages = await async_get_translations(
                self._hass, self._hass.config.language, "exceptions", {DOMAIN}
            )
            prefix = f"component.{DOMAIN}.exceptions."
            persistent_notification.async_create(
                self._hass,
                messages[f"{prefix}{message_key}.message"].format(**values),
                title=messages[f"{prefix}history_{kind}_title.message"],
                notification_id=f"{self._notification_prefix}_{sha256(key.encode()).hexdigest()[:12]}",
            )
        except Exception:
            # Notification failures must not interrupt downloads or checkpoints.
            _LOGGER.exception("Unable to update historical import notification")

    def status(self, account_number: str) -> dict[str, Any]:
        """Return progress without account numbers, supply IDs or raw readings."""
        result = {}
        for kind in ("electricity", "gas"):
            key = f"{account_number}:{kind}"
            state = (self._series or {}).get(key)
            if not state:
                continue
            pending = state.get("pending")
            result[kind] = {
                "state": "retrying"
                if key in self._errors
                else "downloading"
                if pending
                else "ready",
                "requested_start": pending["start"]
                if pending
                else state.get("last_start"),
                "requested_end": pending["end"] if pending else state.get("last_end"),
                "pages_downloaded": len(pending["seen"]) if pending else 0,
                "readings_downloaded": len(pending["readings"])
                if pending
                else len(state.get("readings", [])),
                "next_refresh": state.get("next_refresh"),
            }
        return result

    async def async_refresh(self, accounts: dict[str, dict[str, Any]]) -> None:
        """Advance one due series, sharing a low request budget across accounts."""
        # Recorder finishes starting only after HA starts. Waiting for its
        # commit queue during an integration's first refresh can deadlock setup.
        if not self._hass.is_running:
            return
        if self._series is None:
            self._series = await self._store.async_load() or {}
        now = dt_util.utcnow()
        if now.timestamp() < self._next_request:
            return
        today = dt_util.as_local(now).date()
        candidates = [
            (account_number, kind, account)
            for account_number, account in accounts.items()
            for kind in ("electricity", "gas")
            if account.get("electricity_pod" if kind == "electricity" else "gas_pdr")
            and (kind == "gas" or account.get("electricity_property_id"))
        ]
        for index in range(len(candidates)):
            position = (self._position + index) % len(candidates)
            account_number, kind, account = candidates[position]
            key = f"{account_number}:{kind}"
            supply = account["electricity_pod" if kind == "electricity" else "gas_pdr"]
            identity = [
                supply,
                account.get("electricity_property_id")
                if kind == "electricity"
                else None,
            ]
            state = self._series.get(key, {})
            if (
                state.get("identity") != identity
                or state.get("layout") != STATISTICS_LAYOUT
            ):
                state = {"identity": identity, "layout": STATISTICS_LAYOUT}
            if not state.get("pending") and now.timestamp() < state.get(
                "next_refresh", 0
            ):
                continue
            self._position = position + 1
            self._next_request = now.timestamp() + HISTORY_REQUEST_INTERVAL
            try:
                await self._async_step(
                    key, account_number, kind, account, deepcopy(state), today
                )
            except Exception:
                self._errors.add(key)
                _LOGGER.exception(
                    "Historical %s sync failed; the page will be retried", kind
                )
                await self._async_notify(key, kind, "history_sync_retry")
            else:
                self._errors.discard(key)
            return

    async def _async_step(
        self,
        key: str,
        account_number: str,
        kind: str,
        account: dict[str, Any],
        state: dict[str, Any],
        today: date,
    ) -> None:
        """Checkpoint a successful page; only complete windows may change statistics."""
        if not state.get("pending"):
            start = year_ago(today)
            if state.get("last_end"):
                # Catch up an outage longer than thirty days as well.
                start = min(
                    today - timedelta(days=CORRECTION_DAYS),
                    dt_util.parse_date(state["last_end"]) - timedelta(days=1),
                )
            state["pending"] = {
                "start": (start - timedelta(days=1)).isoformat(),
                "end": today.isoformat(),
                "cursor": None,
                "seen": [],
                "readings": [],
                "notify": not state.get("last_end"),
            }
            _LOGGER.info(
                "Starting historical %s download: %s to %s",
                kind,
                state["pending"]["start"],
                state["pending"]["end"],
            )
            if state["pending"]["notify"]:
                await self._async_notify(
                    key,
                    kind,
                    "history_sync_started",
                    start=state["pending"]["start"],
                    end=state["pending"]["end"],
                )
        pending = state["pending"]
        pending["notify"] = pending.get("notify", False) or key in self._errors
        readings, cursor = await self._api.fetch_readings_page(
            kind,
            account_number,
            state["identity"][0],
            property_id=state["identity"][1],
            start_on=pending["start"],
            end_on=pending["end"],
            after=pending["cursor"],
        )
        if cursor is not None and cursor in pending["seen"]:
            raise ValueError("Historical readings pagination repeated a cursor")
        pending["readings"] = merge_readings(pending["readings"], readings, kind)
        _LOGGER.info(
            "Downloaded historical %s page %d: %d readings collected",
            kind,
            len(pending["seen"]) + 1,
            len(pending["readings"]),
        )
        if cursor is not None:
            pending["seen"].append(cursor)
            pending["cursor"] = cursor
            if pending["notify"]:
                await self._async_notify(
                    key,
                    kind,
                    "history_sync_progress",
                    pages=len(pending["seen"]),
                    readings=len(pending["readings"]),
                    start=pending["start"],
                    end=pending["end"],
                )
        else:
            combined = merge_readings(
                state.get("readings", []), pending["readings"], kind
            )
            payload = {}
            if kind == "electricity":
                payload = {
                    "electricity_readings": electricity_register_series(
                        combined, state["identity"][0]
                    ),
                    "current_electricity_product": account.get(
                        "current_electricity_product"
                    ),
                    "electricity_consumption_charge": account.get(
                        "electricity_consumption_charge"
                    ),
                }
            else:
                payload["gas_readings"] = combined
            written = await self._importer.async_import_account(account_number, payload)
            # Persist completion only after recorder has processed the writes.
            async with asyncio.timeout(30):
                await get_instance(self._hass).async_block_till_done()
            state.update(
                readings=combined,
                last_start=pending["start"],
                last_end=pending["end"],
                next_refresh=dt_util.utcnow().timestamp() + HISTORY_REFRESH,
                pending=None,
            )
            _LOGGER.info(
                "Historical %s import complete: %d readings, %d statistic rows added or corrected",
                kind,
                len(combined),
                written,
            )
        # Saving first prevents a failed checkpoint from advancing in-memory state.
        updated = {**self._series, key: state}
        await self._store.async_save(updated)
        self._series = updated
        if cursor is None and (pending["notify"] or written):
            await self._async_notify(
                key,
                kind,
                "history_sync_complete",
                readings=len(combined),
                rows=written,
                start=pending["start"],
                end=pending["end"],
            )
