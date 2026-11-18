"""Foxglove REST client for listing dataset-version episodes and downloading MCAPs.

The HTTP layer is an opener callable so tests never touch the network. The API
token is read from ``FOXGLOVE_API_TOKEN`` and is redacted from every error this
module raises. ``FOXGLOVE_API_URL`` overrides the default ``https://api.foxglove.dev``.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.foxglove.dev"
_REDACTED = "[REDACTED]"
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True)
class HttpRequest:
    """One outbound request handed to an opener."""

    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None = None
    timeout_s: float = 60.0


class HttpResponse(Protocol):
    status: int
    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes:
        """Read up to ``size`` bytes, or the rest of the body when ``size`` is negative."""

    def __enter__(self) -> HttpResponse: ...

    def __exit__(self, *args: object) -> None: ...


Opener = Callable[[HttpRequest], HttpResponse]
Sleeper = Callable[[float], None]


class _UrllibResponse:
    def __init__(self, raw: object) -> None:
        self._raw = raw
        self.status = int(getattr(raw, "status", 0))
        header_items = getattr(raw, "headers", {})
        self.headers = {str(key).lower(): str(value) for key, value in dict(header_items).items()}

    def read(self, size: int = -1) -> bytes:
        return self._raw.read(size)  # type: ignore[attr-defined]

    def __enter__(self) -> _UrllibResponse:
        return self

    def __exit__(self, *args: object) -> None:
        close = getattr(self._raw, "close", None)
        if close is not None:
            close()


def _urllib_opener(request: HttpRequest) -> HttpResponse:
    req = urllib.request.Request(
        request.url,
        data=request.body,
        headers=request.headers,
        method=request.method,
    )
    try:
        raw = urllib.request.urlopen(req, timeout=request.timeout_s)  # noqa: S310
    except urllib.error.HTTPError as exc:
        raw = exc
    return _UrllibResponse(raw)


def _base_url() -> str:
    return os.environ.get("FOXGLOVE_API_URL", DEFAULT_API_URL).rstrip("/")


def _resolve_token(token: str | None) -> str:
    resolved = os.environ.get("FOXGLOVE_API_TOKEN") if token is None else token
    if not resolved:
        raise RuntimeError("FOXGLOVE_API_TOKEN is not set.")
    return resolved


def _redact(message: str, token: str | None) -> str:
    if token and token in message:
        return message.replace(token, _REDACTED)
    return message


def _header(headers: Mapping[str, str], name: str) -> str | None:
    target = name.lower()
    for key, value in headers.items():
        if key.lower() == target and value:
            return value
    return None


@dataclass(frozen=True)
class ApiEpisode:
    """One episode row from the dataset-version episodes API."""

    id: str
    version: int
    start: str
    end: str
    properties: dict[str, object]
    device_id: str | None
    has_missing_recordings: bool
    added_at: str

    @property
    def task(self) -> str | None:
        """Episode ``task`` property when it is a non-empty string."""
        value = self.properties.get("task")
        if isinstance(value, str) and value.strip():
            return value
        return None


@dataclass(frozen=True)
class SelectionReport:
    """Episodes kept for conversion, plus ids dropped by the outcome filter.

    ``kept`` is ordered by ascending start time. Episodes with no outcome
    property are excluded and listed in ``unlabeled_ids``; ``unlabeled_count``
    is the warning count for those rows.
    """

    kept: list[ApiEpisode]
    rejected_ids: list[str]
    unlabeled_ids: list[str]
    unlabeled_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unlabeled_count", len(self.unlabeled_ids))

    @property
    def kept_ids(self) -> list[str]:
        return [episode.id for episode in self.kept]


def _device_id(recordings: list[dict[str, object]]) -> str | None:
    """Return the one device the episode's recordings came from, or None.

    Only used for display. Downloads are scoped by episode id and version, so an
    episode that spans several devices, or none, still converts.
    """
    found: list[str] = []
    for recording in recordings:
        device_id = recording.get("deviceId")
        if isinstance(device_id, str) and device_id and device_id not in found:
            found.append(device_id)
    return found[0] if len(found) == 1 else None


def _parse_episode_row(row: dict[str, object]) -> ApiEpisode:
    episode = row.get("episode")
    if not isinstance(episode, dict):
        raise ValueError(f"Episode row is missing an 'episode' object: {row!r}.")
    episode_id = str(episode.get("id", ""))
    recordings = episode.get("recordings") or []
    if not isinstance(recordings, list):
        raise ValueError(f"Episode {episode_id} recordings must be a list.")
    recording_rows = [item for item in recordings if isinstance(item, dict)]
    properties = episode.get("properties") or {}
    if not isinstance(properties, dict):
        raise ValueError(f"Episode {episode_id} properties must be an object.")
    try:
        version = int(episode["version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Episode {episode_id} is missing an integer version.") from exc
    start = episode.get("startTime")
    end = episode.get("endTime")
    if not isinstance(start, str) or not isinstance(end, str):
        raise ValueError(f"Episode {episode_id} is missing startTime or endTime.")
    added_at = row.get("addedAt", "")
    return ApiEpisode(
        id=episode_id,
        version=version,
        start=start,
        end=end,
        properties=dict(properties),
        device_id=_device_id(recording_rows),
        has_missing_recordings=bool(row.get("hasMissingRecordings", False)),
        added_at=str(added_at) if added_at is not None else "",
    )


def _request(
    opener: Opener,
    request: HttpRequest,
    *,
    token: str,
    max_retries: int,
    sleeper: Sleeper,
    backoff_s: float,
) -> HttpResponse:
    attempt = 0
    while True:
        try:
            response = opener(request)
        except Exception as exc:
            raise RuntimeError(
                _redact(f"{request.method} {request.url} failed: {exc}", token)
            ) from exc
        status = int(response.status)
        if status in _RETRY_STATUSES and attempt < max_retries:
            attempt += 1
            delay = backoff_s * (2 ** (attempt - 1))
            logger.warning(
                "HTTP %s for %s %s; retry %s/%s in %.1fs.",
                status,
                request.method,
                request.url,
                attempt,
                max_retries,
                delay,
            )
            close = getattr(response, "__exit__", None)
            if close is not None:
                close(None, None, None)
            sleeper(delay)
            continue
        if status >= 400:
            body = response.read().decode("utf-8", errors="replace")
            message = _redact(
                f"HTTP {status} for {request.method} {request.url}: {body[:500]}",
                token,
            )
            raise RuntimeError(message)
        return response


def _read_json(response: HttpResponse, token: str) -> object:
    raw = response.read()
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(_redact(f"Response was not JSON: {exc}", token)) from exc


def list_version_episodes(
    dataset_id: str,
    version: int | str,
    *,
    token: str | None = None,
    opener: Opener | None = None,
    page_limit: int = 2000,
    timeout_s: float = 60.0,
    max_retries: int = 3,
    sleeper: Sleeper | None = None,
) -> list[ApiEpisode]:
    """List every episode in a dataset version, following cursor pagination.

    Requests ``include=properties,recordings``. The next page cursor is the
    ``fg-pagination-next-cursor`` response header. Outcome filtering is not done
    here; callers filter the returned ``properties`` locally.
    """
    resolved_token = _resolve_token(token)
    http = opener or _urllib_opener
    pause = sleeper or time.sleep
    episodes: list[ApiEpisode] = []
    cursor: str | None = None
    seen_cursors: set[str] = set()
    while True:
        if cursor is not None:
            if cursor in seen_cursors:
                raise RuntimeError("Foxglove episode pagination repeated a cursor.")
            seen_cursors.add(cursor)
        query: dict[str, str] = {
            "include": "properties,recordings",
            "limit": str(page_limit),
        }
        if cursor is not None:
            query["cursor"] = cursor
        url = (
            f"{_base_url()}/v1/datasets/{urllib.parse.quote(str(dataset_id), safe='')}"
            f"/versions/{urllib.parse.quote(str(version), safe='')}/episodes"
            f"?{urllib.parse.urlencode(query)}"
        )
        response = _request(
            http,
            HttpRequest(
                method="GET",
                url=url,
                headers={
                    "Authorization": f"Bearer {resolved_token}",
                    "Accept": "application/json",
                },
                timeout_s=timeout_s,
            ),
            token=resolved_token,
            max_retries=max_retries,
            sleeper=pause,
            backoff_s=0.5,
        )
        try:
            payload = _read_json(response, resolved_token)
            next_cursor = _header(response.headers, "fg-pagination-next-cursor")
        finally:
            response.__exit__(None, None, None)
        if not isinstance(payload, dict) or not isinstance(payload.get("episodes"), list):
            raise RuntimeError("Episode list response did not contain an 'episodes' array.")
        for row in payload["episodes"]:
            if not isinstance(row, dict):
                raise ValueError(f"Episode row was not an object: {row!r}.")
            episodes.append(_parse_episode_row(row))
        if not next_cursor:
            return episodes
        cursor = next_cursor


def _start_sort_key(episode: ApiEpisode) -> tuple[str, str]:
    # RFC3339 timestamps from one API share an offset, so lexicographic order matches time order.
    return (episode.start, episode.id)


def select_episodes(
    episodes: list[ApiEpisode],
    *,
    outcome_key: str = "outcome",
    keep_values: tuple[str, ...] = ("success",),
    allow_missing: bool = False,
) -> SelectionReport:
    """Keep episodes whose outcome property is in ``keep_values``.

    A missing outcome property excludes the episode and counts toward the
    unlabeled warning. Episodes that match the outcome and have
    ``has_missing_recordings`` raise unless ``allow_missing`` is set. Episodes
    that do not match are not checked for missing recordings. Kept episodes are
    returned in ascending start-time order.
    """
    keep = set(keep_values)
    kept: list[ApiEpisode] = []
    rejected_ids: list[str] = []
    unlabeled_ids: list[str] = []
    for episode in episodes:
        if outcome_key not in episode.properties:
            unlabeled_ids.append(episode.id)
            continue
        value = episode.properties[outcome_key]
        if value not in keep:
            rejected_ids.append(episode.id)
            continue
        if episode.has_missing_recordings and not allow_missing:
            raise RuntimeError(
                f"Episode {episode.id} has missing recordings. "
                "Pass allow_missing to convert it anyway."
            )
        kept.append(episode)
    kept.sort(key=_start_sort_key)
    if unlabeled_ids:
        logger.warning(
            "Excluded %d episode(s) with no %r property: %s",
            len(unlabeled_ids),
            outcome_key,
            unlabeled_ids,
        )
    return SelectionReport(kept=kept, rejected_ids=rejected_ids, unlabeled_ids=unlabeled_ids)


def download_episode(
    episode: ApiEpisode,
    dest: Path,
    *,
    topics: list[str] | None = None,
    token: str | None = None,
    opener: Opener | None = None,
    timeout_s: float = 120.0,
    download_timeout_s: float = 900.0,
    max_retries: int = 3,
    sleeper: Sleeper | None = None,
) -> Path:
    """Download one episode version's MCAP via ``POST /v1/data/stream`` and the signed link.

    Bytes are streamed to a temporary file next to ``dest`` and renamed into
    place. Retries with exponential backoff cover HTTP 429 and 5xx. The bearer
    token is sent only to the Foxglove API, not to the signed download URL.
    """
    resolved_token = _resolve_token(token)
    http = opener or _urllib_opener
    pause = sleeper or time.sleep
    # Scope the stream to the episode version pinned by the dataset version, the
    # same request the Foxglove SDK's dataset reader makes through foxglove-client.
    # The server uses that episode's time range and recordings.
    payload: dict[str, object] = {
        "episodeId": episode.id,
        "episodeVersion": episode.version,
        "outputFormat": "mcap",
    }
    if topics is not None:
        payload["topics"] = list(topics)
    body = json.dumps(payload).encode("utf-8")
    response = _request(
        http,
        HttpRequest(
            method="POST",
            url=f"{_base_url()}/v1/data/stream",
            headers={
                "Authorization": f"Bearer {resolved_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            body=body,
            timeout_s=timeout_s,
        ),
        token=resolved_token,
        max_retries=max_retries,
        sleeper=pause,
        backoff_s=0.5,
    )
    try:
        parsed = _read_json(response, resolved_token)
    finally:
        response.__exit__(None, None, None)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("link"), str):
        raise RuntimeError("Data stream response did not contain a link.")
    link = parsed["link"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".partial")
    try:
        download = _request(
            http,
            HttpRequest(
                method="GET",
                url=link,
                headers={"Accept": "application/octet-stream"},
                timeout_s=download_timeout_s,
            ),
            token=resolved_token,
            max_retries=max_retries,
            sleeper=pause,
            backoff_s=0.5,
        )
        try:
            with temporary.open("wb") as handle:
                while True:
                    chunk = download.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
        finally:
            download.__exit__(None, None, None)
        temporary.replace(dest)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return dest
