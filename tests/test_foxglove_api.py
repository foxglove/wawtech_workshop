"""Offline tests for the Foxglove dataset API client."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from wawtech_workshop.dataset.foxglove_api import (
    ApiEpisode,
    HttpRequest,
    download_episode,
    list_version_episodes,
    select_episodes,
)

TOKEN = "secret-token-value"


class MemoryResponse:
    def __init__(self, status: int, headers: dict[str, str], body: bytes) -> None:
        self.status = status
        self.headers = {key.lower(): value for key, value in headers.items()}
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> MemoryResponse:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


def _episode_row(
    episode_id: str,
    *,
    start: str,
    end: str,
    outcome: str | None,
    device_ids: list[str | None],
    missing: bool = False,
    version: int = 3,
) -> dict[str, object]:
    properties: dict[str, object] = {}
    if outcome is not None:
        properties["outcome"] = outcome
    recordings = [
        {"id": f"rec-{episode_id}-{index}", "deviceId": device_id}
        for index, device_id in enumerate(device_ids)
    ]
    row: dict[str, object] = {
        "addedAt": "2026-01-01T00:00:00Z",
        "addedInVersion": version,
        "episode": {
            "id": episode_id,
            "version": version,
            "projectId": "proj",
            "startTime": start,
            "endTime": end,
            "properties": properties,
            "recordings": recordings,
        },
    }
    if missing:
        row["hasMissingRecordings"] = True
    return row


def test_list_version_episodes_follows_cursor_header(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test")
    pages = [
        (
            {
                "episodes": [
                    _episode_row(
                        "ep-a",
                        start="2026-01-01T00:00:01Z",
                        end="2026-01-01T00:00:02Z",
                        outcome="success",
                        device_ids=["dev-1"],
                    )
                ]
            },
            "cursor-2",
        ),
        (
            {
                "episodes": [
                    _episode_row(
                        "ep-b",
                        start="2026-01-01T00:00:03Z",
                        end="2026-01-01T00:00:04Z",
                        outcome="failure",
                        device_ids=["dev-1"],
                    )
                ]
            },
            None,
        ),
    ]
    seen: list[str] = []

    def opener(request: HttpRequest) -> MemoryResponse:
        seen.append(request.url)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        body, cursor = pages[len(seen) - 1]
        headers = {}
        if cursor is not None:
            headers["fg-pagination-next-cursor"] = cursor
        return MemoryResponse(200, headers, json.dumps(body).encode())

    episodes = list_version_episodes("dataset 1", 4, token=TOKEN, opener=opener, page_limit=2)
    assert [episode.id for episode in episodes] == ["ep-a", "ep-b"]
    assert "cursor=" not in seen[0]
    assert "cursor=cursor-2" in seen[1]
    assert "/v1/datasets/dataset%201/versions/4/episodes" in seen[0]
    assert "include=properties%2Crecordings" in seen[0]
    assert episodes[0].device_id == "dev-1"
    assert episodes[0].properties == {"outcome": "success"}
    assert episodes[0].has_missing_recordings is False


def test_select_episodes_filters_outcome_and_sorts(caplog: pytest.LogCaptureFixture) -> None:
    episodes = [
        _api("late", start="2026-01-02T00:00:00Z", outcome="success"),
        _api("bad", start="2026-01-01T00:00:00Z", outcome="failure"),
        _api("early", start="2026-01-01T00:00:00Z", outcome="success"),
        _api("blank", start="2026-01-03T00:00:00Z", outcome=None),
    ]
    with caplog.at_level("WARNING"):
        report = select_episodes(episodes)
    assert report.kept_ids == ["early", "late"]
    assert report.rejected_ids == ["bad"]
    assert report.unlabeled_ids == ["blank"]
    assert report.unlabeled_count == 1
    assert "Excluded 1 episode" in caplog.text


def test_select_episodes_missing_recordings_raises_only_for_kept() -> None:
    missing_success = _api("gone", start="2026-01-01T00:00:00Z", outcome="success", missing=True)
    missing_failure = _api("skip", start="2026-01-01T00:00:01Z", outcome="failure", missing=True)
    report = select_episodes([missing_failure])
    assert report.kept_ids == []
    with pytest.raises(RuntimeError, match="gone"):
        select_episodes([missing_success])
    allowed = select_episodes([missing_success], allow_missing=True)
    assert allowed.kept_ids == ["gone"]


def test_device_id_is_display_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test")

    def opener_for(row: dict[str, object]) -> object:
        def opener(request: HttpRequest) -> MemoryResponse:
            del request
            return MemoryResponse(200, {}, json.dumps({"episodes": [row]}).encode())

        return opener

    same = _episode_row(
        "same",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:00:01Z",
        outcome="success",
        device_ids=["dev-9", "dev-9"],
    )
    episodes = list_version_episodes("ds", 1, token=TOKEN, opener=opener_for(same))
    assert episodes[0].device_id == "dev-9"

    multiple = _episode_row(
        "multi",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:00:01Z",
        outcome="success",
        device_ids=["dev-a", "dev-b"],
    )
    episodes = list_version_episodes("ds", 1, token=TOKEN, opener=opener_for(multiple))
    assert episodes[0].device_id is None

    empty = _episode_row(
        "none",
        start="2026-01-01T00:00:00Z",
        end="2026-01-01T00:00:01Z",
        outcome="success",
        device_ids=[],
    )
    episodes = list_version_episodes("ds", 1, token=TOKEN, opener=opener_for(empty))
    assert episodes[0].device_id is None


def test_errors_redact_the_api_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test")

    def opener(request: HttpRequest) -> MemoryResponse:
        assert TOKEN in request.headers["Authorization"]
        body = json.dumps({"error": f"rejected {TOKEN}"}).encode()
        return MemoryResponse(400, {}, body)

    with pytest.raises(RuntimeError) as exc_info:
        list_version_episodes("ds", 1, token=TOKEN, opener=opener)
    message = str(exc_info.value)
    assert TOKEN not in message
    assert "[REDACTED]" in message


def test_download_episode_streams_and_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test")
    episode = _api("ep-1", start="2026-01-01T00:00:00Z", outcome="success")
    calls: list[tuple[str, str]] = []
    payload = b"mcap-bytes" * 1000

    def opener(request: HttpRequest) -> MemoryResponse:
        calls.append((request.method, request.url))
        if request.method == "POST":
            if sum(1 for method, _url in calls if method == "POST") == 1:
                return MemoryResponse(503, {}, b"unavailable")
            assert TOKEN in request.headers["Authorization"]
            body = json.loads(request.body or b"{}")
            assert body == {"episodeId": "ep-1", "episodeVersion": 1, "outputFormat": "mcap"}
            assert body["outputFormat"] == "mcap"
            assert "topics" not in body
            return MemoryResponse(
                200, {}, json.dumps({"link": "https://signed.example.test/file.mcap"}).encode()
            )
        assert "Authorization" not in request.headers
        return MemoryResponse(200, {}, payload)

    dest = tmp_path / "episode.mcap"
    sleeps: list[float] = []
    download_episode(
        episode, dest, token=TOKEN, opener=opener, sleeper=sleeps.append, max_retries=2
    )
    assert dest.read_bytes() == payload
    assert not (tmp_path / "episode.mcap.partial").exists()
    assert sleeps and sleeps[0] > 0
    assert calls[0][0] == "POST"
    assert calls[2][0] == "GET"


def _api(
    episode_id: str,
    *,
    start: str,
    outcome: str | None,
    missing: bool = False,
) -> ApiEpisode:
    properties: dict[str, object] = {}
    if outcome is not None:
        properties["outcome"] = outcome
    return ApiEpisode(
        id=episode_id,
        version=1,
        start=start,
        end="2026-01-01T00:01:00Z",
        properties=properties,
        device_id="dev",
        has_missing_recordings=missing,
        added_at="2026-01-01T00:00:00Z",
    )
