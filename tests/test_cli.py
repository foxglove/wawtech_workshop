"""CLI wiring, including API mode against a fake HTTP opener."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.synthetic import write_synthetic_episode
from tests.test_foxglove_api import MemoryResponse
from wawtech_workshop.dataset import cli
from wawtech_workshop.dataset.foxglove_api import ApiEpisode, HttpRequest, SelectionReport


def test_api_mode_requires_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("FOXGLOVE_API_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            ["--dataset-id", "ds", "--version", "1", "--output", "out", "--repo-id", "local/demo"]
        )
    assert exc_info.value.code == 1
    error = capsys.readouterr().err.strip()
    assert error.startswith("error:")
    assert "FOXGLOVE_API_TOKEN" in error
    assert "\n" not in error


def test_dry_run_prints_selection_without_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "secret-token-value")
    episode = ApiEpisode(
        id="ep-9",
        version=2,
        start="2026-02-01T00:00:00Z",
        end="2026-02-01T00:00:05Z",
        properties={"outcome": "success"},
        device_id="device-7",
        has_missing_recordings=False,
        added_at="2026-02-01T00:00:00Z",
    )
    report = SelectionReport(kept=[episode], rejected_ids=["ep-no"], unlabeled_ids=["ep-blank"])

    def fail_download(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("dry-run must not download")

    monkeypatch.setattr(cli, "list_version_episodes", lambda *_args, **_kwargs: [episode])
    monkeypatch.setattr(cli, "select_episodes", lambda *_args, **_kwargs: report)
    monkeypatch.setattr(cli, "download_episode", fail_download)
    cli.main(["--dataset-id", "ds", "--version", "2", "--dry-run"])
    output = capsys.readouterr().out
    assert "ep-9" in output
    assert "device-7" in output
    assert "ep-no" in output
    assert "ep-blank" in output
    assert "dry-run: not downloading" in output
    assert "secret-token-value" not in output


def _api_opener(mcap_bytes: bytes, properties: dict[str, object]):
    row = {
        "addedAt": "2026-02-01T00:00:00Z",
        "episode": {
            "id": "ep-task",
            "version": 1,
            "projectId": "proj",
            "startTime": "2026-02-01T00:00:00Z",
            "endTime": "2026-02-01T00:00:02Z",
            "properties": properties,
            "recordings": [{"id": "rec-1", "deviceId": "device-1"}],
        },
    }

    def opener(request: HttpRequest) -> MemoryResponse:
        if request.method == "POST":
            body = json.dumps({"link": "https://signed.example/ep.mcap"}).encode()
            return MemoryResponse(200, {}, body)
        if "/episodes" in request.url:
            return MemoryResponse(200, {}, json.dumps({"episodes": [row]}).encode())
        return MemoryResponse(200, {}, mcap_bytes)

    return opener


def test_api_mode_uses_episode_task_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_synthetic_episode(
        tmp_path / "source.mcap", n_frames=4, include_metadata=False, task="not-written"
    )
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "secret-token-value")
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test")
    monkeypatch.setattr(
        "wawtech_workshop.dataset.foxglove_api._urllib_opener",
        _api_opener(source.path.read_bytes(), {"outcome": "success", "task": "pick the cup"}),
    )
    output = tmp_path / "dataset"
    cli.main(
        [
            "--dataset-id",
            "ds",
            "--version",
            "1",
            "--output",
            str(output),
            "--repo-id",
            "local/api-task",
            "--project-dir",
            str(tmp_path / "downloads"),
        ]
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/api-task", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 1
    assert dataset[0]["task"] == "pick the cup"


def test_mcap_dir_task_flag_is_the_fallback(tmp_path: Path) -> None:
    mcap_dir = tmp_path / "mcaps"
    write_synthetic_episode(mcap_dir / "ep.mcap", n_frames=4, include_metadata=False)
    output = tmp_path / "dataset"
    cli.main(
        [
            "--mcap-dir",
            str(mcap_dir),
            "--output",
            str(output),
            "--repo-id",
            "local/flag-task",
            "--task",
            "from the flag",
        ]
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/flag-task", root=output, video_backend="pyav")
    assert dataset[0]["task"] == "from the flag"
