"""CLI wiring, including API mode against a fake Foxglove client."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.fake_foxglove import FakeClient, FakeEpisode
from tests.synthetic import write_synthetic_episode
from wawtech_workshop.dataset import cli, foxglove_source


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


def _use_client(monkeypatch: pytest.MonkeyPatch, client: FakeClient) -> None:
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "secret-token-value")
    monkeypatch.setattr(foxglove_source, "make_client", lambda: client)


def test_dry_run_prints_selection_without_download(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    client = FakeClient(
        [
            FakeEpisode("ep-9", {"outcome": "success"}),
            FakeEpisode("ep-no", {"outcome": "failure"}),
            FakeEpisode("ep-blank", {}),
        ]
    )
    _use_client(monkeypatch, client)
    cli.main(["--dataset-id", "ds", "--version", "2", "--dry-run"])
    output = capsys.readouterr().out
    assert "ep-9 start=2026-01-01" in output
    assert "ep-no" in output
    assert "ep-blank" in output
    assert "dry-run: not downloading" in output
    assert "secret-token-value" not in output
    assert client.streamed == []


def test_api_mode_uses_episode_task_property(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = write_synthetic_episode(
        tmp_path / "source.mcap", n_frames=4, include_metadata=False, task="not-written"
    )
    client = FakeClient(
        [FakeEpisode("ep-task", {"outcome": "success", "task": "pick the cup"}, mcap=source.path)]
    )
    _use_client(monkeypatch, client)
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
            "--cameras",
            "front",
        ]
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/api-task", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 1
    assert dataset[0]["task"] == "pick the cup"
    assert client.streamed[0][2] == (
        "/observation/state",
        "/action/state",
        "/observation/images/front",
    )


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
