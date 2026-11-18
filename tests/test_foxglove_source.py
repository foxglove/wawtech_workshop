"""Episode selection and download through the Foxglove SDK dataset reader."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from mcap.reader import make_reader

from tests.fake_foxglove import FakeClient, FakeEpisode
from tests.synthetic import write_synthetic_episode
from wawtech_workshop.dataset.foxglove_source import default_topics, fetch_episodes, make_client


def _at(day: int) -> datetime:
    return datetime(2026, 1, day, tzinfo=UTC)


def test_selection_filters_outcome_sorts_and_skips_download(
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = FakeClient(
        [
            FakeEpisode("late", {"outcome": "success"}, start=_at(2)),
            FakeEpisode("bad", {"outcome": "failure"}, start=_at(1)),
            FakeEpisode("early", {"outcome": "success"}, start=_at(1)),
            FakeEpisode("blank", {}, start=_at(3)),
        ]
    )
    with caplog.at_level("WARNING"):
        report = fetch_episodes(
            "ds", 2, default_topics(["front"]), download_dir=None, client_factory=lambda: client
        )
    assert report.kept_ids == ["early", "late"]
    assert report.rejected_ids == ["bad"]
    assert report.unlabeled_ids == ["blank"]
    assert report.unlabeled_count == 1
    assert "Excluded 1 episode" in caplog.text
    assert client.streamed == []


def test_kept_episodes_are_streamed_into_mcap_files(tmp_path: Path) -> None:
    source = write_synthetic_episode(tmp_path / "source.mcap", n_frames=4, include_metadata=False)
    client = FakeClient(
        [
            FakeEpisode(
                "keep", {"outcome": "success", "task": "pick"}, mcap=source.path, version=3
            ),
            FakeEpisode("drop", {"outcome": "failure"}, mcap=source.path),
        ]
    )
    topics = default_topics(["front"])
    report = fetch_episodes(
        "ds", 1, topics, download_dir=tmp_path / "out", client_factory=lambda: client
    )
    assert client.streamed == [("keep", 3, tuple(topics))]
    [episode] = report.kept
    assert episode.task == "pick"
    assert episode.path == tmp_path / "out" / "keep.mcap"
    assert not (tmp_path / "out" / "keep.mcap.partial").exists()

    def messages(path: Path) -> list[tuple[str, int, bytes]]:
        with path.open("rb") as handle:
            return [
                (channel.topic, message.log_time, message.data)
                for _, channel, message in make_reader(handle).iter_messages(topics=topics)
            ]

    assert messages(episode.path) == messages(source.path)


def test_client_reads_token_and_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOXGLOVE_API_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="FOXGLOVE_API_TOKEN"):
        make_client()
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "secret-token-value")
    monkeypatch.setenv("FOXGLOVE_API_URL", "https://api.example.test/")
    client = make_client()
    assert client.__url__("/v1/x") == "https://api.example.test/v1/x"


def test_default_topics() -> None:
    assert default_topics(["front", "wrist"]) == [
        "/observation/state",
        "/action/state",
        "/observation/images/front",
        "/observation/images/wrist",
    ]
    with pytest.raises(ValueError):
        default_topics([])
