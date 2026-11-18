"""An in-memory stand-in for ``foxglove.client.Client``, as the SDK dataset reader uses it."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mcap.reader import make_reader


@dataclass
class FakeEpisode:
    id: str
    properties: dict[str, object]
    mcap: Path | None = None
    start: datetime = datetime(2026, 1, 1, tzinfo=UTC)
    version: int = 1


class _Page:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows

    def auto_paging_iter(self) -> Iterator[dict[str, Any]]:
        return iter(self._rows)


@dataclass
class FakeClient:
    episodes: list[FakeEpisode]
    streamed: list[tuple[str, int, tuple[str, ...]]] = field(default_factory=list)

    def get_dataset_version(self, *, dataset_id: str, version_number: int) -> dict[str, Any]:
        return {"has_missing_recordings": False}

    def get_dataset_version_episodes(self, **kwargs: Any) -> _Page:
        assert kwargs["include_properties"] is True
        return _Page(
            [
                {
                    "episode": {
                        "id": episode.id,
                        "version": episode.version,
                        "start_time": episode.start,
                        "end_time": episode.start + timedelta(seconds=10),
                        "properties": episode.properties,
                    },
                    "has_missing_recordings": False,
                }
                for episode in self.episodes
            ]
        )

    def iter_messages(
        self,
        *,
        episode_id: str,
        episode_version: int,
        topics: list[str],
        decoder_factories: list[Any] | None = None,
        **_: Any,
    ) -> Iterator[tuple[Any, Any, Any, Any]]:
        self.streamed.append((episode_id, episode_version, tuple(topics)))
        episode = next(item for item in self.episodes if item.id == episode_id)
        assert episode.mcap is not None
        with episode.mcap.open("rb") as handle:
            reader = make_reader(handle)
            for schema, channel, message in reader.iter_messages(topics=topics):
                yield schema, channel, message, message.data
