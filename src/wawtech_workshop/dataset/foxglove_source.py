"""Select and download dataset-version episodes with the Foxglove SDK.

Episodes are read with ``foxglove.datasets.torch.read_dataset``, the Foxglove
SDK's dataset reader. Planning lists the dataset version's episodes and their
properties. Each kept episode is then streamed with
``EpisodeReader.iter_messages`` and written to a local MCAP file, so the
converter can read it as many times as alignment needs. Rejected episodes are
never downloaded.

The SDK pins every episode to the version the dataset version records and
refuses a dataset version with missing recordings. The client reads
``FOXGLOVE_API_TOKEN``; ``FOXGLOVE_API_URL`` overrides the default
``https://api.foxglove.dev``.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from foxglove.datasets import EpisodeReader
from foxglove.datasets.torch import read_dataset
from mcap.decoder import DecoderFactory
from mcap.records import Schema
from mcap.writer import Writer

from wawtech_workshop.mcap_schema import TOPIC_ACTION_STATE, TOPIC_OBSERVATION_STATE, image_topic

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.foxglove.dev"

ClientFactory = Callable[[], Any]


@dataclass(frozen=True)
class EpisodeSummary:
    """One dataset-version episode, plus the MCAP path once it is downloaded."""

    id: str
    start: datetime
    end: datetime
    properties: dict[str, object]
    path: Path | None = None

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
    property are excluded and listed in ``unlabeled_ids``.
    """

    kept: list[EpisodeSummary]
    rejected_ids: list[str]
    unlabeled_ids: list[str]
    unlabeled_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "unlabeled_count", len(self.unlabeled_ids))

    @property
    def kept_ids(self) -> list[str]:
        return [episode.id for episode in self.kept]


def default_topics(camera_names: Sequence[str]) -> list[str]:
    """State, action, and one compressed-video topic per camera."""
    if not camera_names:
        raise ValueError("At least one camera name is required.")
    return [TOPIC_OBSERVATION_STATE, TOPIC_ACTION_STATE, *map(image_topic, camera_names)]


def make_client() -> Any:
    """Return a ``foxglove.client.Client`` for ``FOXGLOVE_API_TOKEN``."""
    from foxglove.client import Client

    token = os.environ.get("FOXGLOVE_API_TOKEN")
    if not token:
        raise RuntimeError("FOXGLOVE_API_TOKEN is not set.")
    url = os.environ.get("FOXGLOVE_API_URL", DEFAULT_API_URL)
    host = urllib.parse.urlparse(url).netloc or url.strip("/")
    return Client(token=token, host=host)


def outcome_of(
    properties: Mapping[str, object], outcome_key: str, keep_values: Sequence[str]
) -> str:
    """Return ``kept``, ``rejected``, or ``unlabeled`` for one episode."""
    if outcome_key not in properties:
        return "unlabeled"
    return "kept" if properties[outcome_key] in set(keep_values) else "rejected"


def fetch_episodes(
    dataset_id: str,
    version: int,
    topics: Sequence[str],
    *,
    download_dir: Path | None,
    outcome_key: str = "outcome",
    keep_values: Sequence[str] = ("success",),
    keep_unlabeled: bool = False,
    client_factory: ClientFactory | None = None,
) -> SelectionReport:
    """Select episodes by outcome and download the kept ones to ``download_dir``.

    Episodes with no ``outcome_key`` property are excluded, unless
    ``keep_unlabeled`` is set (for an org that has no outcome property yet).

    With ``download_dir`` set to None, only the selection is made and nothing is
    downloaded. Each kept episode is written to ``<download_dir>/<id>.mcap``
    with the selected ``topics``.
    """
    if download_dir is not None:
        download_dir.mkdir(parents=True, exist_ok=True)

    def read_episode(episode: EpisodeReader) -> Iterator[tuple[str, EpisodeSummary]]:
        properties = dict(episode.properties)
        verdict = outcome_of(properties, outcome_key, keep_values)
        if verdict == "unlabeled" and keep_unlabeled:
            verdict = "kept"
        path = None
        if verdict == "kept" and download_dir is not None:
            path = _write_mcap(episode, download_dir / f"{episode.id}.mcap")
        yield (
            verdict,
            EpisodeSummary(
                id=episode.id,
                start=episode.start_time,
                end=episode.end_time,
                properties=properties,
                path=path,
            ),
        )

    dataset = read_dataset(
        dataset_id,
        version=version,
        topics=list(topics),
        read_episode=read_episode,
        include_properties=True,
        client_factory=client_factory or make_client,
    )
    kept: list[EpisodeSummary] = []
    rejected_ids: list[str] = []
    unlabeled_ids: list[str] = []
    for verdict, summary in dataset:
        if verdict == "kept":
            kept.append(summary)
        elif verdict == "rejected":
            rejected_ids.append(summary.id)
        else:
            unlabeled_ids.append(summary.id)
    kept.sort(key=lambda episode: (episode.start, episode.id))
    if unlabeled_ids:
        logger.warning(
            "Excluded %d episode(s) with no %r property: %s",
            len(unlabeled_ids),
            outcome_key,
            unlabeled_ids,
        )
    return SelectionReport(kept=kept, rejected_ids=rejected_ids, unlabeled_ids=unlabeled_ids)


class _RawDecoderFactory(DecoderFactory):
    """Hand back message bytes unchanged; the converter decodes them itself."""

    def decoder_for(self, message_encoding: str, schema: Schema | None) -> Callable[[bytes], bytes]:
        del message_encoding, schema
        return bytes


def _write_mcap(episode: EpisodeReader, dest: Path) -> Path:
    """Stream ``episode`` into an MCAP file, renamed into place when complete."""
    temporary = dest.with_name(dest.name + ".partial")
    schema_ids: dict[int, int] = {}
    channel_ids: dict[int, int] = {}
    try:
        with temporary.open("wb") as handle:
            writer = Writer(handle)
            writer.start()
            for schema, channel, message, _ in episode.iter_messages(
                decoder_factories=[_RawDecoderFactory()]
            ):
                if channel.id not in channel_ids:
                    schema_id = 0
                    if schema is not None:
                        if schema.id not in schema_ids:
                            schema_ids[schema.id] = writer.register_schema(
                                name=schema.name, encoding=schema.encoding, data=schema.data
                            )
                        schema_id = schema_ids[schema.id]
                    channel_ids[channel.id] = writer.register_channel(
                        topic=channel.topic,
                        message_encoding=channel.message_encoding,
                        schema_id=schema_id,
                        metadata=dict(channel.metadata),
                    )
                writer.add_message(
                    channel_id=channel_ids[channel.id],
                    log_time=message.log_time,
                    data=message.data,
                    publish_time=message.publish_time,
                    sequence=message.sequence,
                )
            writer.finish()
        temporary.replace(dest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dest
