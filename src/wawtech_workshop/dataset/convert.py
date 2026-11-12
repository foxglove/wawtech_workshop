"""Convert episode MCAP files into a LeRobot v3 dataset.

Alignment is a fixed grid with zero-order hold. For an episode, ``t0`` is the
latest first timestamp across ``/observation/state``, ``/action/state``, and
every camera topic, and ``t1`` is the earliest last timestamp. Grid times are
``t_k = t0 + k / fps`` while ``t_k <= t1``. At each ``t_k`` the state, action,
and camera frame are the latest sample at or before ``t_k``.

A topic whose consecutive samples inside that window are farther apart than
``max_gap_s`` (default 0.25 s) raises ``ValueError`` naming the episode, topic,
and the time of the later sample. Sample time is the MCAP log time.

Camera frames are decoded one image at a time. Each camera keeps the latest
decoded frame whose log time is at or before the current grid time, and drops
the previous image. The first grid time still needs a frame at or before it.

An ``episode`` metadata record is optional. Whether a Foxglove data-stream
download preserves the recorder's record is unverified, and conversion does not
depend on it. When the record is present it supplies ``task`` and
``episode_index`` and is checked against the labels and camera topics in the
file. Otherwise the task comes from :class:`EpisodeInput`, then
``default_task``, and the episode index comes from :class:`EpisodeInput` or the
position in the input list.

LeRobot assigns dataset episode indices ``0 .. N-1`` in ascending resolved
``episode_index`` order. The task string on every frame is the resolved task.
``LeRobotDataset.finalize`` runs on both success and failure so a partial write
is closed; files already written are left in place.
"""

from __future__ import annotations

import json
import logging
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pyav_utils import detect_available_encoders_pyav
from mcap.reader import make_reader

from wawtech_workshop.dataset.video import iter_decode_h264_messages
from wawtech_workshop.mcap_schema import (
    COMPRESSED_VIDEO_SCHEMA_NAME,
    EPISODE_METADATA_NAME,
    SCALARS_SCHEMA_NAME,
    TOPIC_ACTION_STATE,
    TOPIC_OBSERVATION_STATE,
    EpisodeMetadata,
    camera_name_from_topic,
)

logger = logging.getLogger(__name__)

_NS = 1_000_000_000


@dataclass
class _ScalarStream:
    times_ns: list[int]
    values: np.ndarray
    labels: tuple[str, ...]


@dataclass(frozen=True)
class EpisodeInput:
    """One episode file plus optional task and index overrides.

    A plain path is the same as ``EpisodeInput(path)``. ``task`` and
    ``episode_index`` are used only when the MCAP has no ``episode`` metadata
    record. ``episode_index`` falls back to the position in the input list.
    """

    path: Path
    task: str | None = None
    episode_index: int | None = None


@dataclass
class _ScannedEpisode:
    path: Path
    metadata: EpisodeMetadata
    state: _ScalarStream
    action: _ScalarStream
    camera_times_ns: dict[str, list[int]]


def convert_episodes(
    episode_mcaps: list[Path | EpisodeInput],
    output_dir: Path,
    repo_id: str,
    fps: int = 30,
    robot_type: str = "rebot_b601_follower",
    use_videos: bool = True,
    max_gap_s: float = 0.25,
    topics: list[str] | None = None,
    default_task: str | None = None,
) -> Path:
    """Write ``episode_mcaps`` into a new LeRobot dataset rooted at ``output_dir``.

    ``output_dir`` must not already exist. Episodes are written in ascending
    resolved ``episode_index``. State and action feature names follow the
    recorded label order, which must be identical across episodes, as must the
    camera list. ``default_task`` is the last fallback when a file has no
    metadata task and its :class:`EpisodeInput` has none.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}.")
    if max_gap_s <= 0:
        raise ValueError(f"max_gap_s must be positive, got {max_gap_s}.")
    sources = _episode_inputs(episode_mcaps)
    if not sources:
        raise ValueError("No episode MCAP files to convert.")
    allowed = set(topics) if topics is not None else None
    scanned = [
        _scan_episode(
            source, position, allowed, fps=fps, robot_type=robot_type, default_task=default_task
        )
        for position, source in enumerate(sources)
    ]
    scanned.sort(key=lambda episode: episode.metadata.episode_index)
    _validate_episode_set(scanned, fps=fps, robot_type=robot_type)
    for episode in scanned:
        _check_episode_gaps(episode, fps=fps, max_gap_s=max_gap_s)

    shapes = _first_frame_shapes(scanned[0])
    features = _features(
        state_names=scanned[0].metadata.state_names,
        shapes=shapes,
        use_videos=use_videos,
    )
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=output_dir,
        robot_type=robot_type,
        use_videos=use_videos,
        rgb_encoder=_rgb_encoder(),
    )
    try:
        for episode in scanned:
            _write_episode(dataset, episode, shapes)
    finally:
        dataset.finalize()
    return Path(dataset.root)


def _episode_inputs(episode_mcaps: list[Path | EpisodeInput]) -> list[EpisodeInput]:
    sources: list[EpisodeInput] = []
    for item in episode_mcaps:
        if isinstance(item, EpisodeInput):
            sources.append(item)
        else:
            sources.append(EpisodeInput(path=Path(item)))
    return sources


def _rgb_encoder() -> RGBEncoderConfig:
    if detect_available_encoders_pyav("libsvtav1"):
        return RGBEncoderConfig()
    if detect_available_encoders_pyav("h264"):
        logger.info("libsvtav1 is unavailable; encoding dataset videos with h264.")
        return RGBEncoderConfig(vcodec="h264")
    raise RuntimeError(
        "No LeRobot RGB video encoder is available in this PyAV build (tried libsvtav1 and h264)."
    )


def _features(
    *,
    state_names: tuple[str, ...],
    shapes: dict[str, tuple[int, ...]],
    use_videos: bool,
) -> dict[str, dict[str, object]]:
    names = list(state_names)
    features: dict[str, dict[str, object]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": names,
        },
        "action": {
            "dtype": "float32",
            "shape": (len(names),),
            "names": list(names),
        },
    }
    visual_dtype = "video" if use_videos else "image"
    for camera_name, shape in shapes.items():
        features[f"observation.images.{camera_name}"] = {
            "dtype": visual_dtype,
            "shape": tuple(shape),
            "names": ["height", "width", "channels"],
        }
    return features


def _scan_episode(
    source: EpisodeInput,
    position: int,
    allowed_topics: set[str] | None,
    *,
    fps: int,
    robot_type: str,
    default_task: str | None,
) -> _ScannedEpisode:
    path = source.path
    if not path.is_file():
        raise ValueError(f"Episode MCAP does not exist: {path}.")
    recorded: EpisodeMetadata | None = None
    state_times: list[int] = []
    state_rows: list[list[float]] = []
    state_labels: tuple[str, ...] | None = None
    action_times: list[int] = []
    action_rows: list[list[float]] = []
    action_labels: tuple[str, ...] | None = None
    camera_times: dict[str, list[int]] = {}
    with path.open("rb") as handle:
        reader = make_reader(handle)
        for record in reader.iter_metadata():
            if record.name == EPISODE_METADATA_NAME:
                recorded = EpisodeMetadata.from_mcap_metadata(record.metadata)
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            if allowed_topics is not None and topic not in allowed_topics:
                continue
            if topic in (TOPIC_OBSERVATION_STATE, TOPIC_ACTION_STATE):
                if schema is None or schema.name != SCALARS_SCHEMA_NAME:
                    raise ValueError(
                        f"{path} topic {topic} schema is {getattr(schema, 'name', None)!r}, "
                        f"expected {SCALARS_SCHEMA_NAME!r}."
                    )
                labels, values = _parse_scalars(message.data, topic, path)
                if topic == TOPIC_OBSERVATION_STATE:
                    state_labels = _expect_stable(state_labels, labels, path, topic)
                    state_times.append(int(message.log_time))
                    state_rows.append(values)
                else:
                    action_labels = _expect_stable(action_labels, labels, path, topic)
                    action_times.append(int(message.log_time))
                    action_rows.append(values)
                continue
            camera_name = camera_name_from_topic(topic)
            if camera_name is None:
                continue
            if schema is None or schema.name != COMPRESSED_VIDEO_SCHEMA_NAME:
                raise ValueError(
                    f"{path} topic {topic} schema is {getattr(schema, 'name', None)!r}, "
                    f"expected {COMPRESSED_VIDEO_SCHEMA_NAME!r}."
                )
            if channel.message_encoding != "json":
                raise ValueError(
                    f"{path} topic {topic} message encoding is {channel.message_encoding!r}. "
                    "This converter decodes JSON CompressedVideo messages "
                    "(base64 data, or raw bytes once decoded); protobuf bytes are not parsed."
                )
            camera_times.setdefault(camera_name, []).append(int(message.log_time))
    if state_labels is None or not state_rows:
        raise ValueError(f"{path} has no {TOPIC_OBSERVATION_STATE} messages.")
    if action_labels is None or not action_rows:
        raise ValueError(f"{path} has no {TOPIC_ACTION_STATE} messages.")
    if action_labels != state_labels:
        raise ValueError(
            f"{path} topic {TOPIC_ACTION_STATE} labels {list(action_labels)} do not match episode "
            f"state_names {list(state_labels)}."
        )
    if not camera_times:
        raise ValueError(f"{path} has no /observation/images/<name> messages.")
    camera_names = tuple(sorted(camera_times))
    if recorded is not None:
        if recorded.state_names != state_labels:
            raise ValueError(
                f"{path} topic {TOPIC_OBSERVATION_STATE} labels {list(state_labels)} "
                f"do not match episode state_names {list(recorded.state_names)}."
            )
        if set(recorded.camera_names) != set(camera_names):
            raise ValueError(
                f"{path} episode camera_names {list(recorded.camera_names)} do not match "
                f"image topics {list(camera_names)}."
            )
        if recorded.fps != fps:
            raise ValueError(
                f"Episode {recorded.episode_index} fps is {recorded.fps}, expected {fps}."
            )
        if recorded.robot_type != robot_type:
            raise ValueError(
                f"Episode {recorded.episode_index} robot_type is {recorded.robot_type!r}, "
                f"expected {robot_type!r}."
            )
    task = _resolve_task(path, recorded, source.task, default_task)
    if recorded is not None:
        episode_index = recorded.episode_index
        lerobot_version = recorded.lerobot_version
    elif source.episode_index is not None:
        episode_index = source.episode_index
        lerobot_version = "0.7.0"
    else:
        episode_index = position
        lerobot_version = "0.7.0"
    metadata = EpisodeMetadata(
        task=task,
        episode_index=episode_index,
        fps=fps,
        robot_type=robot_type,
        lerobot_version=lerobot_version,
        camera_names=camera_names,
        state_names=state_labels,
    )
    return _ScannedEpisode(
        path=path,
        metadata=metadata,
        state=_sorted_scalars(state_times, state_rows, state_labels),
        action=_sorted_scalars(action_times, action_rows, action_labels),
        camera_times_ns={name: sorted(camera_times[name]) for name in camera_names},
    )


def _nonempty(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value


def _resolve_task(
    path: Path,
    recorded: EpisodeMetadata | None,
    source_task: str | None,
    default_task: str | None,
) -> str:
    if recorded is not None:
        task = _nonempty(recorded.task)
        if task is not None:
            return task
    task = _nonempty(source_task)
    if task is not None:
        return task
    task = _nonempty(default_task)
    if task is not None:
        return task
    raise ValueError(f"{path} has no task. Pass --task or set a task episode property.")


def _parse_scalars(payload: bytes, topic: str, path: Path) -> tuple[tuple[str, ...], list[float]]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} topic {topic} is not JSON: {exc}") from exc
    scalars = decoded.get("scalars") if isinstance(decoded, dict) else None
    if not isinstance(scalars, list) or not scalars:
        raise ValueError(f"{path} topic {topic} is missing a non-empty scalars array.")
    labels: list[str] = []
    values: list[float] = []
    for item in scalars:
        if not isinstance(item, dict) or "label" not in item or "value" not in item:
            raise ValueError(
                f"{path} topic {topic} has a scalar entry that is not {{label, value}}."
            )
        labels.append(str(item["label"]))
        values.append(float(item["value"]))
    return tuple(labels), values


def _expect_stable(
    seen: tuple[str, ...] | None,
    labels: tuple[str, ...],
    path: Path,
    topic: str,
) -> tuple[str, ...]:
    if seen is not None and labels != seen:
        raise ValueError(f"{path} topic {topic} label order changed mid-episode.")
    return labels


def _sorted_scalars(
    times_ns: list[int],
    rows: list[list[float]],
    labels: tuple[str, ...],
) -> _ScalarStream:
    order = sorted(range(len(times_ns)), key=times_ns.__getitem__)
    times = [times_ns[index] for index in order]
    values = np.asarray([rows[index] for index in order], dtype=np.float32)
    return _ScalarStream(times_ns=times, values=values, labels=labels)


def _validate_episode_set(episodes: list[_ScannedEpisode], *, fps: int, robot_type: str) -> None:
    reference = episodes[0].metadata
    seen_indexes: set[int] = set()
    for episode in episodes:
        metadata = episode.metadata
        if metadata.episode_index in seen_indexes:
            raise ValueError(f"Duplicate episode_index {metadata.episode_index} in {episode.path}.")
        seen_indexes.add(metadata.episode_index)
        if metadata.fps != fps:
            raise ValueError(
                f"Episode {metadata.episode_index} fps is {metadata.fps}, expected {fps}."
            )
        if metadata.robot_type != robot_type:
            raise ValueError(
                f"Episode {metadata.episode_index} robot_type is {metadata.robot_type!r}, "
                f"expected {robot_type!r}."
            )
        if metadata.state_names != reference.state_names:
            raise ValueError(
                f"Episode {metadata.episode_index} state_names {list(metadata.state_names)} "
                f"do not match episode {reference.episode_index} {list(reference.state_names)}."
            )
        if metadata.camera_names != reference.camera_names:
            raise ValueError(
                f"Episode {metadata.episode_index} camera_names {list(metadata.camera_names)} "
                f"do not match episode {reference.episode_index} {list(reference.camera_names)}."
            )


def _overlap(episode: _ScannedEpisode) -> tuple[int, int]:
    streams: list[list[int]] = [
        episode.state.times_ns,
        episode.action.times_ns,
        *episode.camera_times_ns.values(),
    ]
    t0 = max(times[0] for times in streams)
    t1 = min(times[-1] for times in streams)
    if t1 < t0:
        raise ValueError(
            f"Episode {episode.metadata.episode_index} ({episode.path.name}) has no overlapping "
            "time range across state, action, and cameras."
        )
    return t0, t1


def _grid_times_ns(t0: int, t1: int, fps: int) -> list[int]:
    times: list[int] = []
    index = 0
    while True:
        stamp = t0 + round(index * _NS / fps)
        if stamp > t1:
            return times
        times.append(stamp)
        index += 1


def _check_episode_gaps(episode: _ScannedEpisode, *, fps: int, max_gap_s: float) -> None:
    t0, t1 = _overlap(episode)
    grid = _grid_times_ns(t0, t1, fps)
    if not grid:
        raise ValueError(f"Episode {episode.metadata.episode_index} produced an empty time grid.")
    topics: list[tuple[str, list[int]]] = [
        (TOPIC_OBSERVATION_STATE, episode.state.times_ns),
        (TOPIC_ACTION_STATE, episode.action.times_ns),
    ]
    for camera_name, times in episode.camera_times_ns.items():
        topics.append((f"/observation/images/{camera_name}", times))
    max_gap_ns = int(max_gap_s * _NS)
    for topic, times in topics:
        _check_gaps(times, t0, t1, max_gap_ns, episode, topic)


def _check_gaps(
    times_ns: list[int],
    t0: int,
    t1: int,
    max_gap_ns: int,
    episode: _ScannedEpisode,
    topic: str,
) -> None:
    start = bisect_right(times_ns, t0) - 1
    if start < 0:
        raise ValueError(
            f"Episode {episode.metadata.episode_index} topic {topic} has no sample at or before "
            f"t={t0 / _NS:.6f}s."
        )
    window = [times_ns[start]]
    window.extend(stamp for stamp in times_ns[start + 1 :] if stamp <= t1)
    for previous, nxt in zip(window, window[1:], strict=False):
        gap_ns = nxt - previous
        if gap_ns > max_gap_ns:
            raise ValueError(
                f"Episode {episode.metadata.episode_index} ({episode.path.name}) topic {topic}: "
                f"gap of {gap_ns / _NS:.3f}s exceeds max_gap_s={max_gap_ns / _NS:.3f}s "
                f"at t={nxt / _NS:.6f}s."
            )


def _hold(times_ns: list[int], values: np.ndarray, query_ns: int) -> np.ndarray:
    index = bisect_right(times_ns, query_ns) - 1
    if index < 0:
        raise ValueError(f"No sample at or before t={query_ns / _NS:.6f}s.")
    return values[index]


_NO_PEEK: Any = object()


def _first_frame_shapes(episode: _ScannedEpisode) -> dict[str, tuple[int, ...]]:
    """Open a fresh decoder per camera and keep only the first image's shape."""
    shapes: dict[str, tuple[int, ...]] = {}
    missing: list[str] = []
    for camera_name in episode.metadata.camera_names:
        frames = _camera_frames(episode, camera_name)
        try:
            try:
                _stamp, image = next(frames)
            except StopIteration:
                missing.append(camera_name)
                continue
            shapes[camera_name] = tuple(int(dim) for dim in image.shape)
        finally:
            frames.close()
    if missing:
        raise ValueError(
            f"Episode {episode.metadata.episode_index} decoded no frames for {missing}."
        )
    return shapes


def _camera_frames(episode: _ScannedEpisode, camera_name: str) -> Iterator[tuple[int, np.ndarray]]:
    topic = f"/observation/images/{camera_name}"

    def _wrapped() -> Iterator[tuple[int, np.ndarray]]:
        try:
            yield from iter_decode_h264_messages(_iter_video_messages(episode, topic))
        except ValueError as exc:
            text = str(exc)
            if text.startswith("Episode "):
                raise
            raise ValueError(
                f"Episode {episode.metadata.episode_index} ({episode.path.name}) "
                f"topic {topic}: {exc}"
            ) from exc

    return _wrapped()


def _iter_video_messages(
    episode: _ScannedEpisode, topic: str
) -> Iterator[tuple[int, dict[str, Any]]]:
    with episode.path.open("rb") as handle:
        reader = make_reader(handle)
        for schema, channel, message in reader.iter_messages(topics=[topic]):
            if schema is None or schema.name != COMPRESSED_VIDEO_SCHEMA_NAME:
                raise ValueError(
                    f"{episode.path} topic {topic} schema is {getattr(schema, 'name', None)!r}, "
                    f"expected {COMPRESSED_VIDEO_SCHEMA_NAME!r}."
                )
            if channel.message_encoding != "json":
                raise ValueError(
                    f"{episode.path} topic {topic} message encoding is "
                    f"{channel.message_encoding!r}. This converter decodes JSON "
                    "CompressedVideo messages (base64 data, or raw bytes once decoded); "
                    "protobuf bytes are not parsed."
                )
            try:
                payload = json.loads(message.data)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Episode {episode.metadata.episode_index} topic {topic} is not JSON: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Episode {episode.metadata.episode_index} topic {topic} was not a JSON object."
                )
            yield int(message.log_time), payload


class _FrameCursor:
    """Zero-order hold over a monotonic decoded stream. Keeps one image."""

    def __init__(
        self,
        frames: Iterator[tuple[int, np.ndarray]],
        *,
        episode_index: int,
        topic: str,
    ) -> None:
        self._frames = frames
        self._episode_index = episode_index
        self._topic = topic
        self._peek: tuple[int, np.ndarray] | None | Any = _NO_PEEK
        self._current: tuple[int, np.ndarray] | None = None

    def close(self) -> None:
        close = getattr(self._frames, "close", None)
        if close is not None:
            close()

    def frame_at(self, stamp: int, shape: tuple[int, ...]) -> np.ndarray:
        if self._current is None:
            first = self._pull()
            if first is None or first[0] > stamp:
                raise ValueError(
                    f"Episode {self._episode_index} topic {self._topic} has no sample "
                    f"at or before t={stamp / _NS:.6f}s."
                )
            self._remember(first, shape)
        while True:
            upcoming = self._peek_one()
            if upcoming is None or upcoming[0] > stamp:
                break
            self._remember(upcoming, shape)
            self._peek = _NO_PEEK
        assert self._current is not None
        return self._current[1]

    def drain(self, shape: tuple[int, ...]) -> None:
        """Finish the decoder so a short or reordered stream still raises."""
        upcoming = self._peek_one()
        while upcoming is not None:
            self._remember(upcoming, shape)
            self._peek = _NO_PEEK
            upcoming = self._peek_one()
        self._current = None

    def _remember(self, item: tuple[int, np.ndarray], shape: tuple[int, ...]) -> None:
        image = item[1]
        if tuple(image.shape) != tuple(shape):
            raise ValueError(
                f"Episode {self._episode_index} camera {self._topic.rsplit('/', 1)[-1]} "
                f"frame shape {image.shape} does not match {shape}."
            )
        self._current = item

    def _peek_one(self) -> tuple[int, np.ndarray] | None:
        if self._peek is _NO_PEEK:
            self._peek = self._pull()
        peeked = self._peek
        if peeked is None or peeked is _NO_PEEK:
            return None
        return peeked

    def _pull(self) -> tuple[int, np.ndarray] | None:
        try:
            return next(self._frames)
        except StopIteration:
            return None


def _write_episode(
    dataset: LeRobotDataset,
    episode: _ScannedEpisode,
    shapes: dict[str, tuple[int, ...]],
) -> None:
    t0, t1 = _overlap(episode)
    grid = _grid_times_ns(t0, t1, episode.metadata.fps)
    cursors = {
        camera_name: _FrameCursor(
            _camera_frames(episode, camera_name),
            episode_index=episode.metadata.episode_index,
            topic=f"/observation/images/{camera_name}",
        )
        for camera_name in episode.metadata.camera_names
    }
    task = episode.metadata.task
    try:
        for stamp in grid:
            frame: dict[str, object] = {
                "task": task,
                "observation.state": _hold(episode.state.times_ns, episode.state.values, stamp),
                "action": _hold(episode.action.times_ns, episode.action.values, stamp),
            }
            for camera_name, cursor in cursors.items():
                frame[f"observation.images.{camera_name}"] = cursor.frame_at(
                    stamp, shapes[camera_name]
                )
            dataset.add_frame(frame)
        for camera_name, cursor in cursors.items():
            cursor.drain(shapes[camera_name])
    finally:
        for cursor in cursors.values():
            cursor.close()
    dataset.save_episode()
    logger.info(
        "Wrote episode %s (%d frames) from %s.",
        episode.metadata.episode_index,
        len(grid),
        episode.path.name,
    )
