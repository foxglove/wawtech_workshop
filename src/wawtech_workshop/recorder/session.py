"""Write teleop episodes to MCAP with the Foxglove SDK.

A :class:`RecordingSession` owns one Foxglove SDK :class:`foxglove.Context` with
one channel per topic the converter reads:

* ``/observation/state`` and ``/action/state``: ``lerobot.Scalars`` JSON, labels
  in the robot's joint order (``shoulder_pan.pos`` ...).
* ``/observation/images/<camera>``: ``foxglove.CompressedVideo`` JSON, H.264,
  one keyframe per message (see :mod:`wawtech_workshop.h264`).

Each episode opens an MCAP sink on that context, so only messages logged
between :meth:`RecordingSession.start_episode` and
:meth:`RecordingSession.finish_episode` land in the file. The recorder CLI
also serves the same context over a Foxglove WebSocket server (on by default),
so every message logged reaches both the MCAP file and the live view in
Foxglove, which keeps streaming between episodes for teleop.

H.264 encoding runs on one worker thread per camera so the control loop keeps
its rate. A worker queue holds at most one second of frames; a frame that does
not fit is dropped and counted in the ``recording`` metadata record. The
converter rejects gaps over 0.25 s, so a long run of drops shows up there.
"""

from __future__ import annotations

import base64
import json
import logging
import queue
import re
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import foxglove
import numpy as np

from wawtech_workshop.h264 import H264FrameEncoder
from wawtech_workshop.mcap_schema import (
    COMPRESSED_VIDEO_SCHEMA,
    EPISODE_METADATA_NAME,
    RECORDING_METADATA_NAME,
    SCALARS_SCHEMA,
    TOPIC_ACTION_STATE,
    TOPIC_OBSERVATION_STATE,
    EpisodeMetadata,
    image_topic,
)

logger = logging.getLogger(__name__)

_NS = 1_000_000_000
EPISODE_GLOB = "episode_*.mcap"


def episode_filename(episode_index: int, session_id: str) -> str:
    return f"episode_{episode_index:04d}_{session_id}.mcap"


def next_episode_index(output_dir: Path) -> int:
    """One past the highest ``episode_NNNN_*.mcap`` index in ``output_dir``, or 0."""
    highest = -1
    for path in output_dir.glob(EPISODE_GLOB):
        parts = path.stem.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            highest = max(highest, int(parts[1]))
    return highest + 1


def default_session_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


class _CameraWorker:
    """Encode one camera's frames off the control thread and log them."""

    def __init__(self, name: str, channel: foxglove.Channel, fps: int) -> None:
        self.name = name
        self.channel = channel
        self._fps = fps
        self._queue: queue.Queue[tuple[int, np.ndarray] | None] = queue.Queue(maxsize=max(1, fps))
        self._encoder: H264FrameEncoder | None = None
        self._error: BaseException | None = None
        self.dropped = 0
        self.encoded = 0
        self._thread = threading.Thread(target=self._run, name=f"h264-{name}", daemon=True)
        self._thread.start()

    def submit(self, log_time_ns: int, rgb: np.ndarray) -> None:
        self.raise_error()
        try:
            self._queue.put_nowait((log_time_ns, rgb))
        except queue.Full:
            self.dropped += 1

    def drain(self) -> None:
        """Block until every queued frame is logged, then surface any encoder error."""
        self._queue.join()
        self.raise_error()

    def reset_counters(self) -> None:
        self.dropped = 0
        self.encoded = 0

    def stop(self) -> None:
        self._queue.put(None)
        self._thread.join(timeout=10)

    def raise_error(self) -> None:
        if self._error is not None:
            error, self._error = self._error, None
            raise RuntimeError(f"Camera {self.name} encoder failed: {error}") from error

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                if self._error is None:
                    self._encode_and_log(*item)
            except BaseException as exc:  # surfaced on the control thread
                self._error = exc
            finally:
                self._queue.task_done()

    def _encode_and_log(self, log_time_ns: int, rgb: np.ndarray) -> None:
        height, width = int(rgb.shape[0]), int(rgb.shape[1])
        if self._encoder is None or (self._encoder.height, self._encoder.width) != (height, width):
            self._encoder = H264FrameEncoder(width, height, self._fps)
        data = self._encoder.encode(rgb)
        sec, nsec = divmod(log_time_ns, _NS)
        self.channel.log(
            {
                "timestamp": {"sec": int(sec), "nsec": int(nsec)},
                "frame_id": self.name,
                "data": base64.b64encode(data).decode("ascii"),
                "format": "h264",
            },
            log_time=log_time_ns,
        )
        self.encoded += 1


@dataclass
class _OpenEpisode:
    path: Path
    writer: foxglove.mcap.MCAPWriter
    episode_index: int
    task: str
    started_ns: int
    steps: int = 0
    first_ns: int | None = None
    last_ns: int | None = None


@dataclass
class EpisodeResult:
    path: Path
    episode_index: int
    outcome: str
    steps: int
    duration_s: float
    dropped_frames: dict[str, int] = field(default_factory=dict)


class RecordingSession:
    """Channels, camera workers and the per-episode MCAP sink for one recording run."""

    def __init__(
        self,
        *,
        output_dir: Path,
        fps: int,
        robot_type: str,
        state_names: tuple[str, ...],
        camera_names: tuple[str, ...],
        lerobot_version: str,
        session_id: str | None = None,
        context: foxglove.Context | None = None,
    ) -> None:
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}.")
        if not state_names:
            raise ValueError("At least one state label is required.")
        if not camera_names:
            raise ValueError("At least one RGB camera is required; the converter needs images.")
        self.output_dir = output_dir
        self.fps = fps
        self.robot_type = robot_type
        self.state_names = tuple(state_names)
        self.camera_names = tuple(camera_names)
        self.lerobot_version = lerobot_version
        self.session_id = session_id or default_session_id()
        if not re.fullmatch(r"[A-Za-z0-9-]+", self.session_id):
            raise ValueError(
                f"session_id {self.session_id!r} may only use letters, digits and '-'."
            )
        self.context = context or foxglove.Context()
        self._state = foxglove.Channel(
            TOPIC_OBSERVATION_STATE, schema=SCALARS_SCHEMA, context=self.context
        )
        self._action = foxglove.Channel(
            TOPIC_ACTION_STATE, schema=SCALARS_SCHEMA, context=self.context
        )
        self._cameras = {
            name: _CameraWorker(
                name,
                foxglove.Channel(
                    image_topic(name), schema=COMPRESSED_VIDEO_SCHEMA, context=self.context
                ),
                fps,
            )
            for name in self.camera_names
        }
        self._episode: _OpenEpisode | None = None

    @property
    def recording(self) -> bool:
        return self._episode is not None

    def log_step(
        self,
        log_time_ns: int,
        state: Mapping[str, float],
        action: Mapping[str, float],
        images: Mapping[str, np.ndarray],
    ) -> None:
        """Log one control tick. Values are looked up by label; extra keys are ignored."""
        self._state.log(_scalars(self.state_names, state, "state"), log_time=log_time_ns)
        self._action.log(_scalars(self.state_names, action, "action"), log_time=log_time_ns)
        for name, worker in self._cameras.items():
            if name not in images:
                raise KeyError(f"Observation has no image for camera {name!r}.")
            if worker.channel.has_sinks():
                worker.submit(log_time_ns, images[name])
        episode = self._episode
        if episode is not None:
            episode.steps += 1
            episode.first_ns = log_time_ns if episode.first_ns is None else episode.first_ns
            episode.last_ns = log_time_ns

    def start_episode(self, *, task: str, episode_index: int) -> Path:
        if self._episode is not None:
            raise RuntimeError("An episode is already being recorded.")
        if not task.strip():
            raise ValueError("Task must be a non-empty string.")
        # Frames queued for the live view must not land in the new file.
        for worker in self._cameras.values():
            worker.drain()
            worker.reset_counters()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / episode_filename(episode_index, self.session_id)
        writer = foxglove.open_mcap(path, context=self.context)
        self._episode = _OpenEpisode(
            path=path,
            writer=writer,
            episode_index=episode_index,
            task=task,
            started_ns=time.time_ns(),
        )
        return path

    def finish_episode(self, outcome: str) -> EpisodeResult:
        episode = self._require_episode()
        self._drain_cameras()
        dropped = {name: worker.dropped for name, worker in self._cameras.items()}
        metadata = EpisodeMetadata(
            task=episode.task,
            episode_index=episode.episode_index,
            fps=self.fps,
            robot_type=self.robot_type,
            lerobot_version=self.lerobot_version,
            camera_names=self.camera_names,
            state_names=self.state_names,
        )
        duration_s = (
            (episode.last_ns - episode.first_ns) / _NS
            if episode.first_ns is not None and episode.last_ns is not None
            else 0.0
        )
        try:
            episode.writer.write_metadata(EPISODE_METADATA_NAME, metadata.to_mcap_metadata())
            episode.writer.write_metadata(
                RECORDING_METADATA_NAME,
                {
                    "outcome": outcome,
                    "session_id": self.session_id,
                    "started_at": datetime.fromtimestamp(episode.started_ns / _NS, UTC).isoformat(),
                    "steps": str(episode.steps),
                    "duration_s": f"{duration_s:.3f}",
                    "dropped_frames": json.dumps(dropped, sort_keys=True),
                },
            )
        finally:
            episode.writer.close()
            self._episode = None
        if any(dropped.values()):
            logger.warning(
                "Episode %d dropped camera frames %s; the encoder could not keep up.",
                episode.episode_index,
                dropped,
            )
        return EpisodeResult(
            path=episode.path,
            episode_index=episode.episode_index,
            outcome=outcome,
            steps=episode.steps,
            duration_s=duration_s,
            dropped_frames=dropped,
        )

    def discard_episode(self) -> Path:
        episode = self._require_episode()
        try:
            self._drain_cameras()
        finally:
            episode.writer.close()
            self._episode = None
            episode.path.unlink(missing_ok=True)
        return episode.path

    def close(self) -> None:
        try:
            if self._episode is not None:
                self.discard_episode()
        finally:
            for worker in self._cameras.values():
                worker.stop()

    def __enter__(self) -> RecordingSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _require_episode(self) -> _OpenEpisode:
        if self._episode is None:
            raise RuntimeError("No episode is being recorded.")
        return self._episode

    def _drain_cameras(self) -> None:
        for worker in self._cameras.values():
            worker.drain()


def _scalars(labels: tuple[str, ...], values: Mapping[str, float], kind: str) -> dict:
    missing = [label for label in labels if label not in values]
    if missing:
        raise KeyError(f"{kind} is missing joints {missing}.")
    return {"scalars": [{"label": label, "value": float(values[label])} for label in labels]}
