"""Write a synthetic episode MCAP that follows the workshop topic conventions.

Joint state is a sine wave. Action is the same wave evaluated ``ACTION_LEAD_S``
later, sampled at the same timestamps. Timestamps sit near ``fps`` with a
sub-frame jitter so the converter's zero-order hold has something to do.
Images are tiny RGB frames encoded as Annex B H.264, one image per message.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from mcap.writer import Writer

from wawtech_workshop.h264 import encode_h264_frames
from wawtech_workshop.mcap_schema import (
    COMPRESSED_VIDEO_SCHEMA,
    COMPRESSED_VIDEO_SCHEMA_NAME,
    EPISODE_METADATA_NAME,
    SCALARS_SCHEMA,
    SCALARS_SCHEMA_NAME,
    TOPIC_ACTION_STATE,
    TOPIC_OBSERVATION_STATE,
    EpisodeMetadata,
    image_topic,
)

ACTION_LEAD_S = 0.01
FRAME_HEIGHT = 48
FRAME_WIDTH = 64


@dataclass(frozen=True)
class SyntheticEpisode:
    path: Path
    metadata: EpisodeMetadata
    times_ns: list[int]
    state: np.ndarray
    action: np.ndarray
    height: int
    width: int


def sample_times_ns(
    n_frames: int,
    *,
    t0_ns: int,
    fps: int,
    jitter_ns: int = 500_000,
    gap_after: int | None = None,
    gap_ns: int = 500_000_000,
) -> list[int]:
    """Timestamps near ``t0 + i/fps``, strictly increasing.

    When ``gap_after`` is set, an extra ``gap_ns`` is inserted after that sample.
    """
    step = 1_000_000_000 / fps
    times: list[int] = []
    extra = 0
    for index in range(n_frames):
        if gap_after is not None and index > gap_after:
            extra = gap_ns
        sign = 1 if index % 2 == 0 else -1
        stamp = int(round(t0_ns + index * step + sign * jitter_ns + extra))
        if times and stamp <= times[-1]:
            stamp = times[-1] + 1
        times.append(stamp)
    return times


def joint_signal(times_s: np.ndarray, n_joints: int) -> np.ndarray:
    columns = [np.sin(2.0 * np.pi * 0.5 * times_s + joint_index) for joint_index in range(n_joints)]
    return np.stack(columns, axis=-1).astype(np.float64)


def make_frame(index: int, height: int = FRAME_HEIGHT, width: int = FRAME_WIDTH) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = np.uint8((index * 40) % 256)
    frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    frame[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return np.ascontiguousarray(frame)


def write_synthetic_episode(
    path: Path,
    *,
    n_frames: int = 8,
    fps: int = 30,
    task: str = "pick up the block",
    episode_index: int = 0,
    robot_type: str = "rebot_b601_follower",
    lerobot_version: str = "0.7.0",
    camera_names: tuple[str, ...] = ("front",),
    state_names: tuple[str, ...] = ("joint1.pos", "joint2.pos"),
    t0_ns: int = 1_700_000_000_000_000_000,
    gap_after: int | None = None,
    gap_ns: int = 500_000_000,
    height: int = FRAME_HEIGHT,
    width: int = FRAME_WIDTH,
    include_metadata: bool = True,
    metadata_camera_names: tuple[str, ...] | None = None,
) -> SyntheticEpisode:
    """Write one episode MCAP and return the samples that were stored."""
    times_ns = sample_times_ns(n_frames, t0_ns=t0_ns, fps=fps, gap_after=gap_after, gap_ns=gap_ns)
    times_s = np.asarray(times_ns, dtype=np.float64) / 1_000_000_000
    state = joint_signal(times_s, len(state_names))
    action = joint_signal(times_s + ACTION_LEAD_S, len(state_names))
    frames = [make_frame(index, height, width) for index in range(n_frames)]
    packets = encode_h264_frames(frames, fps)
    if len(packets) != n_frames:
        raise RuntimeError(f"Expected {n_frames} H.264 packets, encoded {len(packets)}.")
    metadata = EpisodeMetadata(
        task=task,
        episode_index=episode_index,
        fps=fps,
        robot_type=robot_type,
        lerobot_version=lerobot_version,
        camera_names=camera_names,
        state_names=state_names,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start()
        scalar_schema = writer.register_schema(
            name=SCALARS_SCHEMA_NAME,
            encoding="jsonschema",
            data=json.dumps(SCALARS_SCHEMA).encode(),
        )
        video_schema = writer.register_schema(
            name=COMPRESSED_VIDEO_SCHEMA_NAME,
            encoding="jsonschema",
            data=json.dumps(COMPRESSED_VIDEO_SCHEMA).encode(),
        )
        state_channel = writer.register_channel(
            topic=TOPIC_OBSERVATION_STATE,
            message_encoding="json",
            schema_id=scalar_schema,
        )
        action_channel = writer.register_channel(
            topic=TOPIC_ACTION_STATE,
            message_encoding="json",
            schema_id=scalar_schema,
        )
        video_channels = {
            name: writer.register_channel(
                topic=image_topic(name),
                message_encoding="json",
                schema_id=video_schema,
            )
            for name in camera_names
        }
        for index, stamp in enumerate(times_ns):
            writer.add_message(
                state_channel,
                stamp,
                _scalars_payload(state_names, state[index]).encode(),
                stamp,
            )
            writer.add_message(
                action_channel,
                stamp,
                _scalars_payload(state_names, action[index]).encode(),
                stamp,
            )
            sec, nsec = divmod(stamp, 1_000_000_000)
            video_message = {
                "timestamp": {"sec": int(sec), "nsec": int(nsec)},
                "frame_id": camera_names[0],
                "data": base64.b64encode(packets[index]).decode("ascii"),
                "format": "h264",
            }
            encoded = json.dumps(video_message).encode()
            for channel_id in video_channels.values():
                writer.add_message(channel_id, stamp, encoded, stamp)
        if include_metadata:
            written = metadata
            if metadata_camera_names is not None:
                written = EpisodeMetadata(
                    task=metadata.task,
                    episode_index=metadata.episode_index,
                    fps=metadata.fps,
                    robot_type=metadata.robot_type,
                    lerobot_version=metadata.lerobot_version,
                    camera_names=metadata_camera_names,
                    state_names=metadata.state_names,
                )
            writer.add_metadata(name=EPISODE_METADATA_NAME, data=written.to_mcap_metadata())
        writer.finish()
    return SyntheticEpisode(
        path=path,
        metadata=metadata,
        times_ns=times_ns,
        state=state.astype(np.float32),
        action=action.astype(np.float32),
        height=height,
        width=width,
    )


def _scalars_payload(labels: tuple[str, ...], values: np.ndarray) -> str:
    return json.dumps(
        {
            "scalars": [
                {"label": label, "value": float(value)}
                for label, value in zip(labels, values, strict=True)
            ]
        }
    )
