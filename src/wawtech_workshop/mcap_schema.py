"""MCAP topics and metadata shared by the recorder and the dataset converter.

Scalar topics use the ``lerobot.Scalars`` JSON schema from LeRobot's Foxglove
integration. Camera topics carry ``foxglove.CompressedVideo`` messages with
``format`` ``h264``. The recorder writes an ``episode`` metadata record. Episodes
read through the Foxglove SDK's dataset reader carry messages only, so the
converter does not depend on that record.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

TOPIC_OBSERVATION_STATE = "/observation/state"
TOPIC_ACTION_STATE = "/action/state"
# For the 3D panel: the follower's URDF joint angles (radians) and a fixed base frame.
TOPIC_JOINT_STATES = "/joint_states"
# The leader's action in the same form, for a ghost arm next to the follower.
TOPIC_ACTION_JOINT_STATES = "/action/joint_states"
TOPIC_TF = "/tf"
EPISODE_METADATA_NAME = "episode"
# Recorder-only record: outcome label and session bookkeeping. The converter ignores it.
RECORDING_METADATA_NAME = "recording"
SCALARS_SCHEMA_NAME = "lerobot.Scalars"
COMPRESSED_VIDEO_SCHEMA_NAME = "foxglove.CompressedVideo"

# Reproduced from LeRobot's Foxglove integration (foxglove-sdk so101 example
# ``SCALARS_SCHEMA`` and ``lerobot.utils.foxglove_visualization``).
SCALARS_SCHEMA: dict[str, object] = {
    "type": "object",
    "title": "lerobot.Scalars",
    "properties": {
        "scalars": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "number"},
                },
            },
        }
    },
}

COMPRESSED_VIDEO_SCHEMA: dict[str, object] = {
    "title": "foxglove.CompressedVideo",
    "description": "A single frame of a compressed video bitstream",
    "type": "object",
    "properties": {
        "timestamp": {
            "type": "object",
            "title": "time",
            "properties": {
                "sec": {"type": "integer", "minimum": 0},
                "nsec": {"type": "integer", "minimum": 0, "maximum": 999999999},
            },
            "description": "Timestamp of video frame",
        },
        "frame_id": {"type": "string"},
        "data": {
            "type": "string",
            "contentEncoding": "base64",
            "description": (
                "Annex B H.264 data for exactly one image. Keyframes include SPS and PPS."
            ),
        },
        "format": {"type": "string", "description": "Video format. This workshop writes h264."},
    },
    "required": ["timestamp", "frame_id", "data", "format"],
}


def image_topic(camera_name: str) -> str:
    """Return the compressed-video topic for one camera."""
    if not camera_name or "/" in camera_name:
        raise ValueError(f"Camera name {camera_name!r} is not a single topic segment.")
    return f"/observation/images/{camera_name}"


def camera_name_from_topic(topic: str) -> str | None:
    """Return the camera name for an image topic, or None if ``topic`` is not one."""
    prefix = "/observation/images/"
    if not topic.startswith(prefix):
        return None
    name = topic[len(prefix) :]
    if not name or "/" in name:
        return None
    return name


def _split_csv(value: str, field: str) -> list[str]:
    parts = [part.strip() for part in value.split(",")]
    if not parts or any(not part for part in parts):
        raise ValueError(
            f"Episode metadata field {field!r} must be a comma-separated list, got {value!r}."
        )
    return parts


# Shared by the synthetic writer and the recorder. ``decode_h264_messages``
# accepts one Annex B image per message and rejects B-frames, so the encoder
# must not reorder or buffer pictures.
H264_ENCODER_OPTIONS: dict[str, str] = {
    "bf": "0",
    "g": "1",
    "preset": "ultrafast",
    "tune": "zerolatency",
    "crf": "28",
}


def h264_codec_options(codec_name: str) -> dict[str, str]:
    """Return encoder options that ``decode_h264_messages`` can consume.

    ``bf=0`` disables B-frames. The decoder raises if a stream still has them.
    ``g=1`` forces a keyframe on every picture so each message yields one image
    without waiting for a later keyframe. ``tune=zerolatency`` stops the encoder
    from holding a frame back, which would make one message produce no image and
    a later message produce two. ``preset`` and ``crf`` only trade speed and
    quality; they are fixed so every writer emits the same stream. For
    ``libx264``, ``x264-params`` repeats that contract inside x264: Annex B
    start codes, SPS and PPS on every keyframe, and no B-frames or scene-cut
    keyframe delays. Other H.264 encoders do not accept ``x264-params``.
    """
    options = dict(H264_ENCODER_OPTIONS)
    if codec_name == "libx264":
        options["x264-params"] = (
            "annexb=1:repeat-headers=1:keyint=1:min-keyint=1:scenecut=0:bframes=0"
        )
    return options


@dataclass(frozen=True)
class EpisodeMetadata:
    """File-level ``episode`` metadata record.

    MCAP metadata values are strings. ``camera_names`` and ``state_names`` are
    stored comma-separated, in the order samples are written. The converter
    treats the record as optional. When the record is present it is
    authoritative for ``task`` and ``episode_index`` and is checked against the
    labels and camera topics in the file. Whether Foxglove keeps the record
    when it cuts a recording is unverified.
    """

    task: str
    episode_index: int
    fps: int
    robot_type: str
    lerobot_version: str
    camera_names: tuple[str, ...]
    state_names: tuple[str, ...]

    def to_mcap_metadata(self) -> dict[str, str]:
        if not self.camera_names or not self.state_names:
            raise ValueError("Episode metadata needs at least one camera and one state label.")
        return {
            "task": self.task,
            "episode_index": str(self.episode_index),
            "fps": str(self.fps),
            "robot_type": self.robot_type,
            "lerobot_version": self.lerobot_version,
            "camera_names": ",".join(self.camera_names),
            "state_names": ",".join(self.state_names),
        }

    @classmethod
    def from_mcap_metadata(cls, data: Mapping[str, str]) -> EpisodeMetadata:
        missing = [
            key
            for key in (
                "task",
                "episode_index",
                "fps",
                "robot_type",
                "lerobot_version",
                "camera_names",
                "state_names",
            )
            if key not in data
        ]
        if missing:
            raise ValueError(f"Episode metadata is missing keys: {missing}.")
        try:
            episode_index = int(data["episode_index"])
            fps = int(data["fps"])
        except ValueError as exc:
            raise ValueError(
                "Episode metadata episode_index and fps must be integers, "
                f"got {data['episode_index']!r} and {data['fps']!r}."
            ) from exc
        if fps <= 0:
            raise ValueError(f"Episode metadata fps must be positive, got {fps}.")
        return cls(
            task=str(data["task"]),
            episode_index=episode_index,
            fps=fps,
            robot_type=str(data["robot_type"]),
            lerobot_version=str(data["lerobot_version"]),
            camera_names=tuple(_split_csv(str(data["camera_names"]), "camera_names")),
            state_names=tuple(_split_csv(str(data["state_names"]), "state_names")),
        )
