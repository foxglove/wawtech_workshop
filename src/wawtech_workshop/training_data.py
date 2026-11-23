"""Stream a Foxglove dataset version into PyTorch training samples.

``build_torch_dataset`` wraps ``foxglove.datasets.torch.read_dataset``: episodes
are streamed while the DataLoader iterates and are split across its workers, so
nothing is converted or stored on disk first. ``read_episode`` turns one episode
into samples with the keys and units LeRobot policies (ACT, π₀.₅) read, the same
ones ``build-dataset`` writes:

``observation.state``   float32 ``[S]``, from ``/observation/state``
``action``              float32 ``[S]``, or ``[chunk_size, S]`` when ``chunk_size > 1``
                        (with a bool ``action_is_pad`` of ``[chunk_size]``)
``observation.images.<camera>``
                        float32 ``[3, H, W]`` in ``[0, 1]``, from ``/observation/images/<camera>``
``task``                the episode's ``task`` property (or ``default_task``)

plus ``episode_id``, ``frame_index`` and ``timestamp`` (seconds since the first sample).

Alignment matches ``dataset/convert.py``: a fixed ``fps`` grid starting at the latest
first timestamp of state, action and camera, ending at the earliest last timestamp, with
zero-order hold (each topic contributes its latest sample at or before the grid time).
Episodes whose topics have a gap longer than ``max_gap_s`` are skipped with a warning
instead of holding a stale value across the gap. Episodes are filtered by the ``outcome``
property like ``build-dataset`` does.

Only one camera frame is decoded at a time, so memory use does not grow with episode length.

DataLoader shuffling and ``DistributedSampler`` do not apply to an IterableDataset. Pass
``shuffle=True`` to shuffle episode order. Under DDP, workers get unequal sample counts, so
wrap the training step in ``torch.distributed.algorithms.join.Join``.

Run ``python -m wawtech_workshop.training_data`` for a smoke test that prints batch shapes.
Set ``FOXGLOVE_API_TOKEN`` first; the key needs the ``datasets.list`` and ``data.stream``
capabilities.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Any

import numpy as np
import torch
from foxglove.datasets import EpisodeReader
from foxglove.datasets.torch import read_dataset
from foxglove.datasets.video import decode_h264

from wawtech_workshop.dataset.foxglove_source import make_client, outcome_of
from wawtech_workshop.mcap_schema import TOPIC_ACTION_STATE, TOPIC_OBSERVATION_STATE, image_topic

logger = logging.getLogger(__name__)

DEFAULT_DATASET_ID = "dset_0ek8nUJE8FYkiGun"  # "butter_pass"
DEFAULT_VERSION = 2

_NS = 1_000_000_000

Sample = dict[str, Any]


@dataclass(frozen=True)
class SampleConfig:
    """How episodes become samples. Pickled to DataLoader workers, so keep it plain data."""

    camera: str = "wrist"
    fps: int = 30
    chunk_size: int = 1
    max_gap_s: float = 0.25
    default_task: str | None = None
    outcome_key: str = "outcome"
    keep_outcomes: tuple[str, ...] = ("success",)
    keep_unlabeled: bool = False
    lookback_s: float = 5.0

    def __post_init__(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        if self.chunk_size < 1:
            raise ValueError(f"chunk_size must be at least 1, got {self.chunk_size}.")
        if self.max_gap_s <= 0:
            raise ValueError(f"max_gap_s must be positive, got {self.max_gap_s}.")

    @property
    def topics(self) -> list[str]:
        return [TOPIC_OBSERVATION_STATE, TOPIC_ACTION_STATE, image_topic(self.camera)]


def build_torch_dataset(
    dataset_id: str = DEFAULT_DATASET_ID,
    version: int = DEFAULT_VERSION,
    config: SampleConfig | None = None,
    **read_dataset_kwargs: Any,
) -> torch.utils.data.IterableDataset:
    """Return an ``IterableDataset`` that streams ``version`` of ``dataset_id``.

    ``read_dataset_kwargs`` go to ``read_dataset`` (for example ``shuffle=True, seed=0`` or
    ``split_config`` and ``split``). The client reads ``FOXGLOVE_API_TOKEN``, and
    ``FOXGLOVE_API_URL`` overrides the default ``https://api.foxglove.dev``.
    """
    config = config or SampleConfig()
    return read_dataset(
        dataset_id,
        version=version,
        topics=config.topics,
        read_episode=partial(read_episode, config=config),
        include_properties=True,
        client_factory=make_client,
        **read_dataset_kwargs,
    )


@dataclass
class _Scalars:
    times_ns: np.ndarray  # int64 [N], ascending
    values: np.ndarray  # float32 [N, S]


def _read_scalars(episode: EpisodeReader) -> tuple[_Scalars, _Scalars]:
    """Read state and action. Both are small JSON topics, so they are held in memory."""
    times: dict[str, list[int]] = {TOPIC_OBSERVATION_STATE: [], TOPIC_ACTION_STATE: []}
    rows: dict[str, list[list[float]]] = {TOPIC_OBSERVATION_STATE: [], TOPIC_ACTION_STATE: []}
    labels: dict[str, list[str]] = {}
    topics = [TOPIC_OBSERVATION_STATE, TOPIC_ACTION_STATE]
    for _, channel, message, decoded in episode.iter_messages(topics=topics):
        scalars = decoded["scalars"]
        names = [str(item["label"]) for item in scalars]
        if labels.setdefault(channel.topic, names) != names:
            raise ValueError(f"Episode {episode.id} {channel.topic} label order changed.")
        times[channel.topic].append(int(message.log_time))
        rows[channel.topic].append([float(item["value"]) for item in scalars])
    for topic in topics:
        if not rows[topic]:
            raise ValueError(f"Episode {episode.id} has no {topic} messages.")
    if labels[TOPIC_OBSERVATION_STATE] != labels[TOPIC_ACTION_STATE]:
        raise ValueError(f"Episode {episode.id}: action labels differ from state labels.")

    def stream(topic: str) -> _Scalars:
        order = np.argsort(np.asarray(times[topic], dtype=np.int64), kind="stable")
        return _Scalars(
            times_ns=np.asarray(times[topic], dtype=np.int64)[order],
            values=np.asarray(rows[topic], dtype=np.float32)[order],
        )

    return stream(TOPIC_OBSERVATION_STATE), stream(TOPIC_ACTION_STATE)


def _has_gap(times_ns: np.ndarray, start_ns: int, end_ns: int, max_gap_ns: int) -> bool:
    """Whether consecutive samples between the hold sample at ``start_ns`` and ``end_ns`` gap."""
    first = max(int(np.searchsorted(times_ns, start_ns, side="right")) - 1, 0)
    last = int(np.searchsorted(times_ns, end_ns, side="left"))
    window = times_ns[first : last + 1]
    return bool(window.size > 1 and np.diff(window).max() > max_gap_ns)


def _task_of(properties: Mapping[str, object], default_task: str | None) -> str | None:
    value = properties.get("task")
    if isinstance(value, str) and value.strip():
        return value
    if default_task is not None and default_task.strip():
        return default_task
    return None


def read_episode(episode: EpisodeReader, config: SampleConfig) -> Iterator[Sample]:
    """Yield one aligned training sample per ``config.fps`` tick of ``episode``.

    Module-level so it pickles to DataLoader workers; bind ``config`` with
    ``functools.partial`` (see :func:`build_torch_dataset`).
    """
    properties = dict(episode.properties)
    verdict = outcome_of(properties, config.outcome_key, config.keep_outcomes)
    if verdict == "unlabeled" and config.keep_unlabeled:
        verdict = "kept"
    if verdict != "kept":
        logger.info("Skipping episode %s: %s.", episode.id, verdict)
        return
    task = _task_of(properties, config.default_task)
    if task is None:
        raise ValueError(f"Episode {episode.id} has no task property. Set default_task.")

    state, action = _read_scalars(episode)
    max_gap_ns = int(config.max_gap_s * _NS)
    image_key = f"observation.images.{config.camera}"
    frames = decode_h264(
        episode,
        topic=image_topic(config.camera),
        lookback=timedelta(seconds=config.lookback_s),
    )
    try:
        current = next(frames, None)
        if current is None:
            logger.warning(
                "Skipping episode %s: no frames on %s.", episode.id, image_topic(config.camera)
            )
            return
        t0 = max(int(state.times_ns[0]), int(action.times_ns[0]), current["log_time_ns"])
        t1 = min(int(state.times_ns[-1]), int(action.times_ns[-1]))
        if t1 < t0:
            logger.warning("Skipping episode %s: topics do not overlap in time.", episode.id)
            return
        if _has_gap(state.times_ns, t0, t1, max_gap_ns) or _has_gap(
            action.times_ns, t0, t1, max_gap_ns
        ):
            logger.warning(
                "Skipping episode %s: state or action gap over %.2fs.", episode.id, config.max_gap_s
            )
            return

        count = int((t1 - t0) * config.fps // _NS) + 1
        grid = t0 + np.rint(np.arange(count) * (_NS / config.fps)).astype(np.int64)
        grid_state = state.values[np.searchsorted(state.times_ns, grid, side="right") - 1]
        grid_action = action.values[np.searchsorted(action.times_ns, grid, side="right") - 1]

        next_tick = 0
        while current is not None:
            following = next(frames, None)
            if following is not None and following["log_time_ns"] - current["log_time_ns"] > (
                max_gap_ns
            ):
                logger.warning(
                    "Episode %s: camera gap over %.2fs at %d; ending it there.",
                    episode.id,
                    config.max_gap_s,
                    following["log_time_ns"],
                )
                return
            # The frame is held until the next one arrives. The last frame ends the
            # episode: samples may not be later than the camera's final frame.
            if following is None:
                stop = int(np.searchsorted(grid, current["log_time_ns"], side="right"))
            else:
                stop = int(np.searchsorted(grid, following["log_time_ns"], side="left"))
            if stop > next_tick:
                image = (
                    torch.from_numpy(current["image"]).permute(2, 0, 1).float().div_(255.0)
                ).contiguous()
                for tick in range(next_tick, stop):
                    yield _sample(
                        episode.id,
                        task,
                        tick,
                        config,
                        image_key,
                        image,
                        grid_state,
                        grid_action,
                    )
                next_tick = stop
            current = following
    finally:
        frames.close()


def _sample(
    episode_id: str,
    task: str,
    tick: int,
    config: SampleConfig,
    image_key: str,
    image: torch.Tensor,
    grid_state: np.ndarray,
    grid_action: np.ndarray,
) -> Sample:
    sample: Sample = {
        "observation.state": torch.from_numpy(grid_state[tick].copy()),
        image_key: image,
        "task": task,
        "episode_id": episode_id,
        "frame_index": tick,
        "timestamp": tick / config.fps,
    }
    if config.chunk_size == 1:
        sample["action"] = torch.from_numpy(grid_action[tick].copy())
    else:
        indices = np.arange(tick, tick + config.chunk_size)
        sample["action"] = torch.from_numpy(grid_action[np.minimum(indices, len(grid_action) - 1)])
        sample["action_is_pad"] = torch.from_numpy(indices >= len(grid_action))
    return sample


def main(argv: list[str] | None = None) -> None:
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(
        prog="python -m wawtech_workshop.training_data",
        description="Load a few batches from a Foxglove dataset version and print their shapes.",
    )
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    parser.add_argument("--camera", default="wrist")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--chunk-size", type=int, default=1, help="Action chunk length (ACT).")
    parser.add_argument("--task", default=None, help="Task for episodes with no task property.")
    parser.add_argument("--keep-outcomes", nargs="+", default=["success"])
    parser.add_argument("--keep-unlabeled", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    config = SampleConfig(
        camera=args.camera,
        chunk_size=args.chunk_size,
        default_task=args.task,
        keep_outcomes=tuple(args.keep_outcomes),
        keep_unlabeled=args.keep_unlabeled,
    )
    dataset = build_torch_dataset(args.dataset_id, args.version, config)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    seen = 0
    for batch in loader:
        seen += 1
        print(f"batch {seen}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:32s} {tuple(value.shape)} {value.dtype}")
            else:
                print(f"  {key:32s} {value[:2]}{' ...' if len(value) > 2 else ''}")
        if seen >= args.batches:
            break
    if seen == 0:
        print(
            "No samples. Every episode was filtered out: check the outcome property "
            "(--keep-outcomes, --keep-unlabeled) and the task property (--task)."
        )


if __name__ == "__main__":
    main()
