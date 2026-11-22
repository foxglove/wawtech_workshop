from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from foxglove.datasets.torch import read_dataset

from tests.fake_foxglove import FakeClient, FakeEpisode
from tests.synthetic import write_synthetic_episode
from wawtech_workshop.training_data import SampleConfig, build_torch_dataset, read_episode

T0_NS = 1_700_000_000_000_000_000
START = datetime.fromtimestamp(T0_NS / 1e9, UTC) - timedelta(seconds=1)


class JsonFakeClient(FakeClient):
    """The real client decodes JSON messages; the shared fake hands back raw bytes."""

    def iter_messages(self, **kwargs: Any):
        for schema, channel, message, _ in super().iter_messages(**kwargs):
            yield schema, channel, message, json.loads(message.data)


def _episode(tmp_path: Path, name: str, properties: dict[str, object], **kwargs: Any):
    synthetic = write_synthetic_episode(
        tmp_path / f"{name}.mcap",
        camera_names=("wrist",),
        include_metadata=False,
        t0_ns=T0_NS,
        **kwargs,
    )
    return synthetic, FakeEpisode(name, properties, mcap=synthetic.path, start=START)


def _samples(client: FakeClient, config: SampleConfig) -> list[dict[str, Any]]:
    dataset = read_dataset(
        "ds",
        version=1,
        topics=config.topics,
        read_episode=partial(read_episode, config=config),
        include_properties=True,
        client_factory=lambda: client,
    )
    return list(dataset)


def test_samples_are_aligned_on_the_fps_grid(tmp_path: Path) -> None:
    synthetic, episode = _episode(
        tmp_path, "a", {"outcome": "success", "task": "pass the butter"}, n_frames=12
    )
    samples = _samples(JsonFakeClient([episode]), SampleConfig())

    assert samples
    first = samples[0]
    assert set(first) == {
        "observation.state",
        "action",
        "observation.images.wrist",
        "task",
        "episode_id",
        "frame_index",
        "timestamp",
    }
    assert first["task"] == "pass the butter"
    assert first["episode_id"] == "a"
    assert first["observation.state"].shape == (2,)
    assert first["action"].shape == (2,)
    assert first["observation.images.wrist"].shape == (3, synthetic.height, synthetic.width)
    assert first["observation.images.wrist"].dtype == torch.float32
    assert (
        0.0
        <= first["observation.images.wrist"].min()
        <= first["observation.images.wrist"].max()
        <= 1.0
    )

    # Zero-order hold: each sample carries the latest recorded value at or before its tick.
    times = np.asarray(synthetic.times_ns)
    t0 = times[0]
    for sample in samples:
        tick_ns = t0 + round(sample["frame_index"] * 1e9 / 30)
        index = np.searchsorted(times, tick_ns, side="right") - 1
        np.testing.assert_allclose(
            sample["observation.state"].numpy(), synthetic.state[index], atol=1e-6
        )
        np.testing.assert_allclose(sample["action"].numpy(), synthetic.action[index], atol=1e-6)
    assert [s["frame_index"] for s in samples] == list(range(len(samples)))
    assert len(samples) <= 12


def test_action_chunks_are_padded_at_the_end(tmp_path: Path) -> None:
    _, episode = _episode(tmp_path, "a", {"outcome": "success", "task": "t"}, n_frames=10)
    samples = _samples(JsonFakeClient([episode]), SampleConfig(chunk_size=4))

    assert samples[0]["action"].shape == (4, 2)
    assert samples[0]["action_is_pad"].tolist() == [False] * 4
    last = samples[-1]
    assert last["action_is_pad"].tolist() == [False, True, True, True]
    np.testing.assert_array_equal(last["action"][1].numpy(), last["action"][0].numpy())
    np.testing.assert_array_equal(samples[1]["action"][0].numpy(), samples[0]["action"][1].numpy())


def test_outcome_filter_and_default_task(tmp_path: Path) -> None:
    _, good = _episode(tmp_path, "good", {"outcome": "success"}, n_frames=6)
    _, bad = _episode(tmp_path, "bad", {"outcome": "failure", "task": "t"}, n_frames=6)
    _, unlabeled = _episode(tmp_path, "unlabeled", {"task": "t"}, n_frames=6)
    client = JsonFakeClient([good, bad, unlabeled])

    # The SDK re-raises callback errors as RuntimeError, keeping ours as the cause.
    with pytest.raises(RuntimeError, match="Failed to read episode good") as raised:
        _samples(client, SampleConfig())
    assert "no task" in str(raised.value.__cause__)

    samples = _samples(client, SampleConfig(default_task="pass the butter"))
    assert {s["episode_id"] for s in samples} == {"good"}
    assert {s["task"] for s in samples} == {"pass the butter"}

    both = _samples(client, SampleConfig(default_task="x", keep_unlabeled=True))
    assert {s["episode_id"] for s in both} == {"good", "unlabeled"}


def test_episode_with_a_gap_is_skipped(tmp_path: Path) -> None:
    _, gappy = _episode(
        tmp_path, "gappy", {"outcome": "success", "task": "t"}, n_frames=10, gap_after=4
    )
    _, fine = _episode(tmp_path, "fine", {"outcome": "success", "task": "t"}, n_frames=6)
    samples = _samples(JsonFakeClient([gappy, fine]), SampleConfig())
    assert {s["episode_id"] for s in samples} == {"fine"}


def test_dataset_batches_through_a_dataloader(tmp_path: Path, monkeypatch) -> None:
    from torch.utils.data import DataLoader

    _, episode = _episode(tmp_path, "a", {"outcome": "success", "task": "t"}, n_frames=12)
    monkeypatch.setattr(
        "wawtech_workshop.training_data.make_client", lambda: JsonFakeClient([episode])
    )
    dataset = build_torch_dataset("ds", 1, SampleConfig())
    batch = next(iter(DataLoader(dataset, batch_size=4, num_workers=0)))
    assert batch["observation.state"].shape == (4, 2)
    assert batch["observation.images.wrist"].shape[:2] == (4, 3)
    assert batch["task"] == ["t"] * 4
