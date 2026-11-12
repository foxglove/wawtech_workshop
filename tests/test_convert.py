"""End-to-end conversion of synthetic episode MCAPs."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path

import numpy as np
import pytest

from tests.synthetic import write_synthetic_episode
from wawtech_workshop.dataset.convert import EpisodeInput, convert_episodes


def _grid(times_ns: list[int], fps: int) -> list[int]:
    t0 = times_ns[0]
    t1 = times_ns[-1]
    grid: list[int] = []
    index = 0
    while True:
        stamp = t0 + round(index * 1_000_000_000 / fps)
        if stamp > t1:
            return grid
        grid.append(stamp)
        index += 1


def _hold(values: np.ndarray, times_ns: list[int], stamp: int) -> np.ndarray:
    index = bisect_right(times_ns, stamp) - 1
    return values[index]


def test_convert_three_episodes_round_trips(tmp_path: Path) -> None:
    episodes = [
        write_synthetic_episode(
            tmp_path / "mcap" / "episode_00.mcap",
            n_frames=6,
            episode_index=0,
            task="reach",
            t0_ns=1_700_000_000_000_000_000,
        ),
        write_synthetic_episode(
            tmp_path / "mcap" / "episode_01.mcap",
            n_frames=6,
            episode_index=1,
            task="grasp",
            t0_ns=1_700_000_100_000_000_000,
        ),
        write_synthetic_episode(
            tmp_path / "mcap" / "episode_02.mcap",
            n_frames=3,
            episode_index=2,
            task="lift",
            t0_ns=1_700_000_200_000_000_000,
        ),
    ]
    output = tmp_path / "dataset"
    root = convert_episodes(
        [episode.path for episode in episodes],
        output,
        "local/rebot_b601",
        fps=30,
        robot_type="rebot_b601_follower",
    )
    assert root == output

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/rebot_b601", root=output, video_backend="pyav")
    expected_lengths = [_grid(episode.times_ns, 30) for episode in episodes]
    total = sum(len(grid) for grid in expected_lengths)
    assert len(dataset) == total
    assert dataset.meta.total_episodes == 3
    state_feature = dataset.features["observation.state"]
    action_feature = dataset.features["action"]
    image_feature = dataset.features["observation.images.front"]
    assert tuple(state_feature["shape"]) == (2,)
    assert state_feature["dtype"] == "float32"
    assert state_feature["names"] == ["joint1.pos", "joint2.pos"]
    assert tuple(action_feature["shape"]) == (2,)
    assert action_feature["names"] == ["joint1.pos", "joint2.pos"]
    assert image_feature["dtype"] == "video"
    assert tuple(image_feature["shape"]) == (episodes[0].height, episodes[0].width, 3)
    assert image_feature["names"] == ["height", "width", "channels"]

    cursor = 0
    for episode, grid in zip(episodes, expected_lengths, strict=True):
        for offset, stamp in enumerate(grid):
            item = dataset[cursor + offset]
            assert item["task"] == episode.metadata.task
            np.testing.assert_allclose(
                np.asarray(item["observation.state"], dtype=np.float32),
                _hold(episode.state, episode.times_ns, stamp),
                atol=1e-5,
            )
            np.testing.assert_allclose(
                np.asarray(item["action"], dtype=np.float32),
                _hold(episode.action, episode.times_ns, stamp),
                atol=1e-5,
            )
        cursor += len(grid)

    image = np.asarray(dataset[0]["observation.images.front"])
    assert image.shape in {
        (3, episodes[0].height, episodes[0].width),
        (episodes[0].height, episodes[0].width, 3),
    }
    assert min(expected_lengths, key=len) is expected_lengths[2]
    assert len(expected_lengths[2]) < len(expected_lengths[0])


def test_gap_too_large_names_episode_and_topic(tmp_path: Path) -> None:
    episode = write_synthetic_episode(
        tmp_path / "gap.mcap",
        n_frames=6,
        episode_index=4,
        gap_after=1,
        gap_ns=500_000_000,
    )
    with pytest.raises(ValueError, match=r"Episode 4 .*topic /observation/state"):
        convert_episodes([episode.path], tmp_path / "out", "local/gap", fps=30)


def test_label_order_mismatch_raises(tmp_path: Path) -> None:
    first = write_synthetic_episode(tmp_path / "a.mcap", n_frames=3, episode_index=0, task="a")
    second = write_synthetic_episode(
        tmp_path / "b.mcap",
        n_frames=3,
        episode_index=1,
        task="b",
        state_names=("joint2.pos", "joint1.pos"),
    )
    del first
    with pytest.raises(ValueError, match="state_names"):
        convert_episodes(
            [tmp_path / "a.mcap", second.path],
            tmp_path / "out",
            "local/labels",
            fps=30,
        )


def test_metadata_optional_uses_input_task_and_default(tmp_path: Path) -> None:
    first = write_synthetic_episode(
        tmp_path / "mcap" / "a.mcap",
        n_frames=4,
        episode_index=9,
        task="recorded-but-stripped",
        include_metadata=False,
        t0_ns=1_700_000_000_000_000_000,
    )
    second = write_synthetic_episode(
        tmp_path / "mcap" / "b.mcap",
        n_frames=4,
        task="also-stripped",
        include_metadata=False,
        t0_ns=1_700_000_100_000_000_000,
    )
    output = tmp_path / "dataset"
    convert_episodes(
        [EpisodeInput(first.path, task="from-input"), second.path],
        output,
        "local/fallback-task",
        fps=30,
        default_task="from-default",
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/fallback-task", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 2
    first_len = len(_grid(first.times_ns, 30))
    assert dataset[0]["task"] == "from-input"
    assert dataset[first_len]["task"] == "from-default"


def test_missing_task_names_the_flag(tmp_path: Path) -> None:
    episode = write_synthetic_episode(tmp_path / "bare.mcap", n_frames=3, include_metadata=False)
    with pytest.raises(ValueError, match=r"--task"):
        convert_episodes([episode.path], tmp_path / "out", "local/no-task", fps=30)


def test_metadata_camera_names_must_match_topics(tmp_path: Path) -> None:
    episode = write_synthetic_episode(
        tmp_path / "cameras.mcap",
        n_frames=3,
        camera_names=("front",),
        metadata_camera_names=("wrist",),
    )
    with pytest.raises(ValueError, match="camera_names"):
        convert_episodes([episode.path], tmp_path / "out", "local/cameras", fps=30)


def test_write_retains_only_a_few_frames(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import weakref

    from wawtech_workshop.dataset import convert as convert_mod

    class Tracker:
        def __init__(self) -> None:
            self.live = 0
            self.peak = 0

        def watch(self, image: np.ndarray) -> None:
            self.live += 1
            self.peak = max(self.peak, self.live)
            weakref.finalize(image, self.release)

        def release(self) -> None:
            self.live -= 1

    tracker = Tracker()
    real = convert_mod.iter_decode_h264_messages

    def wrapped(messages):
        for stamp, image in real(messages):
            tracker.watch(image)
            yield stamp, image

    monkeypatch.setattr(convert_mod, "iter_decode_h264_messages", wrapped)
    episode = write_synthetic_episode(tmp_path / "frames.mcap", n_frames=12)
    convert_episodes([episode.path], tmp_path / "out", "local/stream", fps=30)
    assert tracker.peak <= 6
    assert tracker.peak < 12
