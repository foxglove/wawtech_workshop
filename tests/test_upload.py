"""``upload-episodes`` against a fake Foxglove client, and recorded files read back as a
Foxglove dataset by ``build-dataset``."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.fake_foxglove import FakeClient, FakeEpisode
from tests.test_recorder import _record
from wawtech_workshop import upload
from wawtech_workshop.dataset import cli as dataset_cli
from wawtech_workshop.dataset import foxglove_source


@dataclass
class FakeUploadClient:
    """Records calls. Imports complete after ``polls_until_complete`` recording listings."""

    polls_until_complete: int = 1
    fail_key: str | None = None
    uploads: list[dict[str, Any]] = field(default_factory=list)
    created: list[dict[str, Any]] = field(default_factory=list)
    staged: list[dict[str, Any]] = field(default_factory=list)
    commits: list[dict[str, Any]] = field(default_factory=list)
    polls: int = 0
    # Episode custom properties the org defines; None makes listing them fail.
    property_keys: list[str] | None = field(default_factory=lambda: ["task", "outcome"])
    created_properties: list[dict[str, Any]] = field(default_factory=list)

    def get_custom_properties(self, *, resource_type: str) -> list[dict[str, Any]]:
        assert resource_type == "episode"
        if self.property_keys is None:
            raise RuntimeError("403 Forbidden")
        return [{"key": key, "resource_type": "episode"} for key in self.property_keys]

    def create_custom_property(self, **kwargs: Any) -> dict[str, Any]:
        assert self.property_keys is not None
        self.created_properties.append(kwargs)
        self.property_keys.append(kwargs["key"])
        return {"id": f"prop-{kwargs['key']}", **kwargs}

    def get_dataset(self, *, dataset_id: str) -> dict[str, Any]:
        return {"id": dataset_id, "project_id": "proj-from-dataset"}

    def upload_data(self, **kwargs: Any) -> dict[str, Any]:
        data = kwargs.pop("data")
        kwargs["size"] = len(data.read())
        self.uploads.append(kwargs)
        return {"link": "https://upload.invalid", "request_id": "r", "text": "", "code": 200}

    def get_recordings(self, **kwargs: Any) -> list[dict[str, Any]]:
        self.polls += 1
        status = "complete" if self.polls >= self.polls_until_complete else "pending"
        rows = []
        for item in self.uploads:
            key = item["key"]
            rows.append(
                {
                    "id": f"rec-{key}",
                    "key": key,
                    "import_status": "failed" if key == self.fail_key else status,
                }
            )
        # Another device's recording must be ignored.
        rows.append({"id": "rec-other", "key": "someone-else", "import_status": "complete"})
        return rows

    def create_episodes(self, *, project_id: str, episodes: list[dict[str, Any]]) -> list[dict]:
        self.created.append({"project_id": project_id, "episodes": episodes})
        return [{"id": f"ep-{index}"} for index in range(len(episodes))]

    def update_dataset_episodes(self, *, dataset_id: str, add: list[str]) -> dict[str, Any]:
        self.staged.append({"dataset_id": dataset_id, "add": add})
        return {"added": len(add), "updated": 0, "removed": 0, "already_present": 0}

    def commit_dataset(self, *, dataset_id: str, **kwargs: Any) -> dict[str, Any]:
        self.commits.append({"dataset_id": dataset_id, **kwargs})
        return {"committed": {"version_number": 3}, "editable_version_number": 4}


def _recorded(tmp_path: Path, keys: dict[int, str] | None = None) -> Path:
    _record(tmp_path, num_episodes=2, episode_time_s=0.2, keys=keys, cameras=("wrist",))
    return tmp_path / "episodes"


def test_dry_run_reads_metadata_without_a_client(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episodes = _recorded(tmp_path, keys={2: "f"})
    assert upload.run([str(episodes), "--dataset-id", "ds", "--dry-run"], client=object()) == 0
    out = capsys.readouterr().out
    assert "episode 0, outcome failure, task 'pick up the butter'" in out
    assert "episode 1, outcome success" in out
    assert "Dry run: would upload 2 files" in out


def test_upload_stage_and_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    episodes = _recorded(tmp_path, keys={2: "f"})
    client = FakeUploadClient(polls_until_complete=2)
    monkeypatch.setattr(upload.time, "sleep", lambda _s: None)
    code = upload.run(
        [str(episodes), "--dataset-id", "ds-1", "--commit", "--device-name", "arm-1"],
        client=client,
    )
    assert code == 0
    assert [item["key"] for item in client.uploads] == [
        "episode_0000_test",
        "episode_0001_test",
    ]
    assert all(item["device_name"] == "arm-1" for item in client.uploads)
    assert all(item["project_id"] == "proj-from-dataset" for item in client.uploads)
    assert all(item["size"] > 0 for item in client.uploads)
    assert client.polls == 2
    assert client.created == [
        {
            "project_id": "proj-from-dataset",
            "episodes": [
                {
                    "recordings": ["rec-episode_0000_test"],
                    "properties": {"task": "pick up the butter", "outcome": "failure"},
                },
                {
                    "recordings": ["rec-episode_0001_test"],
                    "properties": {"task": "pick up the butter", "outcome": "success"},
                },
            ],
        }
    ]
    assert client.staged == [{"dataset_id": "ds-1", "add": ["ep-0", "ep-1"]}]
    assert client.commits == [{"dataset_id": "ds-1"}]


def test_outcome_filter_and_no_properties(tmp_path: Path) -> None:
    episodes = _recorded(tmp_path, keys={2: "f"})
    client = FakeUploadClient()
    upload.run(
        [str(episodes), "--dataset-id", "ds", "--outcomes", "success", "--no-properties"],
        client=client,
    )
    assert [item["key"] for item in client.uploads] == ["episode_0001_test"]
    assert client.created[0]["episodes"] == [{"recordings": ["rec-episode_0001_test"]}]
    assert client.commits == []


def test_failed_import_raises(tmp_path: Path) -> None:
    episodes = _recorded(tmp_path)
    client = FakeUploadClient(fail_key="episode_0001_test")
    with pytest.raises(RuntimeError, match="failed to import recording episode_0001_test"):
        upload.run([str(episodes), "--dataset-id", "ds"], client=client)
    assert client.created == []


def test_import_timeout(tmp_path: Path) -> None:
    clock = iter([0.0, 0.0, 1000.0])
    with pytest.raises(TimeoutError, match="not imported"):
        upload.wait_for_recordings(
            FakeUploadClient(polls_until_complete=99, uploads=[{"key": "a"}]),
            ["a"],
            device_name="d",
            project_id=None,
            timeout_s=10,
            sleep=lambda _s: None,
            monotonic=lambda: next(clock),
            say=lambda _m: None,
        )


def test_commit_needs_dataset(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        upload.run([str(tmp_path), "--commit"], client=object())


def test_recorded_episodes_through_a_foxglove_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dataset reader streams messages only (no metadata records). The properties the
    uploader sets are what build-dataset filters and labels on."""
    episodes_dir = _recorded(tmp_path, keys={2: "f"})
    local = upload.find_episodes([episodes_dir])
    fake = FakeClient(
        [
            FakeEpisode(
                f"ep-{episode.episode_index}",
                {"task": episode.task, "outcome": episode.outcome},
                mcap=episode.path,
            )
            for episode in local
        ]
    )
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "token")
    monkeypatch.setattr(foxglove_source, "make_client", lambda: fake)
    output = tmp_path / "dataset"
    dataset_cli.main(
        [
            "--dataset-id",
            "ds",
            "--version",
            "1",
            "--output",
            str(output),
            "--repo-id",
            "local/from-foxglove",
            "--project-dir",
            str(tmp_path / "downloads"),
        ]
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/from-foxglove", root=output, video_backend="pyav")
    # Only the success episode is kept by default.
    assert dataset.meta.total_episodes == 1
    assert [item[0] for item in fake.streamed] == ["ep-1"]
    assert dataset[0]["task"] == "pick up the butter"


def test_mcap_dir_conversion_skips_failed_recordings(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episodes_dir = _recorded(tmp_path, keys={2: "f"})
    output = tmp_path / "dataset"
    dataset_cli.main(
        ["--mcap-dir", str(episodes_dir), "--output", str(output), "--repo-id", "local/kept"]
    )
    assert "skipping episode_0000_test.mcap: outcome 'failure'" in capsys.readouterr().out
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/kept", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 1


def test_org_without_properties_uploads_successes_unlabeled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    episodes = _recorded(tmp_path, keys={2: "f"})
    client = FakeUploadClient(property_keys=[])
    assert upload.run([str(episodes), "--dataset-id", "ds"], client=client) == 0
    out = capsys.readouterr().out
    assert "no episode properties ['task', 'outcome']" in out
    assert "--keep-unlabeled" in out
    assert "pass --task to build-dataset" in out
    # The failure is not uploaded, since nothing in the org would mark it.
    assert [item["key"] for item in client.uploads] == ["episode_0001_test"]
    assert client.created[0]["episodes"] == [{"recordings": ["rec-episode_0001_test"]}]
    assert client.created_properties == []


def test_org_with_task_only_keeps_task(tmp_path: Path) -> None:
    episodes = _recorded(tmp_path)
    client = FakeUploadClient(property_keys=["task"])
    upload.run([str(episodes), "--dataset-id", "ds", "--outcomes", "success"], client=client)
    assert client.created[0]["episodes"][0] == {
        "recordings": ["rec-episode_0000_test"],
        "properties": {"task": "pick up the butter"},
    }


def test_unlisted_properties_degrade_to_none(tmp_path: Path) -> None:
    episodes = _recorded(tmp_path)
    client = FakeUploadClient(property_keys=None)
    upload.run([str(episodes), "--dataset-id", "ds"], client=client)
    assert all("properties" not in episode for episode in client.created[0]["episodes"])


def test_create_properties(tmp_path: Path) -> None:
    episodes = _recorded(tmp_path, keys={2: "f"})
    client = FakeUploadClient(property_keys=["task"])
    upload.run([str(episodes), "--dataset-id", "ds", "--create-properties"], client=client)
    assert client.created_properties == [
        {
            "key": "outcome",
            "resource_type": "episode",
            "label": "Outcome",
            "value_type": "enum",
            "enum_values": ["success", "failure"],
        }
    ]
    assert [episode["properties"]["outcome"] for episode in client.created[0]["episodes"]] == [
        "failure",
        "success",
    ]


def test_unlabeled_dataset_episodes_need_keep_unlabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    episodes_dir = _recorded(tmp_path)
    local = upload.find_episodes([episodes_dir])
    fake = FakeClient(
        [FakeEpisode(f"ep-{episode.episode_index}", {}, mcap=episode.path) for episode in local]
    )
    monkeypatch.setenv("FOXGLOVE_API_TOKEN", "token")
    monkeypatch.setattr(foxglove_source, "make_client", lambda: fake)
    base = ["--dataset-id", "ds", "--version", "1", "--project-dir", str(tmp_path / "dl")]
    with pytest.raises(ValueError, match="No episodes left"):
        dataset_cli._run([*base, "--output", str(tmp_path / "none"), "--repo-id", "local/none"])
    output = tmp_path / "dataset"
    dataset_cli.main(
        [
            *base,
            "--keep-unlabeled",
            "--task",
            "pick up the butter",
            "--output",
            str(output),
            "--repo-id",
            "local/unlabeled",
        ]
    )
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset("local/unlabeled", root=output, video_backend="pyav")
    assert dataset.meta.total_episodes == 2
    assert dataset[0]["task"] == "pick up the butter"
