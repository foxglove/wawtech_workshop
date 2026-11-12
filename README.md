# wawtech_workshop

Workshop materials for **From Teleop to Policy**: record teleoperation demos on a Seeed reBot B601 arm (DM motors) and an Orbbec Gemini 305, curate episodes in a Foxglove dataset, convert them to a LeRobot v3 dataset, and train ACT.

Camera and arm behaviour is **unverified on hardware**. This slice runs offline: the Orbbec plugin is exercised with a fake backend, and the converter is exercised with synthetic MCAP files.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
```

`uv sync` installs this package and the `lerobot_camera_orbbec` workspace member. `pyorbbecsdk2==2.1.3` pins an older PyAV; this project overrides that pin to `av>=15,<16`, which LeRobot 0.7's dataset writer requires. The Orbbec SDK itself does not import PyAV.

Convert a directory of episode MCAP files without calling the Foxglove API:

```bash
uv run build-dataset --mcap-dir data/episodes --output outputs/dataset --repo-id local/rebot_b601
```

To download a dataset version instead, set `FOXGLOVE_API_TOKEN` (optional override: `FOXGLOVE_API_URL`) and pass `--dataset-id` and `--version`. The token is never written to logs. `--dry-run` prints the episode selection and the episodes that would be downloaded.

Each episode MCAP carries `/observation/state`, `/action/state`, and `/observation/images/<camera>` (`foxglove.CompressedVideo`, JSON, `format` `h264`). The recorder also writes an `episode` metadata record. Whether a Foxglove data-stream download preserves that record is unverified, and the converter does not depend on it. When the record is present, its task and episode index are used and its labels and cameras are checked against the file. Otherwise the task comes from the episode `task` property, then from `--task`. Pass `--task` when neither source has one.
