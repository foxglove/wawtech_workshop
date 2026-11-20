# wawtech_workshop

Workshop materials for **From Teleop to Policy**: record teleoperation demos on a Seeed reBot B601 arm (DM motors) with a reBot Arm 102 leader and a wrist UVC camera (an Orbbec Gemini 305 front camera is optional), curate episodes in a Foxglove dataset, convert them to a LeRobot v3 dataset, and train ACT.

Camera and arm behaviour is **unverified on hardware**. Everything here runs offline in tests: the Orbbec plugin with a fake backend, the recorder with a fake follower and leader, the uploader with a fake Foxglove client, and the converter with recorded and synthetic MCAP files.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run ruff check .
```

`uv sync` installs this package and the `lerobot_camera_orbbec` workspace member. `pyorbbecsdk2==2.1.3` pins an older PyAV; this project overrides that pin to `av>=15,<16`, which LeRobot 0.7's dataset writer requires. The Orbbec SDK itself does not import PyAV.

## Record episodes

`record-episode` drives the follower from the leader and writes one MCAP per episode with the Foxglove SDK. It takes the same `--robot.*` and `--teleop.*` flags as `lerobot-record`:

```bash
uv run record-episode \
  --robot.type=rebot_b601_follower --robot.port=/dev/ttyACM0 --robot.id=follower \
  --teleop.type=rebot_102_leader --teleop.port=/dev/ttyUSB0 --teleop.id=leader \
  --task="Pick up the butter and put it on the plate" \
  --num_episodes=10 --episode_time_s=45 --reset_time_s=10 \
  --output_dir=data/episodes
```

The leader, `rebot_102_leader`, is the reBot Arm 102 (Seeed's StarArm102): seven FashionStar UART servos read through `motorbridge-smart-servo`, which `lerobot[rebot]` installs. Its joint names and ranges match the B601 follower key for key. On its first connect it asks you to put the arm in its zero pose with the gripper closed, then saves that calibration under `--teleop.id`.

Cameras come from `--cameras` unless `--robot.cameras` is given:

- `wrist` (default): a UVC camera through OpenCV. Pick it with `--wrist_camera=0` (an index) or `--wrist_camera=/dev/video0`.
- `front`: the Orbbec, through the `orbbec` plugin in `packages/`. Record both with `--cameras=wrist,front`; `--front_camera=<serial>` picks one Orbbec if several are attached.

Both run at `--fps` and `--camera_width` x `--camera_height` (default 640x480). The camera names become the image topics and dataset features (`observation.images.wrist`), so switching to `wrist,front` later starts a dataset with a different shape: record the episodes for one training run with the same set. `--robot.cameras` overrides the presets entirely, as in `lerobot-record`.

While it runs: `n` (or Right) ends the episode as a success, `f` as a failure, `r` (or Left) throws it away and records it again, `q` (or Esc) stops. An episode that runs out of time is saved with `--timeout_outcome` (default `success`). Between episodes the arm stays in teleop for `--reset_time_s` without recording; `n` skips the rest of the reset.

While recording, the same data streams live to Foxglove: open `ws://127.0.0.1:8765` in the app to watch the arm and cameras, including between episodes, which is the teleop view. Use `--stream_port` if 8765 is taken (the recorder stops before connecting the arms if it is), `--stream_host=0.0.0.0` to watch from another machine on the network (not with `--control=foxglove` or `both`), and `--stream=false` to record files only.

### Start and finish from Foxglove

With `--control=foxglove` (or `both`, which keeps the keys too) the recorder waits in teleop, not recording, before every episode, and the Foxglove app starts and ends it:

```bash
uv run record-episode ... --control=foxglove --episode_time_s=0 --reset_time_s=0
```

Connect to `ws://127.0.0.1:8765`, then import `docs/foxglove-layout.json` (Layouts menu, Import from file). It has a state light (READY, RECORDING, RESET), the `/recorder/status` message, camera images, state-vs-action plots and buttons that call these services:

| Service | Request | Effect |
|---|---|---|
| `/recorder/start` | `{"task": "..."}` (task optional, and sticks for later episodes) | Begin the next episode. Only while ready. |
| `/recorder/finish` | `{"outcome": "success" \| "failure"}` | Save the episode. |
| `/recorder/discard` | `{}` | Throw it away and go back to ready. |
| `/recorder/quit` | `{}` | Stop the session. A running episode is saved as a success. |

A call that does not fit the state (finish while ready) is refused with `{"ok": false, "error": ...}` and changes nothing. The services only start and stop recording; they never move the arm. They have no authentication, so the recorder refuses them unless `--stream_host` is a loopback address. `--episode_time_s=0` makes an episode last until you press Success or Failure; otherwise it ends at the time limit with `--timeout_outcome`. `/recorder/status` is not written to the episode files. Regenerate the layout with `uv run python scripts/make_layout.py`.

### See the arm move in Foxglove

The recorder logs the follower's joint angles on `/joint_states` (`foxglove.JointStates`, radians, under the URDF's joint names `joint1` to `joint6`) and a fixed `world` to `base_link` frame on `/tf`, every tick. It also serves the arm model from `assets/urdf/` to the app, so there is nothing to download. The layout's **Arm** panel is a 3D panel whose URDF layer is in "Joint states" control mode, reading `/joint_states` from the recorder. The layer reads the URDF from disk (File path mode, desktop app only), so `docs/foxglove-layout.json` holds the absolute path of this checkout: regenerate it with `uv run python scripts/make_layout.py` after moving the repo or on another machine. `--urdf-source=url` makes it fetch the URDF from the running recorder instead (URL `package://rebot-devarm_description_fixend/urdf/reBot-DevArm_fixend.urdf`). To add the arm to your own layout: add a URDF layer in a 3D panel, set the source, Control mode to "Joint states" and the topic to `/joint_states`.

The leader's action goes on `/action/joint_states` the same way, and the layout draws it as a second, 35% opaque blue **ghost** arm over the follower: where the leader is, against where the follower has got to. Toggle or restyle it from the layer's settings in the 3D panel. The gripper has no joint in the model, so it is not drawn. The topics are also written into every episode file, so a replayed or uploaded episode shows the arm too. The model is Seeed Studio's [reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm) description (CERN-OHL-W 2.0), about 24 MB of meshes; see `assets/urdf/README.md`.

Each tick logs the follower's joint positions on `/observation/state`, the leader's action on `/action/state` (before the follower's `max_relative_target` clipping, as `lerobot-record` does), and every RGB camera on `/observation/images/<name>` as H.264. Files are named `episode_NNNN_<session>.mcap`; the index continues from the files already in `--output_dir`. Each file carries the `episode` metadata record the converter checks and a `recording` record with the outcome. Depth streams are not recorded.

`--fps` (default 30) must match `build-dataset --fps`. At 640x480, each camera adds about 1.5 MB/s.

## Put episodes in a Foxglove dataset

```bash
export FOXGLOVE_API_TOKEN=...
uv run upload-episodes data/episodes --dry-run
uv run upload-episodes data/episodes --dataset-id <dataset id>
```

This uploads each file (the file name is the recording key, so re-uploading is a no-op), waits for the imports, creates one Foxglove episode per recording with `task` and `outcome` episode properties, and stages them in the dataset's editable version. Review them in Foxglove and commit the version there, or pass `--commit`. Without `--dataset-id` it only uploads.

The properties are set only if the org defines them as episode custom properties. Before staging, the uploader lists them and leaves out any that are missing, with a warning, so an org that is not set up yet still works:

- Without `outcome`, only success episodes are uploaded (unless `--outcomes` says otherwise), and `build-dataset --dataset-id` needs `--keep-unlabeled` to read them.
- Without `task`, `build-dataset` needs `--task`.

To set the org up later, run once with `--create-properties`: it creates `task` (string) and `outcome` (enum: `success`, `failure`). `--no-properties` skips properties altogether.

Recording and `build-dataset --mcap-dir` never touch the Foxglove org, so the whole path from teleop to a LeRobot dataset works before the org exists.

## Build the LeRobot dataset

Convert a directory of episode MCAP files without calling the Foxglove API:

```bash
uv run build-dataset --mcap-dir data/episodes --output outputs/dataset --repo-id local/rebot_b601
```

In this mode, files from `record-episode` whose outcome is not in `--keep-outcomes` (default `success`) are skipped. Files without a recorded outcome are kept.

To read a dataset version instead, set `FOXGLOVE_API_TOKEN` (optional override: `FOXGLOVE_API_URL`) and pass `--dataset-id` and `--version`. Episodes are read with the Foxglove SDK's dataset reader (`foxglove.datasets`, from `foxglove-sdk[torch]`): episodes whose `outcome` property is in `--keep-outcomes` (default `success`) are streamed to local MCAP files, then converted. Episodes with no `outcome` property are left out unless `--keep-unlabeled` is set. The SDK reads each episode at the version the dataset version pins, and refuses a dataset version with missing recordings. By default the state and action topics and `/observation/images/wrist` are read; use `--cameras` or `--topics` to change that. The token is never written to logs. `--dry-run` prints the episode selection without downloading anything.

Training options (ACT on one GPU, or a π₀.₅ fine-tune on Ray/Anyscale) are outlined in [docs/training.md](docs/training.md).

Each episode MCAP carries `/observation/state`, `/action/state`, and `/observation/images/<camera>` (`foxglove.CompressedVideo`, JSON, `format` `h264`). The recorder also writes an `episode` metadata record. The SDK streams messages only, so episodes read from a dataset version do not carry that record, and the converter does not depend on it. When the record is present, its task and episode index are used and its labels and cameras are checked against the file. Otherwise the task comes from the episode `task` property, then from `--task`. Pass `--task` when neither source has one.
