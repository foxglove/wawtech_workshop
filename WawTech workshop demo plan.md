# WawTech workshop demo plan

Nov 18, 2026 · @Mat Sadowski

## Summary

The demo is reachable by Friday Nov 27 if the follower moves under teleop by Thursday Nov 19 and five pilot episodes go through the whole pipeline on Friday Nov 20. After that it is mostly volume: recording, training and rehearsing.

The demo, per MKT-1685: teleop butter pick-up episodes on the reBot B601-DM with the reBot Arm 102 leader, review and curate them in Foxglove, convert to LeRobot, train ACT, run it live with a recorded backup, and fix one failure mode through the data.

The biggest software gap is the recorder. PR #1 converts episode MCAPs into a LeRobot dataset, but nothing in the repo writes those MCAPs yet, and LeRobot 0.7 only streams to Foxglove live without saving MCAP. Step 1 of Part 2 covers it, with a fallback.

| Day | Goal | Who |
| --- | --- | --- |
| Wed 18 to Thu 19 | Arm bring-up and tests (Part 1) | Mat |
| Thu 19 | Recorder and read-only arm check script; cameras mounted | Agent builds, Mat tests |
| Fri 20 | Pilot: 5 episodes through record, Foxglove, dataset and a 2,000-step training run | Both |
| Sat 21 to Sun 22 | Record about 60 episodes | Mat |
| Mon 23 | Curate in Foxglove, build dataset v1, train run 1 overnight | Mat curates, agent runs |
| Tue 24 | Evaluate v1, record about 20 fix-up episodes, dataset v2, train run 2 overnight | Both |
| Wed 25 | Evaluate v2, record the backup video, gather slide assets, pack | Mat |
| Thu 26 | Venue setup and full dry run | Mat |
| Fri 27 | Workshop | Mat |

**First three things to do**

1. Check you have the reBot Arm 102 leader, a 24 V 15 A supply, the USB-CAN board and clamps, then run Part 1 steps 1 and 2 (mechanical, wiring, first power-up).
2. Scan the follower's motors and confirm all seven answer at the expected IDs (Part 1 step 3).
3. Say whether an agent should build the recorder now, on a new branch, so it is ready by the time the arm moves.

## Part 1: Arm bring-up and test routine

Work through these in order; each step assumes the one before passed. Budget about a day. Commands use LeRobot 0.7.0 as the repo pins it, so the types are `rebot_b601_follower` and `rebot_102_leader`. Seeed's wiki uses Seeed's own LeRobot fork with other names (`seeed_b601_dm_follower`, `rebot_arm_102_leader`), so don't mix commands from the two.

**Safety, the whole time**

- Clamp both arms to the table and keep about 1 m clear around the follower.
- Switch the 24 V supply off before plugging or unplugging any XT30 or JST cable. Never hot-plug motors.
- There is no hardware e-stop. Put the 24 V supply on a switched power strip within reach of your free hand and treat that switch as the e-stop.
- Cutting power or torque drops the arm under gravity. Keep a hand under the forearm when testing stops, and do early tests near the rest pose.
- Keep fingers out of the gripper and elbow pinch points while torque is on.

### 0. Software setup (30 min)

- Use the computer you will demo on. Seeed supports Ubuntu 22.04 and 24.04 best; macOS works with the serial bridge (the port shows as `/dev/tty.usbmodem…`) but is less tested.
- In the repo run `uv sync`. It installs LeRobot 0.7.0 with the `rebot` extra, which brings motorbridge (follower) and motorbridge-smart-servo (leader).
- Linux only: `sudo apt remove brltty`, and after plugging in, `sudo chmod 666 /dev/ttyACM* /dev/ttyUSB*`.

### 1. Mechanical and wiring check, power off (20 min)

- Confirm the follower is the DM model; the B601-RS looks almost the same.
- Check screws, and that the harness between motors 1 and 2 is plugged in (Seeed calls it easy to miss).
- Move each joint by hand through its full range and watch for cables that snag or pull.
- Follower chain: base motor, XT30 2+2 cable, signal/power separation board, 2-pin JST, USB-CAN board, USB-C to the computer.
- Supply: 24 V, about 15 A (Seeed lists a 24 V 15 A XT30 supply or a MeanWell 24 V 14.6 A). Set its mains switch to 230 V. Measure 24 V at the XT30 with a multimeter before connecting it. The leader takes 12 V 2 A.

### 2. First power-up (5 min)

- Rest the follower in its zero pose (the folded sit-down pose, gripper closed) and switch on. LEDs light and nothing moves.
- Plug in USB-C only now. Run `uv run lerobot-find-port` and confirm the follower port (the bridge) is `/dev/tty.usbmodem00000000050C1` and the leader port is `/dev/tty.usbserial-2140` (on Linux they would be `/dev/ttyACM0` and `/dev/ttyUSB0`).

### 3. Motor discovery and IDs (30 min)

Start the gateway, then open MotorBridge Studio in a browser and click Scan Damiao:

```bash
uv run motorbridge-gateway -- --bind 127.0.0.1:9002 --vendor damiao --transport dm-serial --serial-port /dev/tty.usbmodem00000000050C1 --serial-baud 921600 --dt-ms 20
```

Expect seven motors. LeRobot uses exactly these IDs, so they have to match:

| Joint | Motor | CAN ID | Master ID |
| --- | --- | --- | --- |
| 1 shoulder\_pan | J4340P | 0x01 | 0x11 |
| 2 shoulder\_lift | J4340P | 0x02 | 0x12 |
| 3 elbow\_flex | J4340P | 0x03 | 0x13 |
| 4 wrist\_flex | J4310 | 0x04 | 0x14 |
| 5 wrist\_yaw | J4310 | 0x05 | 0x15 |
| 6 wrist\_roll | J4310 | 0x06 | 0x16 |
| 7 gripper | J4310 | 0x07 | 0x17 |

- A motor missing: power off and reseat the harness to it.
- Wrong or duplicate IDs: fix one motor at a time in DM-Tools at 921600 baud, with only that motor on the 3-pin line. Read, set CAN ID and Master ID, write.
- In DM-Tools never press the parameter calibration button (it can corrupt factory inertia parameters), and avoid the read/set buttons beside the CAN ID field (they can give every motor on the bus the same ID).
- If all seven answer, there is no firmware to flash.

### 4. Per-joint test in MotorBridge Studio (45 min)

One motor at a time, with the arm supported:

- Enable (steady green LED), move by 0.2 rad, move back to 0, disable.
- It should move smoothly, hold position without buzzing, and report no faults. Note which way positive goes on each joint.
- Gripper (motor 7): open and close slowly.
- Keep torque small in any MIT torque test, never set kd to 0 for position moves, and disable when done.
- Touch each motor afterwards: warm is fine, hot is not.
- Stop the gateway before using LeRobot; only one program can hold the serial port.

### 5. Calibrate the follower (10 min)

Put the arm in the zero pose (sit-down position, gripper fully closed; Seeed's LeRobot page has a photo), hold it still and press Enter when asked:

```bash
uv run lerobot-calibrate --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 --robot.id=follower --robot.can_adapter=damiao
```

- Then test that the zero survives a power cycle: switch off, switch on, put the arm in the zero pose and read the joint positions (step 7). All should be within a degree or two of 0. If not, re-zero at the start of every session and add that to the demo checklist.
- LeRobot clips commands to soft limits: shoulder\_pan ±150°, shoulder\_lift and elbow\_flex −200° to 1°, wrist\_flex −80° to 90°, wrist\_yaw and wrist\_roll ±90°, gripper −270° to 0°.

### 6. Calibrate the leader (10 min)

Same zero pose on the leader, gripper closed:

```bash
uv run lerobot-calibrate --teleop.type=rebot_102_leader --teleop.port=/dev/tty.usbserial-2140 --teleop.id=leader
```

To redo either calibration, type `c` at the prompt, or delete the file under `~/.cache/huggingface/lerobot/calibration/`.

### 7. Direction check with follower torque off (15 min)

- Move one leader joint by about +20° and the same follower joint by hand. Both should report the same sign and roughly the same angle. Repeat for all seven.
- A reversed joint gets fixed in `--teleop.joint_directions`. Defaults: `{"shoulder_pan":-1,"shoulder_lift":-1,"elbow_flex":1,"wrist_flex":1,"wrist_yaw":1,"wrist_roll":-1,"gripper":-6}` (the gripper's −6 also scales its travel).
- Stock LeRobot has no torque-off readout, so this step needs a small read-only script. An agent can write it (Part 2, step 1).

### 8. First teleop, slow (30 min)

Both arms in the zero pose, free hand on the power switch. Then connect the Foxglove app to `ws://localhost:8765` and plot `/observation/state` against `/action/state`.

```bash
uv run lerobot-teleoperate \
  --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 --robot.id=follower --robot.can_adapter=damiao \
  --robot.control_mode=pos_vel --robot.pos_vel_velocity=30 --robot.max_relative_target=5 \
  --teleop.type=rebot_102_leader --teleop.port=/dev/tty.usbserial-2140 --teleop.id=leader \
  --display_data=true --display_mode=foxglove
```

- `pos_vel` caps joint speed at 30 °/s and `max_relative_target` limits each command to a 5° step, so a mistake is slow.
- Move one joint at a time, then combine, then cover the whole workspace.
- Once it behaves, drop the three speed flags (back to the default MIT mode) and compare the feel. Record with the settings you will use on stage.
- A single speed number also slows the gripper. If the CLI rejects a single number, pass a per-joint dict.

### 9. Gripper and butter (15 min)

- Open and close fully through the leader.
- Grip the real demo object. The default grip is 7% of peak torque. Check it holds without crushing; adjust with `--robot.gripper_torque_ratio` (try 0.05 to 0.1).
- Have several blocks. Butter softens under stage lights and after many grasps.

### 10. Stop drill (10 min)

- Ctrl+C in teleop: LeRobot disables torque on disconnect, so the arm goes limp and drops. Practise it with a hand under the forearm.
- Power switch: same result. Try both until they are reflexes.
- Seeed's rule: if power, a contact or a signal line fails during teleop, stop the program, return the arm to its zero pose, then reconnect and restart.

### 11. Cameras (45 min)

- Wrist UVC camera: `uv run lerobot-find-cameras opencv` and note its index. Orbbec: use the repo's plugin (`type: orbbec`), which has never run on real hardware, so test it here first.
- Give each camera its own USB port on the computer, not a shared hub (Seeed's advice).
- Add both to teleop and check them in Foxglove:

```bash
--robot.cameras='{ front: {type: orbbec, width: 640, height: 480, fps: 30}, wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }'
```

- Check for a steady 30 fps, no frozen frames, the wrist view showing the gripper fingers and the butter from the home pose, and the front view covering the whole work area. If 640x480 is not an Orbbec profile, the plugin's error lists the ones it has.
- Lock the camera mounts. A camera that moves between recording and the demo breaks the policy.
- Camera names are a commitment: they become topic and feature names and must be the same at training and evaluation. This plan uses `front` and `wrist`.

### 12. A short recorded episode in Foxglove (30 min)
Until the recorder exists, use LeRobot's own. This uses only the wrist RGB camera (UVC, index 0); add `front` back once the Orbbec is available.
Until the recorder exists, use LeRobot's own:

```bash
uv run lerobot-record \
  --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 --robot.id=follower --robot.can_adapter=damiao \
  --robot.cameras='{ wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"} }' \
  --teleop.type=rebot_102_leader --teleop.port=/dev/tty.usbserial-2140 --teleop.id=leader \
  --display_data=true --display_mode=foxglove \
  --dataset.repo_id=local/bringup_test --dataset.root=data/bringup_test \
  --dataset.single_task="Pick up the butter" --dataset.num_episodes=3 \
  --dataset.episode_time_s=20 --dataset.reset_time_s=10 --dataset.push_to_hub=false
```

- Right arrow ends an episode early, left arrow re-records it, Esc stops.
- Replay it in Foxglove: `uv run lerobot-dataset-viz --repo-id local/bringup_test --root data/bringup_test --episode-index 0 --display-mode foxglove`
- Check the joint plots line up with the video, no frames drop, and the gripper channel follows.

### 13. Soak test (30 min)

Do the real task by teleop for 20 to 30 minutes. Watch for hot motors, feedback errors (LeRobot cuts torque when motor feedback fails), USB drops and camera stalls. This is the workshop in miniature.

**Done when:** all seven motors answer at the right IDs, both arms are calibrated and the zero holds after a power cycle, every joint follows the leader the right way, the gripper holds the butter, the stop drill is rehearsed, both cameras run at 30 fps, a recorded episode replays cleanly in Foxglove, and 30 minutes of teleop pass without faults.

## Part 2: From tested arms to the demo

The order matters more than the dates: prove the whole loop on five episodes before recording sixty, and plan the failure mode on purpose so Tuesday's fix is a sure thing.

1. **Build the recorder (Thu 19, agent).** A `record-episodes` command (run as `uv run record-episodes`) that runs the LeRobot control loop (follower, leader, cameras), streams live to Foxglove, and writes one MCAP per episode in the format PR #1's converter reads: `/observation/state`, `/action/state`, `/observation/images/<camera>` as h264 `foxglove.CompressedVideo`, plus the `episode` metadata record. Same branch, a read-only arm check script for Part 1 step 7. Fallback if it isn't solid by Friday: record with `uv run lerobot-record` and export each episode to MCAP for review in Foxglove. Both need your yes before anything goes to GitHub; PR #1 should be merged first so the new work doesn't stack on an open branch.
2. **Set the scene (Thu 19 to Fri 20, Mat).** Fix the table layout, tape a start area for the butter (about 20 by 20 cm) and the target spot, fix the lighting (no sunlight), and fix the home pose every episode starts from. Write the task sentence once, for example "Pick up the butter and place it on the plate", and use it everywhere.
3. **Pilot the whole loop (Fri 20, both).** Record 5 episodes, upload them to Foxglove, put them in a dataset with an `outcome` property, then run `uv run build-dataset --dry-run`, the real conversion, a 2,000-step `uv run lerobot-train`, and 10 seconds of the policy on the arm. It will be bad; the point is that every format and command works before the big recording.
4. **Record (Sat 21 to Sun 22, Mat).** About 60 episodes of 15 to 25 s from the home pose, butter spread across the taped area and rotated up to about 30°. Keep a steady pace. Leave one condition out on purpose, such as the far-left corner of the area or a 90° rotation: that becomes the failure you show and fix. Allow about two hours with resets and breaks.

   Full-speed command (default MIT mode: no `pos_vel`, no speed cap; no episode time limit, so you end each episode by hand with `n` for success or `f` for failure):

   ```bash
   uv run record-episode \
     --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 --robot.id=follower --robot.can_adapter=damiao \
     --teleop.type=rebot_102_leader --teleop.port=/dev/tty.usbserial-2140 --teleop.id=leader \
     --task="Pick up the butter and put it on the plate" \
     --num_episodes=60 --episode_time_s=0 --reset_time_s=10 \
     --output_dir=data/episodes
   ```

   Add `--cameras=wrist,front` once the Orbbec is mounted, and use the same camera set for every episode. Keep a hand on the power switch.
5. **Upload and curate (Mon 23, Mat in Foxglove).** Upload the MCAPs, review each episode, set `outcome` to `success` or `failure` with a short reason, and save dataset version 1. Screenshot a couple of rejected takes for the slides. An agent can script the upload.
6. **Build the LeRobot dataset (Mon 23, agent).** With `FOXGLOVE_API_TOKEN` set: `uv run build-dataset --dataset-id <id> --version 1 --cameras front wrist --output outputs/rebot_butter_v1 --repo-id local/rebot_butter_v1`, after a `--dry-run` to check which episodes are kept. This is the converter's first run against the live API, so allow time for a fix.
7. **Train ACT v1 (Mon 23 overnight, agent runs it on your GPU).**

```bash
uv run lerobot-train --policy.type=act --policy.device=cuda \
  --dataset.repo_id=local/rebot_butter_v1 --dataset.root=outputs/rebot_butter_v1 --dataset.video_backend=pyav \
  --batch_size=8 --steps=80000 --save_freq=10000 \
  --output_dir=outputs/train/act_butter_v1 --job_name=act_butter_v1 \
  --wandb.enable=true --policy.push_to_hub=false
```

- Compute: one NVIDIA GPU with 24 GB (RTX 4090, L4, A10G or similar). My rough estimate for 60 episodes (about 36,000 frames) at 50,000 to 80,000 steps is 2 to 5 hours; the pilot run gives a real number. Training on a Mac is too slow to plan around.
- Weights & Biases (or TensorBoard) gives the training curves for the slides; checkpoints every 10,000 steps let you show early versus late behaviour.

8. **Evaluate v1 (Tue 24, Mat at the arm).** Copy the checkpoint to the demo laptop and run 10 trials: 7 across the trained area, 3 in the left-out condition. Stream to Foxglove so you can see where actions and state diverge, and log success per position. Also time inference on the demo laptop itself.

```bash
uv run lerobot-record \
  --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 --robot.id=follower --robot.can_adapter=damiao \
  --robot.cameras='<same cameras as recording>' \
  --policy.path=outputs/train/act_butter_v1/checkpoints/last/pretrained_model \
  --dataset.repo_id=local/eval_butter_v1 --dataset.root=data/eval_butter_v1 \
  --dataset.single_task="Pick up the butter and place it on the plate" \
  --dataset.num_episodes=10 --dataset.episode_time_s=30 \
  --display_data=true --display_mode=foxglove
```

- Eval dataset names must start with `eval_`, and camera names must match training. `uv run lerobot-rollout --strategy.type=base` is a lighter option for the live run on stage.

9. **Fix through the data (Tue 24, both).** Record about 20 episodes in the failing condition, add them to the Foxglove dataset as version 2, rebuild, and train v2 overnight with the same settings.
10. **Evaluate v2 and capture the backup (Wed 25, Mat).** Repeat the same 10 trials and put the before and after success counts on a slide. Film several clean v2 runs (phone video plus a Foxglove screen recording) as the backup. Save the Foxglove layout and export the curves.
11. **Pack (Wed 25, Mat).** Both arms, clamps, both supplies, the switched power strip, USB-CAN board, spare cables, cameras and mounts, butter in a cooler bag plus spares, the plate, tape for the start area, multimeter, and the laptop with the environment, dataset and checkpoints stored locally so nothing depends on venue Wi-Fi.
12. **Dry run (Thu 26, at the venue if you can get in).** Run the full 55 minutes with a timer. Check the zero pose, how the venue lighting changes the camera images, and run the policy 5 times. Decide then whether the live run leads, with the video as backup, or the other way round.
13. **Workshop (Fri 27).**

**Who does what**

| Mat | Agent |
| --- | --- |
| Part 1 bring-up, all hardware steps | Recorder, arm check script, upload script (after your yes) |
| Scene setup, recording, curation calls in Foxglove | `build-dataset` runs and fixes to the converter |
| Evaluation trials and the stop drill | Launching and watching training on a GPU you give access to |
| Backup video, packing, dry run | Failure analysis from eval recordings, slide text, README and QR text for the repo |

One open item from MKT-1685 is still unticked: the UTM link and QR code to the workshop repo and sign-up.

## Open questions

- [ ] Is the leader a reBot Arm 102? Every teleop command here assumes it.
- [ ] How many arms: one follower and one leader, or a second pair for the bimanual post?
- [ ] Which computer drives the arm on stage, the Mac or a Linux laptop?
- [ ] Which GPU will train the policy (a Foxglove machine or a cloud account)?
- [ ] Cameras: Orbbec front plus wrist, wrist only, or the overhead rig?
- [ ] Is the slot confirmed for Friday Nov 27, and can you set up and test on Thursday?
- [ ] Do attendees run anything on their own laptops, or is it a stage demo they follow?

## Sources

- [reBot B601 in the LeRobot docs](https://huggingface.co/docs/lerobot/rebot_b601): install, calibration, teleop, joint directions
- [reBot Arm B601-DM in LeRobot (Seeed wiki)](https://wiki.seeedstudio.com/rebot_arm_b601_dm_lerobot/): wiring, safety, cameras, record, train, evaluate
- [reBot Arm B601-DM getting started (Seeed wiki, Chinese)](https://wiki.seeedstudio.com/cn/rebot_b601_dm_getting_started/): power supply, motor ID table, DM-Tools warnings
- [Damiao motor setup, reBot course chapter 7 (Seeed wiki)](https://wiki.seeedstudio.com/robotics/rebot_physical_ai_course_chapter_7/): MotorBridge gateway and Studio, scanning, test motions
- [LeRobot in the Foxglove docs](https://docs.foxglove.dev/docs/getting-started/frameworks/lerobot): live streaming flags and topics
- LeRobot 0.7.0 source (`rebot_b601_follower` config): default IDs, gains, soft limits and speed limits
- [MKT-1685](https://linear.app/foxglove/issue/MKT-1685/wawtech-workshop) and PR #1 in foxglove/wawtech\_workshop
