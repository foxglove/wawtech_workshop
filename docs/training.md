# Training on the recorded data: ACT on one GPU vs π₀.₅ on Ray/Anyscale

Status: Path A (ACT on one GPU) has been run twice: on 6 episodes, and on the 67-episode `butter_pass` version 2 (below). Neither policy has been tested on the arm yet. Path B (π₀.₅ on Anyscale) is still an outline: no cloud job has been started and no Anyscale credits have been spent.

Both paths start from the same LeRobot v3 dataset:

```bash
uv run record-episode ...                       # MCAPs in data/episodes
uv run upload-episodes data/episodes --dataset-id <id>   # optional: curate in Foxglove, commit
uv run build-dataset --dataset-id <id> --version <n> \
    --cameras wrist --output outputs/dataset --repo-id local/rebot_b601
# or, without Foxglove: uv run build-dataset --mcap-dir data/episodes ...
```

The dataset already carries the quantile statistics (`q01` ... `q99`) that π₀.₅ normalizes with; LeRobot 0.7 writes them when the converter finalizes the dataset, so no separate stats pass is needed.

Training needs extras the recording machine does not: `lerobot[training]` for both, plus `lerobot[pi]` for π₀.₅. Install them on the training machine, not in this workspace's lock.

## Streaming a dataset version into your own PyTorch loop

`lerobot-train` needs the converted dataset above. For a custom loop, `wawtech_workshop.training_data` streams a committed dataset version straight into a DataLoader with the SDK's `read_dataset`, with no conversion step. Samples use LeRobot's keys (`observation.state`, `action`, `observation.images.wrist`, `task`) and the same 30 fps zero-order-hold alignment and `outcome == "success"` filter as `build-dataset`.

```bash
export FOXGLOVE_API_TOKEN=...   # key with datasets.list and data.stream; never commit it
uv run python -m wawtech_workshop.training_data --dataset-id dset_0ek8nUJE8FYkiGun --version 2 --task "pass the butter"
```

```python
from torch.utils.data import DataLoader
from wawtech_workshop.training_data import SampleConfig, build_torch_dataset

dataset = build_torch_dataset("dset_0ek8nUJE8FYkiGun", 2, SampleConfig(chunk_size=100))  # ACT chunks
loader = DataLoader(dataset, batch_size=8, num_workers=4)
```

DataLoader shuffling and `DistributedSampler` do not apply; pass `shuffle=True` to `build_torch_dataset` to shuffle episode order. Under DDP, workers yield unequal sample counts, so use `torch.distributed.algorithms.join.Join`.

## Path A: ACT on one GPU (the safe baseline)

```bash
lerobot-train \
  --dataset.repo_id=local/rebot_b601 --dataset.root=outputs/dataset \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=50000 --save_freq=10000 \
  --output_dir=outputs/train/act_rebot
```

- Small model (ImageNet-pretrained ResNet-18 backbone and a transformer, the rest trained from scratch on our episodes). Any recent 16–24 GB GPU works, including a workstation card or one cloud GPU.
- Standard for single-task pick and place on low-cost arms with roughly 50 demonstrations. It does not use the task text.
- Low risk. If it fails, the cause is almost always the data (camera placement, inconsistent demos), which we can fix by recording again.

## Path B: π₀.₅ fine-tune on Anyscale (Ray)

```bash
lerobot-train \
  --dataset.repo_id=local/rebot_b601 --dataset.root=outputs/dataset \
  --policy.type=pi05 --policy.pretrained_path=lerobot/pi05_base \
  --policy.dtype=bfloat16 --policy.gradient_checkpointing=true \
  --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=32 --steps=3000 \
  --output_dir=outputs/train/pi05_rebot
```

- 2B-parameter vision-language backbone plus a 300M action expert. A full fine-tune is sized for one 80 GB GPU (A100 or H100). To fit a smaller card, add `--policy.train_expert_only=true` or `--policy.freeze_vision_encoder=true`, or LoRA with `--policy.use_peft=true` (`lerobot[peft]`).
- Needs a Hugging Face token that has accepted the gated PaliGemma licence (the tokenizer is `google/paligemma-3b-pt-224`).
- π₀.₅ was pretrained with three camera views. With the wrist camera alone, `--policy.empty_cameras=2` pads the missing views (`1` with wrist and front). A wrist-only view is the weakest setup for either policy, since nothing sees where the object is before the gripper is close: if the Orbbec arrives, record the training episodes with `--cameras=wrist,front`.
- Uses the task string, so one checkpoint can take several tasks (for example "pick up the butter" and "put the butter on the plate").

How it would run on Anyscale:

1. Put the dataset where the cluster can read it: push it to a private Hugging Face dataset repo, or copy `outputs/dataset` to the cloud storage bucket attached to the Anyscale workspace.
2. Create an Anyscale Job with one GPU worker node (1× A100/H100 80 GB), an image with `lerobot[training,pi]`, and `HF_TOKEN` as a secret. The entrypoint is the command above; for several GPUs on one node, `torchrun --nproc-per-node=<n> $(which lerobot-train) ... --parallelism.dp_shard=<n>`.
3. Save checkpoints to the bucket and copy the final one to the demo laptop.

Ray here mostly provisions and schedules the GPU node. LeRobot's trainer uses torchrun and Accelerate, not Ray Train, so wrapping it in a Ray `TorchTrainer` adds work without speeding up a single-node run. The Foxglove SDK also ships `foxglove.datasets.ray.read_dataset` (`foxglove-sdk[ray]`), which streams a dataset version into Ray Data. That would let a Ray job convert episodes in the cluster instead of on a laptop, but LeRobot still trains from a LeRobot dataset, so conversion stays a separate step.

Before launching, check the hourly rate of the chosen GPU node in the Anyscale console and set a maximum runtime on the job, so a stuck run cannot burn the 3k credits.

## Training on the training PC (what we ran)

The ACT run on `butter_pass` version 2 (dataset `dset_0ek8nUJE8FYkiGun`, 67 episodes, 54,699 frames at 30 fps) ran on a separate Linux PC with an RTX 2070 SUPER (8 GB), not on the laptop. `scripts/train_act.sh` does the whole job: it converts the dataset version, then trains, and writes logs.

| | |
|---|---|
| Settings | ACT, batch 8, 50,000 steps (about 7.3 epochs), checkpoint every 5,000, log every 100; every other setting is LeRobot's ACT default (`chunk_size` 100, `resnet18`, `lr` 1e-5, `kl_weight` 10) |
| Speed | 5.9 steps/s, 2 h 22 min in total. GPU memory: 2.1 GB allocated by PyTorch, 5.5 GB peak as `nvidia-smi` sees it (of 8 GB) |
| Loss | 2.02 at step 1k, 0.20 at 10k, 0.124 at 20k, 0.098 at 30k, 0.084 at 40k, **0.074 at 50k** |
| Settled? | Nearly: the last 5k steps averaged about 5% below the 5k before. There is no validation set, so only a rollout on the arm shows whether a checkpoint is good |

### Run it again

1. **Put the code and the token on the PC.** The PC's copy of the repo is a plain folder, not a git checkout. Use `tar` over ssh (the macOS `rsync` is `openrsync`, which reported success without copying here). `TRAIN_PC` is `user@address` of the PC:

   ```bash
   TRAIN_PC=mat@<pc-address>
   tar --exclude __pycache__ -cf - src scripts | ssh $TRAIN_PC 'cd ~/wawtech_workshop && tar -xf -'
   ssh $TRAIN_PC 'umask 077; cat > ~/wawtech_workshop/.env' < .env   # FOXGLOVE_API_TOKEN, mode 600
   ```

   The key needs the `datasets.list` and `data.stream` capabilities. Delete the PC's `.env` when you are done.

2. **One-time environment fixes.** The PC's virtualenv needed two repairs, which a fresh PC will probably need too:

   ```bash
   cd ~/wawtech_workshop; export PATH=$HOME/.local/bin:$PATH VIRTUAL_ENV=$PWD/.venv
   # Two OpenCV builds installed over each other and `import cv2` crashed. Keep only headless.
   uv pip uninstall opencv-python opencv-python-headless
   rm -rf .venv/lib/python3.12/site-packages/cv2
   uv pip install --no-deps "opencv-python-headless==4.13.0.92"
   # lerobot-train needs accelerate and friends, which are not in the locked environment.
   uv pip install "lerobot[training]==0.7.0"
   ```

3. **Start it in `tmux`** so it survives the ssh session:

   ```bash
   ssh $TRAIN_PC
   cd ~/wawtech_workshop
   tmux new-session -d -s train "bash scripts/train_act.sh; echo exit=\$?; sleep 86400"
   ```

   Settings come from variables: `STEPS`, `RUN` (names the output folders), `VERSION`, `DATASET_ID`, `TASK`, `BATCH_SIZE`, `SAVE_FREQ`, `LOG_FREQ`. For example `STEPS=30000 RUN=act_v2_short bash scripts/train_act.sh`. The script skips the conversion if `outputs/dataset_v<VERSION>` exists, and `lerobot-train` refuses to start if `outputs/train/<RUN>` exists, so pick a new `RUN` or delete the old folder.

   The v2 episodes carry no `outcome` or `task` property, so the script converts with `--keep-unlabeled --task "pass the butter"`. Without those two flags the filter keeps zero episodes. The first request to the Foxglove API can take about 10 minutes before the downloads start.

4. **Watch it.** Training prints a progress bar and one metrics line per 100 steps:

   ```bash
   ssh $TRAIN_PC "tr '\r' '\n' < ~/wawtech_workshop/outputs/logs/act_v2/train.log | grep -a 'loss:' | tail -5"
   ssh -t $TRAIN_PC tmux attach -t train      # detach with Ctrl-b d, never Ctrl-C
   ```

5. **Copy the results back.** The weights are the `pretrained_model` folders (about 207 MB each). The `training_state` folders (about 385 MB each) hold the optimizer and are only needed to resume, so leave them behind:

   ```bash
   ssh $TRAIN_PC 'cd ~/wawtech_workshop && tar -cf - --exclude=training_state outputs/train/act_v2 outputs/logs/act_v2' | tar -xf -
   uv run python scripts/parse_train_log.py outputs/logs/act_v2    # metrics.csv (+ PNGs with matplotlib)
   ```

Logs for a run live in `outputs/logs/<RUN>/`, not in the training output folder: `train.log`, `gpu.csv` (GPU use, memory, temperature and power every 5 s), `run_info.txt` and `build_dataset.log`. `parse_train_log.py` rebuilds exact step numbers, because LeRobot prints rounded ones (`step:2K`).

## Run inference with the trained weights

The run left ten checkpoints on the laptop, in `outputs/train/act_v2/checkpoints/{005000,...,050000}/pretrained_model` (`last` points to `050000`). Each folder is self-contained: `model.safetensors` plus the config and the input and output normalisers, so nothing else is needed to run it. The policy takes `observation.state` (7 values) and `observation.images.wrist` (640x480 RGB) and returns 7 joint targets, in chunks of 100 steps.

**1. Check the weights load, with no arm attached.** LeRobot falls back from `cuda` to `mps` (Apple GPU) or CPU on its own:

```python
import time, torch
from lerobot.policies.act.modeling_act import ACTPolicy

path = "outputs/train/act_v2/checkpoints/050000/pretrained_model"
policy = ACTPolicy.from_pretrained(path).eval()  # device=cuda in the config falls back to mps/cpu
device = next(policy.parameters()).device
batch = {k: torch.rand(1, *f.shape, device=device) for k, f in policy.config.input_features.items()}
with torch.no_grad():
    policy.reset()
    start = time.time()
    action = policy.select_action(batch)
print(device, tuple(action.shape), f"{(time.time() - start) * 1000:.0f} ms")
```

On the laptop this printed `mps:0 (1, 7)`. A warm call that plans a new chunk took about 11 ms on `mps` and 63 ms on CPU; the very first call is slower (about 200 ms) from warm-up. That is fast enough for 30 fps control on the laptop.

**2. Run it on the arm.** In LeRobot 0.7 policies run through `lerobot-rollout`. `lerobot-record` has no `--policy.*` flags there, so the `lerobot-record --policy.path` command in the demo plan does not work.

```bash
uv run lerobot-rollout --strategy.type=base \
  --robot.type=rebot_b601_follower --robot.port=/dev/tty.usbmodem00000000050C1 \
  --robot.id=follower --robot.can_adapter=damiao \
  --robot.control_mode=pos_vel --robot.pos_vel_velocity=30 --robot.max_relative_target=5 \
  --robot.cameras='{wrist: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}' \
  --policy.path=outputs/train/act_v2/checkpoints/050000/pretrained_model --policy.device=mps \
  --task="pass the butter" --duration=20 \
  --display_data=true --display_mode=foxglove
```

Not yet run on the arm. Read these before the first run:

- **Safety.** The arm has no hardware e-stop: the switched 24 V power strip is the e-stop, and cutting power drops the arm under gravity. Keep a hand on the switch. `pos_vel_velocity=30` (degrees per second) and `max_relative_target=5` (degrees per command) are conservative starting values I chose, not tested ones; raise them only once the motion looks right. `--duration` is in seconds, and `0` runs until you stop it. Start with a short one.
- **Match training.** The camera must be named `wrist`, at 640x480 and 30 fps, mounted as it was for recording. A camera that moved, or different lighting, breaks the policy. Start the butter and plate where they were in the episodes.
- **Which checkpoint.** Loss is only a guide, so run the same trials on `020000`, `035000` and `050000` by changing `--policy.path`, and keep the earliest one that does as well as the last. If the loss is better but the arm is not, the data is the problem (see Path A).
- **Watch it in Foxglove.** `--display_mode=foxglove` serves a live view you can open as a WebSocket connection. Compare `/observation/state` with the policy's actions to see where they diverge.
- **If a flag is rejected,** run `uv run lerobot-rollout --help`. `--policy.device=mps` can be dropped, because it falls back by itself.

## Tradeoffs for 27 November

| | ACT, one GPU | π₀.₅, Anyscale |
|---|---|---|
| Setup | `lerobot-train` on any CUDA box | Job image, secrets, storage, gated model access |
| Hardware | 16–24 GB GPU | 80 GB GPU (smaller with LoRA or frozen parts) |
| Data needed | ~50 good demos, one task | Fewer demos per task can work; benefits from varied tasks |
| Inference on the demo laptop | Real time on a laptop GPU | Needs a large GPU at the arm, or a policy server (`lerobot-policy-server`) over the network |
| Risk | Low, well-trodden | Higher: first run on our data, first Anyscale job, network inference |
| Story for the talk | Teleop to policy in one afternoon | Fine-tuning a foundation model from a Foxglove dataset |

Recommendation: record once and train ACT first on a single GPU as the demo's guaranteed path. Run the π₀.₅ fine-tune on Anyscale in parallel from the same dataset version, and show it only if it rolls out reliably on the real arm by the 25th. The deciding question is inference: π₀.₅ needs a big GPU near the arm at run time, which the venue may not have.
