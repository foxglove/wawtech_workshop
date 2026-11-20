# Training on the recorded data: ACT on one GPU vs π₀.₅ on Ray/Anyscale

Status: an outline, nothing here has been run. No cloud job has been started and no Anyscale credits have been spent.

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

## Path A: ACT on one GPU (the safe baseline)

```bash
lerobot-train \
  --dataset.repo_id=local/rebot_b601 --dataset.root=outputs/dataset \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --batch_size=8 --steps=50000 --save_freq=10000 \
  --output_dir=outputs/train/act_rebot
```

- Small model (ResNet-18 backbone and a transformer), trained from scratch on our episodes. Any recent 16–24 GB GPU works, including a workstation card or one cloud GPU.
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
