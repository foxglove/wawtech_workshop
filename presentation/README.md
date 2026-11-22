# Presentation material

Everything for the talk that came from the training run on the training PC: logs, dataset facts, charts and sample camera frames. The 2 GB of model weights stay in `outputs/train/act_v2/checkpoints/` (the `pretrained_model` folders), not here.

## Key numbers

| | |
|---|---|
| Data | 67 teleoperated episodes of "pass the butter", 54,699 frames at 30 fps, 30 minutes (18 to 50 s each, mean 27 s). One wrist camera, 640x480 |
| Model | ACT: an ImageNet-pretrained ResNet-18 vision backbone plus a transformer with a VAE latent (32 dims), 7 joint targets per step, 100-step action chunks. Everything except the backbone's starting weights is trained from scratch on these episodes |
| Training | 50,000 steps, batch 8, about 7.3 passes over the data. 2 h 22 min on one RTX 2070 SUPER (8 GB), 5.9 steps/s |
| Loss | Total 10.2 at step 100 to 0.074 at step 50,000. Total = L1 action error + 10 x KL term. L1 fell 0.60 to 0.072 (8x); the KL term was near 0 by about step 10k |
| Still improving? | Slowly: the last 5,000 steps averaged about 5% below the 5,000 before |
| Earlier run | 6 episodes (6,325 frames, 3.5 minutes), 30,000 steps, about 38 passes over the data. Same settings otherwise |
| Inference | On the laptop (Apple GPU) one forward pass that plans a new chunk takes about 11 ms warm |

Not measured yet, so do not put them on a slide: how well the policy does on the arm, and any held-out error (there is no validation set). The earlier 6-episode run left no loss log, so there is no curve to compare against, only its settings.

## Contents

| Folder | What is in it |
|---|---|
| `charts/` | Five charts, each as `_light.png` and `_dark.png` (1600 px wide): `loss_curve`, `loss_terms` (L1 and KL in their own panels), `grad_norm`, `gpu_use` (utilisation, temperature, power over the run), `episodes` (length of each demonstration) |
| `frames/` | Six wrist-camera stills from episode 0, at 2% to 97% of the episode: what the policy sees |
| `data/metrics.csv` | One row per 100 training steps: step, loss, `l1_loss`, `kld_loss`, grad norm, step time, GPU memory, samples per second |
| `data/train.log`, `data/gpu.csv`, `data/run_info.txt` | The raw training log, the GPU log (one sample every 5 s) and the run's settings and versions |
| `data/episodes.csv` | Length in frames of each of the 67 episodes |
| `data/dataset_v2_info.json`, `data/dataset_v2_stats.json` | The converted dataset's features and per-feature statistics |
| `data/train_config_67ep_act_v2.json`, `data/train_config_6ep_act_30k.json` | The full training settings of both runs |
| `data/hardware.txt` | The training PC: Ryzen 5 3600X, 46 GB RAM, RTX 2070 SUPER, Ubuntu 24.04 |
| `data/dataset_6ep_info.json` | The 6-episode dataset for the comparison |
| `make_charts.py` | Redraws all charts from `data/` |

## Redraw the charts

```bash
uv run --no-project --with matplotlib --with numpy python presentation/make_charts.py
```

`--no-project` keeps matplotlib out of the project's environment. To rebuild `data/metrics.csv` from a new log, run `uv run python scripts/parse_train_log.py <log folder>`. For ideas, a loss curve with the checkpoints marked (every 5,000 steps) pairs well with a rollout video of an early and a late checkpoint on the arm.
