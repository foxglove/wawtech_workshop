"""Turn a ``lerobot-train`` log into a CSV and, when matplotlib is installed, loss charts.

    python scripts/parse_train_log.py outputs/logs/act_v2

Reads ``train.log`` (and ``gpu.csv`` if present) in the given directory and writes
``metrics.csv``, ``loss.png`` and ``gpu.png`` next to them. LeRobot prints one line per log
step like ``step:250 smpl:2K ep:1 epch:0.37 loss:5.123 grdn:80.1 lr:1.0e-05 updt_s:0.12
data_s:0.00``; numbers may end in K, M, B or T.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

_PAIR = re.compile(r"(?<![\w.])([a-z][a-z0-9_/]*):(-?[0-9][0-9.]*(?:e[+-]?\d+)?[KMBT]?)(?=\s|$)")
_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}


def _number(text: str) -> float:
    if text[-1] in _SUFFIX:
        return float(text[:-1]) * _SUFFIX[text[-1]]
    return float(text)


def parse_train_log(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="replace").splitlines():
        pairs = {key: _number(value) for key, value in _PAIR.findall(line)}
        if "step" in pairs and "loss" in pairs:
            rows.append(pairs)
    # LeRobot prints big numbers rounded ("step:2K"), so rebuild exact steps from the log
    # interval, which is exact for the first row (a few hundred steps at most).
    if rows:
        interval = rows[0]["step"]
        for index, row in enumerate(rows, start=1):
            row["step"] = interval * index
    return rows


def _write_csv(rows: list[dict[str, float]], path: Path) -> None:
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float]], directory: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; wrote the CSV only.")
        return
    steps = [row["step"] for row in rows]
    panels = [
        (key, label)
        for key, label in (("loss", "Loss"), ("grdn", "Gradient norm"), ("lr", "Learning rate"))
        if key in rows[0]
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 3 * len(panels)), sharex=True)
    for axis, (key, label) in zip([axes] if len(panels) == 1 else axes, panels, strict=True):
        axis.plot(steps, [row.get(key) for row in rows], linewidth=1.5)
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        if key == "loss":
            axis.set_yscale("log")
    axes[-1].set_xlabel("Training step")
    fig.suptitle("ACT training")
    fig.tight_layout()
    fig.savefig(directory / "loss.png", dpi=160)

    gpu = directory / "gpu.csv"
    if gpu.exists():
        with gpu.open() as handle:
            lines = list(csv.reader(handle, skipinitialspace=True))
        data = [line for line in lines[1:] if len(line) == 5]
        if data:
            fig, axis = plt.subplots(figsize=(9, 3))
            axis.plot([float(line[1].split()[0]) for line in data], linewidth=1)
            axis.set_ylabel("GPU utilisation (%)")
            axis.set_xlabel("Sample (every 5 s)")
            axis.set_ylim(0, 100)
            axis.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(directory / "gpu.png", dpi=160)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    directory = Path(sys.argv[1])
    rows = parse_train_log(directory / "train.log")
    if not rows:
        raise SystemExit(f"No 'step: ... loss: ...' lines in {directory / 'train.log'}.")
    _write_csv(rows, directory / "metrics.csv")
    print(f"{len(rows)} log steps; last: {rows[-1]}")
    _plot(rows, directory)


if __name__ == "__main__":
    main()
