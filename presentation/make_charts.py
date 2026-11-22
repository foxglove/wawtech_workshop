"""Render the slide charts from ``data/`` into ``charts/`` as light and dark PNGs.

    uv run --no-project --with matplotlib --with numpy python presentation/make_charts.py

Inputs (all in ``data/``): ``metrics.csv`` (from scripts/parse_train_log.py), ``gpu.csv``
(nvidia-smi every 5 s), ``episodes.csv`` (episode lengths in frames at 30 fps).

Every chart is one measure on one axis. Where two measures have different scales (loss
terms, GPU metrics) they get their own panels instead of a second y-axis. A single series
uses one accent colour (the palette's blue) and is labelled directly, so there is no legend.
"""

from __future__ import annotations

import csv
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

HERE = Path(__file__).parent
DATA = HERE / "data"
OUT = HERE / "charts"
FPS = 30

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "series": "#2a78d6"},
    "dark": {"surface": "#1a1a19", "ink": "#f0efec", "ink2": "#c3c2b7", "series": "#3987e5"},
}


def read_metrics() -> dict[str, np.ndarray]:
    with (DATA / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}


def read_gpu() -> dict[str, np.ndarray]:
    times, util, mem, temp, power = [], [], [], [], []
    for line in (DATA / "gpu.csv").read_text().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5 or not parts[0][:1].isdigit():
            continue
        times.append(datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp())
        util.append(float(parts[1].split()[0]))
        mem.append(float(parts[2].split()[0]))
        temp.append(float(parts[3]))
        power.append(float(parts[4].split()[0]))
    minutes = (np.array(times) - times[0]) / 60
    return {
        "minutes": minutes,
        "util": np.array(util),
        "mem_gb": np.array(mem) / 1024,
        "temp": np.array(temp),
        "power": np.array(power),
    }


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered geometric mean (a mean on the log scale, matching the log axis).

    The window shrinks symmetrically at both ends, so the line meets the first and last
    values. An arithmetic mean would jump when the huge first values leave the window, and a
    trailing one would lag the curve.
    """
    logs = np.log(values)
    half = window // 2
    out = np.empty_like(values)
    for index in range(len(values)):
        reach = min(half, index, len(values) - 1 - index)
        out[index] = np.exp(logs[index - reach : index + reach + 1].mean())
    return out


def plain_log_labels(ax, minor: bool = False) -> None:
    """Show 0.1 and 0.2 instead of 10^-1 and 2x10^-1 on a log y-axis."""

    def label(value, _):
        return f"{value:g}"

    ax.yaxis.set_major_formatter(label)
    ax.yaxis.set_minor_formatter(label if minor else matplotlib.ticker.NullFormatter())


def style(ax, theme: dict[str, str]) -> None:
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["ink2"])
        ax.spines[side].set_alpha(0.5)
    ax.tick_params(which="both", colors=theme["ink2"], labelsize=10)
    ax.tick_params(which="major", length=3)
    ax.grid(axis="y", color=theme["ink2"], alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)


def figure(theme: dict[str, str], rows: int = 1, height: float = 4.6, sharex: bool = False):
    fig, axes = plt.subplots(
        rows, 1, figsize=(8, height), sharex=sharex, facecolor=theme["surface"], squeeze=False
    )
    axes = axes[:, 0]
    for ax in axes:
        style(ax, theme)
    return fig, axes


def finish(fig, theme: dict[str, str], title: str, subtitle: str, name: str, mode: str) -> None:
    fig.text(0.02, 0.965, title, color=theme["ink"], fontsize=15, fontweight="bold", va="top")
    subtitle = textwrap.fill(subtitle, 96)
    lines = subtitle.count("\n") + 1
    fig.text(0.02, 0.915, subtitle, color=theme["ink2"], fontsize=10.5, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.88 - 0.045 * (lines - 1)))
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}_{mode}.png", dpi=200, facecolor=theme["surface"])
    plt.close(fig)


def annotate(ax, theme, x, y, text, dx=0, dy=0, ha="left"):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        color=theme["ink"],
        fontsize=10,
        ha=ha,
        va="center",
    )


def loss_curve(m, theme, mode):
    fig, (ax,) = figure(theme)
    step, loss = m["step"], m["loss"]
    ax.plot(step, loss, color=theme["series"], alpha=0.28, linewidth=1.2)
    ax.plot(step, smooth(loss, 10), color=theme["series"], linewidth=2.2)
    ax.set_yscale("log")
    plain_log_labels(ax)
    ax.set_xlim(0, step[-1] * 1.12)
    ax.set_xlabel("Training step", color=theme["ink2"])
    ax.set_ylabel("Loss (log scale)", color=theme["ink2"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v / 1000)}k" if v else "0")
    for target, dx, dy in ((100, 10, 0), (10_000, 10, 12), (30_000, 10, 14), (50_000, 8, 0)):
        index = int(np.argmin(np.abs(step - target)))
        ax.plot(
            step[index],
            loss[index],
            "o",
            color=theme["series"],
            markersize=7,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
        )
        annotate(ax, theme, step[index], loss[index], f"{loss[index]:.3g}", dx, dy)
    fall = loss[0] / loss[-1]
    finish(
        fig,
        theme,
        f"Loss fell {fall:.0f}x in {int(step[-1]):,} steps",
        "Total = L1 action error + 10 x KL term; the KL part is gone by about step 10k. "
        "Faint line: every 100 steps. Solid line: smoothed over 1,000 steps.",
        "loss_curve",
        mode,
    )


def loss_terms(m, theme, mode):
    fig, (top, bottom) = figure(theme, rows=2, height=6.2, sharex=True)
    step = m["step"]
    top.plot(step, m["l1_loss"], color=theme["series"], linewidth=2)
    top.set_yscale("log")
    plain_log_labels(top)
    top.set_yticks([0.07, 0.1, 0.2, 0.3, 0.5])
    top.set_ylabel("L1 loss: action error\n(log scale)", color=theme["ink2"])
    bottom.plot(step, m["kld_loss"], color=theme["series"], linewidth=2)
    bottom.set_ylabel("KL term: latent regulariser", color=theme["ink2"])
    bottom.set_xlabel("Training step", color=theme["ink2"])
    bottom.xaxis.set_major_formatter(lambda v, _: f"{int(v / 1000)}k" if v else "0")
    top.set_xlim(0, step[-1])
    l1 = m["l1_loss"]
    finish(
        fig,
        theme,
        f"Action error fell {l1[0] / l1[-1]:.0f}x; the KL term collapsed early",
        f"L1 went from {l1[0]:.2f} to {l1[-1]:.3f}. KL is printed to 3 decimals, so it reads 0 "
        "late on. Different scales, so one panel each.",
        "loss_terms",
        mode,
    )


def grad_norm(m, theme, mode):
    fig, (ax,) = figure(theme)
    step, grdn = m["step"], m["grdn"]
    ax.plot(step, grdn, color=theme["series"], alpha=0.28, linewidth=1.2)
    ax.plot(step, smooth(grdn, 10), color=theme["series"], linewidth=2.2)
    ax.set_yscale("log")
    plain_log_labels(ax)
    ax.set_xlim(0, step[-1] * 1.1)
    ax.set_xlabel("Training step", color=theme["ink2"])
    ax.set_ylabel("Gradient norm (log scale)", color=theme["ink2"])
    ax.xaxis.set_major_formatter(lambda v, _: f"{int(v / 1000)}k" if v else "0")
    annotate(ax, theme, step[0], grdn[0], f"{grdn[0]:.0f}", 10, 0)
    annotate(ax, theme, step[-1], grdn[-1], f"{grdn[-1]:.1f}", 8, 0)
    finish(
        fig,
        theme,
        "Gradients shrank and stayed stable",
        "No spikes or divergence over the run. Faint line: every 100 steps. "
        "Solid line: smoothed over 1,000 steps.",
        "grad_norm",
        mode,
    )


def gpu_use(g, theme, mode):
    fig, axes = figure(theme, rows=3, height=7.2, sharex=True)
    panels = (
        ("util", "GPU utilisation (%)", (0, 105)),
        ("temp", "Temperature (°C)", None),
        ("power", "Power draw (W)", None),
    )
    for ax, (key, label, limits) in zip(axes, panels, strict=True):
        ax.plot(g["minutes"], g[key], color=theme["series"], linewidth=1.4)
        ax.set_ylabel(label, color=theme["ink2"])
        if limits:
            ax.set_ylim(*limits)
    axes[-1].set_xlabel("Minutes since training started", color=theme["ink2"])
    axes[0].set_xlim(0, g["minutes"][-1])
    finish(
        fig,
        theme,
        f"The GPU ran flat out for {g['minutes'][-1] / 60:.1f} hours",
        f"RTX 2070 SUPER, one sample every 5 s. Median utilisation {np.median(g['util']):.0f}%, "
        f"peak memory {g['mem_gb'].max():.1f} GB of 8.",
        "gpu_use",
        mode,
    )


def episodes(theme, mode):
    with (DATA / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    seconds = np.array([int(row["length"]) for row in rows]) / FPS
    fig, (ax,) = figure(theme)
    ax.bar(np.arange(len(seconds)), seconds, color=theme["series"], width=0.78)
    ax.hlines(
        seconds.mean(),
        -1,
        len(seconds) - 0.5,
        color=theme["ink2"],
        linewidth=1.2,
        linestyle=(0, (4, 3)),
    )
    mean_label = f"mean {seconds.mean():.0f} s"
    annotate(ax, theme, len(seconds) - 0.5, seconds.mean(), mean_label, 6, 0)
    ax.set_xlabel("Episode (in dataset order)", color=theme["ink2"])
    ax.set_ylabel("Length (seconds)", color=theme["ink2"])
    ax.set_xlim(-1, len(seconds) + 9)
    ax.set_xticks(range(0, len(seconds), 10))
    finish(
        fig,
        theme,
        f"{len(seconds)} demonstrations, {seconds.sum() / 60:.0f} minutes of teleoperation",
        f"Butter pass v2. Shortest {seconds.min():.0f} s, longest {seconds.max():.0f} s, "
        f"{int(seconds.sum() * FPS):,} frames at {FPS} fps.",
        "episodes",
        mode,
    )


def main() -> None:
    metrics, gpu = read_metrics(), read_gpu()
    for mode, theme in THEMES.items():
        loss_curve(metrics, theme, mode)
        loss_terms(metrics, theme, mode)
        grad_norm(metrics, theme, mode)
        gpu_use(gpu, theme, mode)
        episodes(theme, mode)
    print(f"wrote {len(list(OUT.glob('*.png')))} charts to {OUT}")


if __name__ == "__main__":
    main()
