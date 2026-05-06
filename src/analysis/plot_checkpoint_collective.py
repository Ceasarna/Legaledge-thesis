"""
Plot training curves from TensorBoard logs for checkpoint_collective runs.
Stitches consecutive runs into one continuous timeline per agent,
with EMA smoothing and rolling std-dev shaded band.

Produces:
  - Per-agent plots for the "with smart contract" run set
  - Per-agent plots for the "without smart contract" run set
  - Overlay plots comparing the two run sets (averaged across agents)

Usage:
    python -m src.analysis.plot_checkpoint_collective
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Match the visual style used by plot_individual_governance_training.py
sns.set_theme(style="darkgrid")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
LOG_BASE = os.path.join(
    PROJECT_ROOT, ".", "logs", "benchmark",
    "federated", "checkpoint_collective",
)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "checkpoint_collective_plots")

# With smart contract
RUNS_ORDERED_1 = [
    ("20260422_133254", "Run 1"),
]

# Without smart contract
RUNS_ORDERED_2 = [
    ("20260421_141531", "Run 1"),
]

RUN_SETS = {
    "with_contract":    ("With Smart Contract",    RUNS_ORDERED_1),
    "without_contract": ("Without Smart Contract", RUNS_ORDERED_2),
}

AGENTS = ["agent_0", "agent_1", "agent_2", "agent_3"]

AGENT_COLORS = {
    "agent_0": "#1f77b4",
    "agent_1": "#ff7f0e",
    "agent_2": "#2ca02c",
    "agent_3": "#d62728",
}
AGENT_LABELS = {
    "agent_0": "Tesla Model 3 LR",
    "agent_1": "Nissan Leaf",
    "agent_2": "Renault Zoe",
    "agent_3": "Volvo XC90 PHEV",
}

# Overlay (mean view): one curve per configuration — used for compliance metrics
OVERLAY_MEAN_STYLES = {
    "with_contract":    {"color": "#d62728", "label": "Model w/ contract"},
    "without_contract": {"color": "#1f77b4", "label": "Model no contract"},
}

# Overlay (per-agent view): linestyle distinguishes config, color distinguishes agent
OVERLAY_PER_AGENT_STYLES = {
    "with_contract":    {"linestyle": "-",  "label_suffix": "w/ contract"},
    "without_contract": {"linestyle": "--", "label_suffix": "no contract"},
}

# Tags whose overlay should collapse to one curve per config (instead of per-agent).
# Collective utilization is shared across agents in this scenario, so a single
# curve per model is clearer than four overlapping ones.
COMPLIANCE_TAGS = {
    "avg100/compliance_rate",
    "avg100/total_compliance_reward",
    "live/collective_utilization",
}

# Horizontal reference lines to draw on the mean-overlay plot for specific tags.
OVERLAY_HLINES = {
    "live/collective_utilization": (1.0, "Contract limit (util = 1)"),
}

METRICS = [
    ("avg100/episode_energy_kwh", "Episode Energy (kWh)"),
    ("avg100/final_soc",          "Final SoC (%)"),
    ("avg100/total_price_reward", "Price Reward"),
    ("avg100/total_soc_reward",   "SoC Reward"),
    ("avg100/compliance_rate",    "Compliance Rate"),
    ("avg100/total_compliance_reward", "Compliance Reward"),
    ("live/collective_utilization", "Live Collective Util"),
    ("live/budget_usage",          "Live Budget Usage"),
]

# Metrics that get a bottom-left zoom inset on the overlay plot
OVERLAY_ZOOM = {
    "avg100/final_soc": {
        "bounds": [0.06, 0.08, 0.40, 0.40],  # [x0, y0, width, height] in axes fraction
        "tail_frac": 0.25,
    },
}

SMOOTH_WEIGHT = 0.95
STD_WINDOW = 200

_ea_cache = {}


def _get_accumulator(log_dir: str) -> EventAccumulator:
    if log_dir not in _ea_cache:
        print(f"  Loading: {os.path.basename(os.path.dirname(log_dir))}/{os.path.basename(log_dir)}")
        ea = EventAccumulator(log_dir)
        ea.Reload()
        _ea_cache[log_dir] = ea
    return _ea_cache[log_dir]


def load_scalars(log_dir: str, tag: str) -> pd.DataFrame:
    ea = _get_accumulator(log_dir)
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    events = ea.Scalars(tag)
    return pd.DataFrame([(e.step, e.value) for e in events], columns=["step", "value"])


def stitch_runs(runs: list, agent: str, tag: str) -> pd.DataFrame:
    """Load a metric across the given runs and offset steps so they are continuous."""
    frames = []
    step_offset = 0

    for run_dir, _ in runs:
        log_dir = os.path.join(LOG_BASE, run_dir, agent)
        df = load_scalars(log_dir, tag)
        if df.empty:
            continue
        df = df.copy()
        df["step"] = df["step"] + step_offset
        step_offset = df["step"].iloc[-1] + 1
        frames.append(df)

    if not frames:
        return pd.DataFrame(columns=["step", "value"])
    return pd.concat(frames, ignore_index=True)


def stitch_and_average(runs: list, tag: str):
    """Stitch each agent across runs, then average across agents on a shared step grid."""
    agent_dfs = []
    for agent in AGENTS:
        df = stitch_runs(runs, agent, tag)
        if not df.empty:
            agent_dfs.append(df)

    if not agent_dfs:
        return np.array([]), np.array([])

    all_steps = np.sort(np.unique(np.concatenate([df["step"].values for df in agent_dfs])))
    interp = [np.interp(all_steps, df["step"].values, df["value"].values) for df in agent_dfs]
    mean_curve = np.vstack(interp).mean(axis=0)
    return all_steps, mean_curve


def smooth(values: np.ndarray, weight: float = SMOOTH_WEIGHT) -> np.ndarray:
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def rolling_std(values: np.ndarray, window: int = STD_WINDOW) -> np.ndarray:
    s = pd.Series(values)
    return s.rolling(window, min_periods=1, center=True).std().fillna(0).values


def get_run_boundaries(runs: list, tag: str) -> list:
    """Get the x-positions where runs transition for the given run set."""
    boundaries = []
    offset = 0
    for run_dir, _ in runs[:-1]:
        log_dir = os.path.join(LOG_BASE, run_dir, AGENTS[0])
        df = load_scalars(log_dir, tag)
        if not df.empty:
            offset += df["step"].iloc[-1] + 1
            boundaries.append(offset)
    return boundaries


def _tail_xlim(zoom_data, frac):
    max_step = max(steps[-1] for steps, _, _ in zoom_data)
    return (max_step * (1 - frac), max_step)


def _tail_ylim(zoom_data, frac, pad=0.05):
    vals = []
    for steps, ys, _ in zoom_data:
        cutoff = steps[-1] * (1 - frac)
        vals.extend(ys[steps >= cutoff])
    if not vals:
        return None
    lo, hi = float(min(vals)), float(max(vals))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return (lo - pad * span, hi + pad * span)


def _safe_name(tag: str) -> str:
    return tag.replace("/", "_").replace("avg100_", "")


def _save_fig(fig, out_dir: str, basename: str):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{basename}.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")


def plot_run_set(set_key: str, set_label: str, runs: list, out_dir: str):
    """Per-agent plots for a single run set."""
    for tag, label in METRICS:
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for agent in AGENTS:
            df = stitch_runs(runs, agent, tag)
            if df.empty:
                continue

            steps = df["step"].values
            raw = df["value"].values
            smoothed = smooth(raw)
            std = rolling_std(raw)

            color = AGENT_COLORS[agent]
            ax.plot(steps, smoothed, label=AGENT_LABELS[agent], color=color, linewidth=1.4)
            ax.fill_between(steps, smoothed - std, smoothed + std, color=color, alpha=0.15)

        for bx in get_run_boundaries(runs, tag):
            ax.axvline(bx, color="gray", linestyle=":", alpha=0.5, linewidth=0.8)

        ax.set_title(f"{label}  ({set_label})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Training Step")
        ax.set_ylabel(label)
        ax.legend(fontsize=8, loc="best")
        ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_fontsize(9)

        plt.tight_layout()
        _save_fig(fig, out_dir, f"checkpoint_collective_{set_key}_{_safe_name(tag)}")
        plt.close(fig)


def _add_zoom_inset(ax, tag: str, zoom_data: list):
    """zoom_data items: (steps, smoothed_values, color, linestyle, linewidth)."""
    zc = OVERLAY_ZOOM.get(tag)
    if not zc or not zoom_data:
        return
    axins = ax.inset_axes(zc["bounds"])
    for steps, ys, color, ls, lw in zoom_data:
        axins.plot(steps, ys, color=color, linestyle=ls, linewidth=lw)
    xlim = _tail_xlim([(s, y, c) for s, y, c, _, _ in zoom_data], zc["tail_frac"])
    ylim = _tail_ylim([(s, y, c) for s, y, c, _, _ in zoom_data], zc["tail_frac"])
    axins.set_xlim(xlim)
    if ylim is not None:
        axins.set_ylim(ylim)
    axins.tick_params(labelsize=8)
    axins.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    axins.xaxis.get_offset_text().set_fontsize(8)
    ax.indicate_inset_zoom(axins, edgecolor="black")


def _plot_overlay_mean(tag: str, label: str, out_dir: str):
    """One mean curve per configuration (used for compliance metrics)."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    zoom_data = []
    for set_key, (_, runs) in RUN_SETS.items():
        steps, mean_curve = stitch_and_average(runs, tag)
        if len(steps) == 0:
            continue
        smoothed = smooth(mean_curve)
        style = OVERLAY_MEAN_STYLES[set_key]
        ax.plot(steps, smoothed, label=style["label"], color=style["color"], linewidth=2.0)
        zoom_data.append((steps, smoothed, style["color"], "-", 2.0))

    hline = OVERLAY_HLINES.get(tag)
    if hline is not None:
        y, hlabel = hline
        ymin, ymax = ax.get_ylim()
        if ymax > y:
            ax.axhspan(y, ymax, color="red", alpha=0.07, zorder=0, label="Breach zone")
            ax.set_ylim(ymin, ymax)  # lock so the span doesn't expand the axis
        ax.axhline(y, color="black", linestyle=":", linewidth=1.2, alpha=0.8, label=hlabel)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(label)
    ax.legend(fontsize=10, loc="best")
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(9)

    _add_zoom_inset(ax, tag, zoom_data)

    plt.tight_layout()
    _save_fig(fig, out_dir, f"checkpoint_collective_overlay_{_safe_name(tag)}")
    plt.close(fig)


def _plot_overlay_per_agent(tag: str, label: str, out_dir: str):
    """All four agents per configuration — color = agent, linestyle = config."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    zoom_data = []
    for set_key, (_, runs) in RUN_SETS.items():
        style = OVERLAY_PER_AGENT_STYLES[set_key]
        for agent in AGENTS:
            df = stitch_runs(runs, agent, tag)
            if df.empty:
                continue
            steps = df["step"].values
            smoothed = smooth(df["value"].values)
            color = AGENT_COLORS[agent]
            ax.plot(
                steps, smoothed,
                label=f"{AGENT_LABELS[agent]} ({style['label_suffix']})",
                color=color,
                linestyle=style["linestyle"],
                linewidth=1.4,
            )
            zoom_data.append((steps, smoothed, color, style["linestyle"], 1.4))

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Step")
    ax.set_ylabel(label)
    ax.legend(fontsize=7, loc="best", ncol=2)
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(9)

    _add_zoom_inset(ax, tag, zoom_data)

    plt.tight_layout()
    _save_fig(fig, out_dir, f"checkpoint_collective_overlay_{_safe_name(tag)}")
    plt.close(fig)


def plot_overlay(out_dir: str):
    """Overlay both run sets — mean for compliance, per-agent for everything else."""
    for tag, label in METRICS:
        if tag in COMPLIANCE_TAGS:
            _plot_overlay_mean(tag, label, out_dir)
        else:
            _plot_overlay_per_agent(tag, label, out_dir)


def print_summary_table():
    for set_label, runs in RUN_SETS.values():
        print("\n" + "=" * 70)
        print(f"  Summary [{set_label}]: Final 500-step averages (stitched)")
        print("=" * 70)

        header = f"{'Agent':<12}"
        for _, short in METRICS:
            header += f" {short:>14}"
        print(header)
        print("-" * (12 + 15 * len(METRICS)))

        for agent in AGENTS:
            row = f"{agent:<12}"
            for tag, _ in METRICS:
                df = stitch_runs(runs, agent, tag)
                if df.empty:
                    row += f" {'N/A':>14}"
                else:
                    tail_avg = df["value"].tail(500).mean()
                    row += f" {tail_avg:>14.3f}"
            print(row)
        print("-" * (12 + 15 * len(METRICS)))


if __name__ == "__main__":
    for set_key, (set_label, runs) in RUN_SETS.items():
        out_dir = os.path.join(OUTPUT_DIR, set_key)
        plot_run_set(set_key, set_label, runs, out_dir)

    plot_overlay(os.path.join(OUTPUT_DIR, "overlay"))
