"""
Plot training curves for the four Individual Governance configurations
comparing no-contract vs contract-aware training for Centralized RL and FRL.

Configurations:
  1. Centralized RL, no contract  → logs/benchmark/agent_4201
  2. Centralized RL, contract     → logs/benchmark/agent_4202
  3. FRL, no contract             → logs/benchmark/federated/no_contract/20260416_161136
  4. FRL, contract                → logs/benchmark/federated/contract/20260416_142923

Produces two figures:
  - individual_governance_reward_curves.png (+ .pdf): reward over training timesteps
  - individual_governance_compliance_curves.png (+ .pdf): compliance rate over timesteps

X-axis is PPO training timesteps (0..1,920,000) so all four curves share the
same budget.

Usage:
    python -m pytorchexample.eval.plot_individual_governance_training
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Apply the OpenAI Spinning Up visual style
sns.set_theme(style="darkgrid")

PROJECT_ROOT = r"C:\Users\toek3476\FRL\Thesis_2\Thesis_code"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "quickstart-pytorch"))

LOG_BASE = os.path.join(PROJECT_ROOT, "quickstart-pytorch", "logs", "benchmark")
RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "quickstart-pytorch", "pytorchexample",
    "eval", "individual_governance_results",
)

# Per-episode axis is logged every 1 episode; each episode is 24 PPO steps.
STEPS_PER_EPISODE = 24
TOTAL_TIMESTEPS = 1_920_000

# Tag conventions discovered in the existing logs:
#   - rollout/ep_rew_mean lives in {agent_dir}/PPO_0/ and its step IS PPO timesteps
#   - avg100/compliance_rate lives in {agent_dir}/ and its step is an episode
#     index, so we multiply by STEPS_PER_EPISODE to put it on the same axis.
REWARD_TAG = "rollout/ep_rew_mean"
REWARD_SUBDIR = "PPO_0"
REWARD_STEP_IS_TIMESTEPS = True

COMPLIANCE_TAG = "avg100/compliance_rate"
COMPLIANCE_SUBDIR = ""  # tag lives directly under the agent dir
COMPLIANCE_STEP_IS_TIMESTEPS = False

# (label, agent_dirs) -for Centralized there is one agent dir; for FRL there
# are four client dirs whose curves we average.
CONFIGS = [
    {
        "label":  "Centralized, no contract",
        "color":  "#1f77b4",
        "agent_dirs": [os.path.join(LOG_BASE, "agent_4201")],
    },
    {
        "label":  "Centralized, contract",
        "color":  "#ff7f0e",
        "agent_dirs": [os.path.join(LOG_BASE, "agent_4202")],
    },
    {
        "label":  "FRL, no contract",
        "color":  "#2ca02c",
        "agent_dirs": [
            os.path.join(LOG_BASE, "federated", "no_contract",
                         "20260416_161136", f"agent_{i}")
            for i in range(4)
        ],
    },
    {
        "label":  "FRL, contract",
        "color":  "#d62728",
        "agent_dirs": [
            os.path.join(LOG_BASE, "federated", "contract",
                         "20260416_142923", f"agent_{i}")
            for i in range(4)
        ],
    },
]


_ea_cache = {}


def _get_accumulator(log_dir: str) -> EventAccumulator:
    if log_dir not in _ea_cache:
        ea = EventAccumulator(log_dir)
        ea.Reload()
        _ea_cache[log_dir] = ea
    return _ea_cache[log_dir]


def load_scalars(log_dir: str, tag: str) -> pd.DataFrame:
    if not os.path.isdir(log_dir):
        return pd.DataFrame(columns=["step", "value"])
    ea = _get_accumulator(log_dir)
    if tag not in ea.Tags().get("scalars", []):
        return pd.DataFrame(columns=["step", "value"])
    events = ea.Scalars(tag)
    return pd.DataFrame([(e.step, e.value) for e in events], columns=["step", "value"])


def smooth_ema(values: np.ndarray, weight: float = 0.0) -> np.ndarray:
    """TensorBoard-style exponential moving average."""
    if len(values) == 0:
        return values
    s = pd.Series(values)
    return s.ewm(alpha=1-weight, adjust=False).mean().values


def _load_config_curve(config: dict, tag: str, subdir: str,
                       step_is_timesteps: bool, smoothing_weight: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a tag across all agent_dirs and return (steps, raw_mean, smoothed_mean).

    Multiple agents (FRL) are aligned on a shared step grid via linear
    interpolation and their values are averaged. The returned step axis is
    always in PPO training timesteps.
    """
    dfs = []
    for agent_dir in config["agent_dirs"]:
        log_dir = os.path.join(agent_dir, subdir) if subdir else agent_dir
        df = load_scalars(log_dir, tag)
        if df.empty:
            print(f"  [MISS] {log_dir} :: {tag}")
            continue
        if not step_is_timesteps:
            df = df.copy()
            df["step"] = df["step"] * STEPS_PER_EPISODE
        dfs.append(df)

    if not dfs:
        return np.array([]), np.array([]), np.array([])

    # Shared step grid: union of all steps across agents, then interpolate.
    all_steps = np.sort(np.unique(np.concatenate([df["step"].values for df in dfs])))
    
    raw_agents = []
    for df in dfs:
        raw_agents.append(np.interp(all_steps, df["step"].values, df["value"].values))

    raw_mean = np.vstack(raw_agents).mean(axis=0)
    smoothed_mean = smooth_ema(raw_mean, weight=smoothing_weight)

    return all_steps, raw_mean, smoothed_mean


def plot_metric(tag: str, subdir: str, step_is_timesteps: bool,
                 ylabel: str, title: str, out_basename: str, smoothing_weight: float = 0.0,
                 zoom_kwargs: dict = None):
    fig, ax = plt.subplots(figsize=(8, 4.5))

    axins = None
    if zoom_kwargs:
        axins = ax.inset_axes(zoom_kwargs.get("bounds", [0.4, 0.3, 0.4, 0.4]))

    any_plotted = False
    for config in CONFIGS:
        steps, raw_curve, smoothed_curve = _load_config_curve(config, tag, subdir, step_is_timesteps, smoothing_weight)
        if len(steps) == 0:
            print(f"  [SKIP] {config['label']} -no data for {tag}")
            continue

        color = config["color"]
        # Plot thick smoothed line on top
        ax.plot(steps, smoothed_curve, label=config["label"], color=color, linewidth=2.0)

        if axins is not None:
            axins.plot(steps, smoothed_curve, color=color, linewidth=2.0)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"[SKIP] No data for {tag}; skipping figure {out_basename}")
        return

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Training Timesteps")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, TOTAL_TIMESTEPS)
    ax.legend(fontsize=9, loc="lower right")
    ax.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
    ax.xaxis.get_offset_text().set_fontsize(9)

    if axins is not None:
        axins.set_xlim(zoom_kwargs["xlim"])
        axins.set_ylim(zoom_kwargs["ylim"])
        axins.ticklabel_format(axis="x", style="scientific", scilimits=(0, 0))
        axins.tick_params(labelsize=8)
        ax.indicate_inset_zoom(axins, edgecolor="black")

    plt.tight_layout()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for ext in ("png", "pdf"):
        out_path = os.path.join(RESULTS_DIR, f"{out_basename}.{ext}")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved -> {out_path}")
    plt.close(fig)


def print_summary():
    print("\n" + "=" * 75)
    print("  Final-window averages (last 500 points, smoothed)")
    print("=" * 75)
    print(f"  {'Configuration':<30} {'Reward':>12} {'Compliance':>14}")
    print("  " + "-" * 60)
    for config in CONFIGS:
        _, _, rew_smooth = _load_config_curve(config, REWARD_TAG, REWARD_SUBDIR,
                                       REWARD_STEP_IS_TIMESTEPS, smoothing_weight=0.8)
        _, _, comp_smooth = _load_config_curve(config, COMPLIANCE_TAG, COMPLIANCE_SUBDIR,
                                        COMPLIANCE_STEP_IS_TIMESTEPS, smoothing_weight=0.95)
        rew_tail = rew_smooth[-500:].mean() if len(rew_smooth) else float("nan")
        comp_tail = comp_smooth[-500:].mean() if len(comp_smooth) else float("nan")
        print(f"  {config['label']:<30} {rew_tail:>12.2f} {comp_tail:>14.3f}")
    print("  " + "-" * 60)


def main():
    print("=" * 75)
    print("  Individual Governance -Training Curves")
    print("=" * 75)

    print("\nResolving log directories:")
    for config in CONFIGS:
        print(f"  {config['label']}:")
        for d in config["agent_dirs"]:
            status = "OK " if os.path.isdir(d) else "MISSING"
            print(f"    [{status}] {d}")

    print("\nPlotting reward curves...")
    plot_metric(
        tag=REWARD_TAG,
        subdir=REWARD_SUBDIR,
        step_is_timesteps=REWARD_STEP_IS_TIMESTEPS,
        ylabel="Mean Episode Reward",
        title="Individual Governance — Reward over Training",
        out_basename="individual_governance_reward_curves",
        smoothing_weight=0.0,
        zoom_kwargs={
            "bounds": [0.25, 0.25, 0.4, 0.4], # [x0, y0, width, height] relative to main axes
            "xlim": (1.5e6, 1.92e6),          # Zoom in on the final 420k steps
            "ylim": (0.5, 2.2)                # Focus on the tight reward convergence window
        }
    )

    print("\nPlotting compliance curves...")
    plot_metric(
        tag=COMPLIANCE_TAG,
        subdir=COMPLIANCE_SUBDIR,
        step_is_timesteps=COMPLIANCE_STEP_IS_TIMESTEPS,
        ylabel="Compliance Rate",
        title="Individual Governance — Compliance over Training",
        out_basename="individual_governance_compliance_curves",
        smoothing_weight=0.95,
        zoom_kwargs={
            "bounds": [0.25, 0.15, 0.4, 0.4], # [x0, y0, width, height] relative to main axes
            "xlim": (1.5e6, 1.92e6),          # Zoom in on the final 420k steps
            "ylim": (0.5, 1.0)                # Focus on the compliance convergence window
        }
    )

    print_summary()


if __name__ == "__main__":
    main()
