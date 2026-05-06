"""
Throughput & Overhead Analysis
==============================

Reads throughput_metrics.csv files from training runs (different agent counts)
and produces:
  1. LaTeX tables:
     - throughput_window_tps.tex   : headline TPS numbers (window-based)
     - throughput_summary.tex      : full per-agent breakdown with mean +/- std
     - throughput_breakdown.tex    : time percentages (RL / BC / FL)
  2. Plots (PNG + PDF):
     - throughput_window_tps.png   : BC + System TPS vs agents (window-based, thesis-ready)
     - throughput_tps_comparison.png : window vs legacy overlay (sanity-check)
     - throughput_breakdown.png    : stacked bar of RL / BC / FL %
     - tx_latency.png              : per-tx blockchain latency
  3. throughput_summary.csv        : raw grouped data

Usage:
    python -m pytorchexample.eval.eval_throughput

    By default, scans ./logs/benchmark/federated/contract/*/throughput_metrics.csv
    Override by passing paths as arguments:
        python -m pytorchexample.eval.eval_throughput path1.csv path2.csv
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = r"C:\Users\toek3476\FRL\Thesis_2\Thesis_code"
OUT_DIR = os.path.join(PROJECT_ROOT, "quickstart-pytorch", "logs", "eval_throughput")

# ── Locate CSV files ────────────────────────────────────────────────────
def find_csvs(args):
    if args:
        return args
    patterns = [
        os.path.join(PROJECT_ROOT, "quickstart-pytorch", "logs", "benchmark", "federated", "contract", "*", "throughput_metrics.csv"),
        os.path.join(PROJECT_ROOT, "quickstart-pytorch", "logs", "benchmark", "federated", "no_contract", "*", "throughput_metrics.csv"),
    ]
    found = []
    for p in patterns:
        found.extend(glob.glob(p))
    return found


def load_and_concat(paths):
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # Tag with source dir for identification
            df["source"] = os.path.basename(os.path.dirname(p))
            dfs.append(df)
        except Exception as e:
            print(f"  Skipping {p}: {e}")
    if not dfs:
        print("No valid CSV files found.")
        sys.exit(1)
    return pd.concat(dfs, ignore_index=True)


def summary_table(df):
    """Group by n_agents → mean ± std for key metrics, including TPS."""

    # Ensure aggregation_s exists (fallback to 0 if missing from older logs)
    if "aggregation_s" not in df.columns:
        df["aggregation_s"] = 0.0

    # Reconstruct the total Wall-Clock time per round
    if "wall_s" in df.columns:
        df["total_round_time"] = df["wall_s"]
    else:
        df["total_round_time"] = (
            df["rl_time_s"] +
            df["blockchain_time_s"] +
            df["fl_overhead_s"] +
            df["aggregation_s"]
        )

    # --- SAFE DIVISION FOR TPS ---
    # (Old "mean of per-agent time" approach — kept for back-compat with older runs.)
    safe_round_time = df["total_round_time"].replace(0, np.nan)
    df["system_tps"] = (df["total_episodes"] / safe_round_time).fillna(0.0)

    safe_bc_time = df["blockchain_time_s"].replace(0, np.nan)
    df["blockchain_tps"] = (df["total_episodes"] / safe_bc_time).fillna(0.0)

    # --- WINDOW-BASED TPS (preferred when present) ---
    # Computed server-side from per-agent Python wall-clock first/last tx stamps.
    # Doesn't rely on chain timestamps and is correct when agents run in parallel.
    if "blockchain_window_tps" not in df.columns:
        df["blockchain_window_tps"] = 0.0
    if "system_window_tps" not in df.columns:
        df["system_window_tps"] = 0.0
    if "blockchain_window_s" not in df.columns:
        df["blockchain_window_s"] = 0.0
    if "round_window_s" not in df.columns:
        df["round_window_s"] = 0.0
    if "total_blockchain_tx" not in df.columns:
        df["total_blockchain_tx"] = 0

    grouped = df.groupby("n_agents").agg(
        rounds=("round", "count"),
        mean_rl_time=("rl_time_s", "mean"),
        std_rl_time=("rl_time_s", "std"),
        mean_bc_time=("blockchain_time_s", "mean"),
        std_bc_time=("blockchain_time_s", "std"),
        mean_fl_overhead=("fl_overhead_s", "mean"),
        std_fl_overhead=("fl_overhead_s", "std"),
        mean_agg_time=("aggregation_s", "mean"),
        mean_tx_latency=("mean_tx_latency_ms", "mean"),
        std_tx_latency=("mean_tx_latency_ms", "std"),
        mean_rl_pct=("rl_pct", "mean"),
        mean_bc_pct=("blockchain_pct", "mean"),
        mean_episodes=("total_episodes", "mean"),
        
        # --- NEW METRICS FOR CSV EXPORT ---
        mean_wall_s=("total_round_time", "mean"),
        mean_system_tps=("system_tps", "mean"),
        std_system_tps=("system_tps", "std"),
        mean_blockchain_tps=("blockchain_tps", "mean"),
        std_blockchain_tps=("blockchain_tps", "std"),
        # Window-based TPS (preferred — uses true wall-clock window across agents)
        mean_blockchain_window_tps=("blockchain_window_tps", "mean"),
        std_blockchain_window_tps=("blockchain_window_tps", "std"),
        mean_system_window_tps=("system_window_tps", "mean"),
        std_system_window_tps=("system_window_tps", "std"),
        mean_blockchain_window_s=("blockchain_window_s", "mean"),
        mean_round_window_s=("round_window_s", "mean"),
        mean_total_blockchain_tx=("total_blockchain_tx", "mean"),
    ).reset_index()

    return grouped

def plot_tps_comparison(grouped, out_path):
    """Line chart: System TPS vs Blockchain TPS (legacy + window-based)."""
    agents = grouped["n_agents"].astype(int).values
    sys_tps = grouped["mean_system_tps"].values
    sys_tps_std = grouped["std_system_tps"].values
    bc_tps = grouped["mean_blockchain_tps"].values
    bc_tps_std = grouped["std_blockchain_tps"].values
    sys_win_tps = grouped["mean_system_window_tps"].values
    sys_win_tps_std = grouped["std_system_window_tps"].values
    bc_win_tps = grouped["mean_blockchain_window_tps"].values
    bc_win_tps_std = grouped["std_blockchain_window_tps"].values

    fig, ax = plt.subplots(figsize=(6, 4))

    # Legacy (mean-of-per-agent-time) — kept for back-compat
    ax.errorbar(agents, sys_tps, yerr=sys_tps_std, fmt='-o', color="#90CAF9",
                linewidth=1.5, capsize=4, alpha=0.7, label="System TPS (legacy mean)")
    ax.errorbar(agents, bc_tps, yerr=bc_tps_std, fmt='--s', color="#FFCC80",
                linewidth=1.5, capsize=4, alpha=0.7, label="Blockchain TPS (legacy mean)")

    # Window-based — preferred
    ax.errorbar(agents, sys_win_tps, yerr=sys_win_tps_std, fmt='-o', color="#2196F3",
                linewidth=2, capsize=4, label="System TPS (wall-clock window)")
    ax.errorbar(agents, bc_win_tps, yerr=bc_win_tps_std, fmt='--s', color="#FF9800",
                linewidth=2, capsize=4, label="Blockchain TPS (wall-clock window)")
    
    ax.set_xlabel("Number of Concurrent Agents")
    ax.set_ylabel("Throughput (Transactions / Second)")
    ax.set_title("Throughput Scaling: System vs. Blockchain")
    
    ax.set_xticks(agents)
    ax.set_xticklabels([str(a) for a in agents])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.png', '.pdf')}")
    plt.close()

def plot_window_tps(grouped, out_path):
    """Thesis-ready figure: window-based BC + System TPS vs agent count."""
    agents = grouped["n_agents"].astype(int).values
    sys_tps = grouped["mean_system_window_tps"].values
    sys_tps_std = grouped["std_system_window_tps"].values
    bc_tps = grouped["mean_blockchain_window_tps"].values
    bc_tps_std = grouped["std_blockchain_window_tps"].values

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(agents, bc_tps, yerr=bc_tps_std, fmt='-s', color="#FF9800",
                linewidth=2, capsize=4, label="Blockchain TPS")
    ax.errorbar(agents, sys_tps, yerr=sys_tps_std, fmt='--o', color="#2196F3",
                linewidth=2, capsize=4, label="System TPS (end-to-end)")

    ax.set_xlabel("Number of Concurrent Agents")
    ax.set_ylabel("Throughput (Transactions / Second)")
    ax.set_title("Throughput Scaling (wall-clock window)")
    ax.set_xticks(agents)
    ax.set_xticklabels([str(a) for a in agents])
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.png', '.pdf')}")
    plt.close()


def to_latex_window_tps(grouped):
    """Thesis-ready headline table: window-based throughput per agent count."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{End-to-end and blockchain throughput versus number of federated agents, "
        r"measured over the wall-clock window during which at least one agent was interacting with the chain.}",
        r"  \label{tab:throughput-window}",
        r"  \begin{tabular}{r r r r r r}",
        r"    \toprule",
        r"    \textbf{Agents} & \textbf{Total Tx} & \textbf{BC Window (s)} "
        r"& \textbf{BC TPS} & \textbf{System TPS} & \textbf{Tx Latency (ms)} \\",
        r"    \midrule",
    ]
    for _, r in grouped.iterrows():
        lines.append(
            f"    {int(r['n_agents'])} "
            f"& {r['mean_total_blockchain_tx']:.0f} "
            f"& {r['mean_blockchain_window_s']:.2f} "
            f"& ${r['mean_blockchain_window_tps']:.2f} \\pm {r['std_blockchain_window_tps']:.2f}$ "
            f"& ${r['mean_system_window_tps']:.2f} \\pm {r['std_system_window_tps']:.2f}$ "
            f"& ${r['mean_tx_latency']:.1f} \\pm {r['std_tx_latency']:.1f}$ \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_latex_summary(grouped):
    """Full breakdown: per-agent RL / BC / FL times + window TPS."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Blockchain throughput and overhead by number of federated agents.}",
        r"  \label{tab:throughput-summary}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{r c c c c c c}",
        r"    \toprule",
        r"    \textbf{Agents} & \textbf{RL Time (s)} & \textbf{BC Time (s)} "
        r"& \textbf{FL Overhead (s)} & \textbf{Tx Latency (ms)} "
        r"& \textbf{BC TPS (window)} & \textbf{Episodes/Round} \\",
        r"    \midrule",
    ]
    for _, r in grouped.iterrows():
        lines.append(
            f"    {int(r['n_agents'])} "
            f"& ${r['mean_rl_time']:.2f} \\pm {r['std_rl_time']:.2f}$ "
            f"& ${r['mean_bc_time']:.2f} \\pm {r['std_bc_time']:.2f}$ "
            f"& ${r['mean_fl_overhead']:.2f} \\pm {r['std_fl_overhead']:.2f}$ "
            f"& ${r['mean_tx_latency']:.1f} \\pm {r['std_tx_latency']:.1f}$ "
            f"& ${r['mean_blockchain_window_tps']:.2f} \\pm {r['std_blockchain_window_tps']:.2f}$ "
            f"& {r['mean_episodes']:.0f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def to_latex_breakdown(grouped):
    """Table 2: Percentage breakdown of round time."""
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Round time breakdown by component (\%).}",
        r"  \label{tab:throughput-breakdown}",
        r"  \begin{tabular}{r c c c}",
        r"    \toprule",
        r"    \textbf{Agents} & \textbf{RL (\%)} & \textbf{Blockchain (\%)} & \textbf{FL Overhead (\%)} \\",
        r"    \midrule",
    ]
    for _, r in grouped.iterrows():
        fl_pct = 100.0 - r["mean_rl_pct"] - r["mean_bc_pct"]
        lines.append(
            f"    {int(r['n_agents'])} "
            f"& {r['mean_rl_pct']:.1f} "
            f"& {r['mean_bc_pct']:.1f} "
            f"& {fl_pct:.1f} \\\\"
        )
    lines += [
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def plot_breakdown(grouped, out_path):
    """Stacked bar chart: RL vs Blockchain vs FL overhead."""
    agents = grouped["n_agents"].astype(int).values
    rl_pct = grouped["mean_rl_pct"].values
    bc_pct = grouped["mean_bc_pct"].values
    fl_pct = 100.0 - rl_pct - bc_pct

    x = np.arange(len(agents))
    width = 0.5

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x, rl_pct, width, label="RL Training", color="#2196F3")
    ax.bar(x, bc_pct, width, bottom=rl_pct, label="Blockchain", color="#FF9800")
    ax.bar(x, fl_pct, width, bottom=rl_pct + bc_pct, label="FL Overhead", color="#4CAF50")

    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Percentage of Round Time (%)")
    ax.set_title("Round Time Breakdown by Component")
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.png', '.pdf')}")
    plt.close()


def plot_tx_latency(grouped, out_path):
    """Bar chart: per-tx latency vs agent count."""
    agents = grouped["n_agents"].astype(int).values
    latency = grouped["mean_tx_latency"].values
    latency_std = grouped["std_tx_latency"].values

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(np.arange(len(agents)), latency, yerr=latency_std,
           capsize=4, color="#FF9800", alpha=0.85)
    ax.set_xlabel("Number of Agents")
    ax.set_ylabel("Mean Tx Latency (ms)")
    ax.set_title("Per-Transaction Blockchain Latency")
    ax.set_xticks(np.arange(len(agents)))
    ax.set_xticklabels(agents)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(out_path.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"  Saved: {out_path}")
    print(f"  Saved: {out_path.replace('.png', '.pdf')}")
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    csv_paths = find_csvs(sys.argv[1:])
    print(f"Found {len(csv_paths)} CSV file(s):")
    for p in csv_paths:
        print(f"  {p}")

    df = load_and_concat(csv_paths)
    print(f"\nLoaded {len(df)} rows total")
    print(f"Agent counts present: {sorted(df['n_agents'].unique())}")

    # Warmup filter — only drop the first round per (source, n_agents) group,
    # and only if that group has more than 2 rounds. For 1-round probes (short
    # scaling sweeps) this is a no-op; for long FL runs it trims the noisy first
    # round where Hardhat contract deploy / account warm-up skews the numbers.
    rounds_per_group = df.groupby(["source", "n_agents"])["round"].transform("count")
    filterable = rounds_per_group > 2
    df = df[(~filterable) | (df["round"] > 1)].copy()
    print(f"After warmup filter: {len(df)} rows")

    if len(df) == 0:
        print("Nothing to analyse — all rows filtered out. Exiting.")
        sys.exit(1)

    grouped = summary_table(df).sort_values("n_agents").reset_index(drop=True)

    # ── Console summary ──
    print(f"\n{'='*120}")
    print("THROUGHPUT SUMMARY  (legacy = mean-of-per-agent-time, window = true concurrent wall-clock window)")
    print(f"{'='*120}")
    for _, r in grouped.iterrows():
        print(
            f"  {int(r['n_agents']):3d} agents | "
            f"Wall: {r['mean_wall_s']:6.2f}s | "
            f"Tx Lat: {r['mean_tx_latency']:5.1f}ms || "
            f"legacy Sys TPS: {r['mean_system_tps']:6.2f} | "
            f"legacy BC TPS: {r['mean_blockchain_tps']:6.2f} || "
            f"window Sys TPS: {r['mean_system_window_tps']:6.2f} | "
            f"window BC TPS: {r['mean_blockchain_window_tps']:6.2f} "
            f"({r['mean_total_blockchain_tx']:.0f} tx / {r['mean_blockchain_window_s']:.2f}s)"
        )

    # ── LaTeX tables ──
    latex_outputs = {
        "throughput_window_tps.tex": to_latex_window_tps(grouped),
        "throughput_summary.tex":    to_latex_summary(grouped),
        "throughput_breakdown.tex":  to_latex_breakdown(grouped),
    }
    for fname, content in latex_outputs.items():
        path = os.path.join(OUT_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
        print(f"  Saved: {path}")

    # ── Plots ──
    plot_window_tps(grouped,     os.path.join(OUT_DIR, "throughput_window_tps.png"))
    plot_tps_comparison(grouped, os.path.join(OUT_DIR, "throughput_tps_comparison.png"))
    plot_breakdown(grouped,      os.path.join(OUT_DIR, "throughput_breakdown.png"))
    plot_tx_latency(grouped,     os.path.join(OUT_DIR, "tx_latency.png"))

    # ── Raw data export for further analysis ──
    grouped.to_csv(os.path.join(OUT_DIR, "throughput_summary.csv"), index=False)
    print(f"  Saved: {os.path.join(OUT_DIR, 'throughput_summary.csv')}")


if __name__ == "__main__":
    main()
    