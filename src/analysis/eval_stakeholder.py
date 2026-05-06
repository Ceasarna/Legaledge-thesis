"""
Stakeholder Comparison: Pareto-Optimal vs Scalarized Baseline

Reads sweep_results.csv and compares stakeholder-specific Pareto-optimal
weight configurations against an equal-weight scalarized baseline.

Stakeholders:
  1. Budget EV Owner   — minimize net cost (best price point on Pareto front)
  2. SoC-Aware Owner   — minimize SoC error (best SoC point on Pareto front)
  3. Grid Operator      — maximize compliance (best compliance point on Pareto front)
  4. Balanced User      — closest to utopia across all 3 objectives (knee point)
  5. Scalarized Baseline — equal weights (0.33, 0.33, 0.33) or nearest

Outputs:
  - Console summary
  - LaTeX table for thesis
  - Bar chart comparing stakeholders
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sweep_results", "sweep_results.csv"
)
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "stakeholder_results"
)

def pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """Return boolean mask of non-dominated points. All objectives minimized."""
    n = len(objectives)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        for j in range(n):
            if i == j or not mask[j]:
                continue
            if np.all(objectives[j] <= objectives[i]) and np.any(objectives[j] < objectives[i]):
                mask[i] = False
                break
    return mask


def find_stakeholder_points(df: pd.DataFrame) -> dict:
    """Identify the best Pareto-optimal point for each stakeholder."""

    # Compute 3D Pareto front (minimize: net_cost, soc_error, non_compliance)
    objectives = np.column_stack([
        df["avg_net_cost"].values,
        df["avg_soc_error"].values,
        1.0 - df["avg_compliance"].values,
    ])
    pareto_mask = pareto_front_mask(objectives)
    pareto_df = df[pareto_mask].copy()
    pareto_obj = objectives[pareto_mask]

    # ── Scalarized baseline: closest to equal weights ──
    weight_dist = (
        (df["w_price"] - 1/3)**2 +
        (df["w_soc"] - 1/3)**2 +
        (df["w_compliance"] - 1/3)**2
    )
    baseline_idx = weight_dist.idxmin()
    baseline = df.loc[baseline_idx]

    # ── Budget EV Owner: lowest net cost on Pareto front ──
    budget_idx = pareto_df["avg_net_cost"].idxmin()
    budget = pareto_df.loc[budget_idx]

    # ── SoC-Aware Owner: lowest SoC error on Pareto front ──
    soc_idx = pareto_df["avg_soc_error"].idxmin()
    soc_aware = pareto_df.loc[soc_idx]

    # ── Grid Operator: highest compliance on Pareto front ──
    compliance_idx = pareto_df["avg_compliance"].idxmax()
    grid_op = pareto_df.loc[compliance_idx]

    # ── Balanced User: closest to utopia (normalized) on Pareto front ──
    obj_min = pareto_obj.min(axis=0)
    obj_max = pareto_obj.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0
    obj_norm = (pareto_obj - obj_min) / obj_range
    dist_to_utopia = np.linalg.norm(obj_norm, axis=1)
    balanced_local_idx = np.argmin(dist_to_utopia)
    balanced = pareto_df.iloc[balanced_local_idx]

    return {
        "Budget EV Owner": budget,
        "SoC-Aware Owner": soc_aware,
        "Grid Operator": grid_op,
        "Balanced User": balanced,
        "Scalarized Baseline": baseline,
    }


def compute_improvement(stakeholder_row, baseline_row):
    """Compute % improvement of stakeholder vs baseline for each metric."""
    improvements = {}

    # Net cost: lower is better
    if baseline_row["avg_net_cost"] != 0:
        improvements["net_cost"] = (
            (baseline_row["avg_net_cost"] - stakeholder_row["avg_net_cost"])
            / abs(baseline_row["avg_net_cost"]) * 100
        )
    else:
        improvements["net_cost"] = 0.0

    # SoC error: lower is better
    if baseline_row["avg_soc_error"] != 0:
        improvements["soc_error"] = (
            (baseline_row["avg_soc_error"] - stakeholder_row["avg_soc_error"])
            / baseline_row["avg_soc_error"] * 100
        )
    else:
        improvements["soc_error"] = 0.0

    # Compliance: higher is better
    if baseline_row["avg_compliance"] != 0:
        improvements["compliance"] = (
            (stakeholder_row["avg_compliance"] - baseline_row["avg_compliance"])
            / baseline_row["avg_compliance"] * 100
        )
    else:
        improvements["compliance"] = 0.0

    return improvements


def to_latex_table(points: dict) -> str:
    baseline = points["Scalarized Baseline"]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Stakeholder-specific Pareto-optimal configurations vs.\ scalarized baseline ($w = \frac{1}{3}$ each).}",
        r"  \label{tab:stakeholder-comparison}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l c c c c c c}",
        r"    \toprule",
        r"    \textbf{Stakeholder} & \textbf{$w_p$} & \textbf{$w_s$} & \textbf{$w_c$} "
        r"& \textbf{Net Cost (EUR)} & \textbf{SoC Error (\%)} & \textbf{Compliance} \\",
        r"    \midrule",
    ]

    for name, row in points.items():
        is_baseline = name == "Scalarized Baseline"

        cost_str = f"${row['avg_net_cost']:.3f} \\pm {row['std_net_cost']:.3f}$"
        soc_str = f"${row['avg_soc_error']:.1f} \\pm {row['std_soc_error']:.1f}$"
        comp_str = f"{row['avg_compliance']:.3f}"

        if not is_baseline:
            # Absolute differences
            d_cost = baseline["avg_net_cost"] - row["avg_net_cost"]  # positive = cheaper
            d_soc = baseline["avg_soc_error"] - row["avg_soc_error"]  # positive = better
            d_comp = row["avg_compliance"] - baseline["avg_compliance"]  # positive = better

            def delta_str(val, unit=""):
                if val > 0:
                    return f"\\textcolor{{green!60!black}}{{$\\downarrow${abs(val):.2f}{unit}}}"
                elif val < 0:
                    return f"\\textcolor{{red}}{{$\\uparrow${abs(val):.2f}{unit}}}"
                return ""

            def delta_str_higher(val, unit=""):
                if val > 0:
                    return f"\\textcolor{{green!60!black}}{{$\\uparrow${abs(val):.2f}{unit}}}"
                elif val < 0:
                    return f"\\textcolor{{red}}{{$\\downarrow${abs(val):.2f}{unit}}}"
                return ""

            cost_str += f" {delta_str(d_cost)}"
            soc_str += f" {delta_str(d_soc)}"
            comp_str += f" {delta_str_higher(d_comp)}"

        prefix = r"    \textit{" + name + "}" if is_baseline else f"    {name}"
        if is_baseline:
            w_str = r"$\approx$0.33 & $\approx$0.33 & $\approx$0.33"
        else:
            w_str = f"{row['w_price']:.2f} & {row['w_soc']:.2f} & {row['w_compliance']:.2f}"
        lines.append(
            f"{prefix} & {w_str} "
            f"& {cost_str} & {soc_str} & {comp_str} \\\\"
        )

        if is_baseline:
            lines.append(r"    \midrule")

    lines += [
        r"    \bottomrule",
        r"    \multicolumn{7}{l}{\footnotesize "
        r"$\downarrow$/$\uparrow$ = absolute change vs.\ baseline. "
        r"\textcolor{green!60!black}{Green} = improvement, \textcolor{red}{red} = degradation.} \\",
        r"  \end{tabular}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def plot_comparison(points: dict, out_dir: str):
    """Three separate subplots, one per metric, so scales don't clash."""
    order = ["Scalarized Baseline", "Budget EV Owner", "SoC-Aware Owner", "Grid Operator", "Balanced User"]
    names = [n for n in order if n in points]
    # Include weights in labels
    short_names = []
    for n in names:
        r = points[n]
        label = {
            "Scalarized Baseline": "Baseline",
            "Budget EV Owner": "Budget",
            "SoC-Aware Owner": "SoC-Aware",
            "Grid Operator": "Grid Op.",
            "Balanced User": "Balanced",
        }.get(n, n)
        short_names.append(f"{label}\n({r['w_price']:.1f}/{r['w_soc']:.1f}/{r['w_compliance']:.1f})")

    net_costs = [points[n]["avg_net_cost"] for n in names]
    soc_errors = [points[n]["avg_soc_error"] for n in names]
    compliances = [points[n]["avg_compliance"] * 100 for n in names]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = np.arange(len(names))

    # Highlight baseline in gray, others in color
    colors_cost = ["#9E9E9E"] + ["#2196F3"] * (len(names) - 1)
    colors_soc = ["#9E9E9E"] + ["#FF9800"] * (len(names) - 1)
    colors_comp = ["#9E9E9E"] + ["#4CAF50"] * (len(names) - 1)

    # Net Cost
    axes[0].bar(x, net_costs, color=colors_cost, alpha=0.85)
    axes[0].set_ylabel("Net Cost (EUR)")
    axes[0].set_title("Net Charging Cost")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short_names, rotation=25, ha="right")
    axes[0].grid(True, alpha=0.3, axis="y")

    # SoC Error
    axes[1].bar(x, soc_errors, color=colors_soc, alpha=0.85)
    axes[1].set_ylabel("SoC Error (%)")
    axes[1].set_title("SoC Target Error")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, rotation=25, ha="right")
    axes[1].grid(True, alpha=0.3, axis="y")

    # Compliance
    axes[2].bar(x, compliances, color=colors_comp, alpha=0.85)
    axes[2].set_ylabel("Compliance (%)")
    axes[2].set_title("Contract Compliance")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short_names, rotation=25, ha="right")
    axes[2].set_ylim(0, 105)
    axes[2].grid(True, alpha=0.3, axis="y")

    fig.suptitle("Stakeholder-Specific vs Scalarized Baseline\n"
                 r"Weights shown as ($w_{price}$ / $w_{soc}$ / $w_{compliance}$)",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stakeholder_comparison.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "stakeholder_comparison.pdf"), bbox_inches="tight")
    print(f"  Saved: {os.path.join(out_dir, 'stakeholder_comparison.png')}")
    plt.close()


def plot_radar(points: dict, out_dir: str):
    """Radar chart showing each stakeholder's profile across 3 normalized metrics."""
    names = ["Scalarized Baseline", "Budget EV Owner", "SoC-Aware Owner", "Grid Operator", "Balanced User"]
    names = [n for n in names if n in points]

    metrics = ["avg_net_cost", "avg_soc_error", "avg_compliance"]
    metric_labels = ["Low Cost", "Low SoC Error", "High Compliance"]

    # Normalize: all to [0, 1] where 1 = best
    all_vals = {m: [points[n][m] for n in names] for m in metrics}
    norm = {}
    for m in metrics:
        vals = np.array(all_vals[m])
        if m == "avg_compliance":
            # Higher is better
            norm[m] = (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)
        else:
            # Lower is better → invert
            norm[m] = 1.0 - (vals - vals.min()) / (vals.max() - vals.min() + 1e-8)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = ["#9E9E9E", "#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for i, name in enumerate(names):
        values = [norm[m][i] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=name, color=colors[i % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title("Stakeholder Profiles (Normalized)", fontsize=13, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "stakeholder_radar.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(out_dir, "stakeholder_radar.pdf"), bbox_inches="tight")
    print(f"  Saved: {os.path.join(out_dir, 'stakeholder_radar.png')}")
    plt.close()


def main():

    
    csv_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CSV
    if not os.path.exists(csv_path):
        print(f"CSV not found: {csv_path}")
        sys.exit(1)

    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} weight configurations from {csv_path}")

    points = find_stakeholder_points(df)
    baseline = points["Scalarized Baseline"]

    # ── Console summary ──
    print(f"\n{'='*80}")
    print("STAKEHOLDER COMPARISON")
    print(f"{'='*80}")
    print(f"{'Stakeholder':<22s} {'Weights':>18s} {'Net Cost':>12s} {'SoC Err':>10s} {'Compliance':>12s}")
    print("-" * 80)

    for name, row in points.items():
        w_str = f"({row['w_price']:.2f},{row['w_soc']:.2f},{row['w_compliance']:.2f})"
        print(
            f"{name:<22s} {w_str:>18s} "
            f"{row['avg_net_cost']:>10.3f}€ "
            f"{row['avg_soc_error']:>8.1f}% "
            f"{row['avg_compliance']:>10.1%}"
        )

    # ── Improvements vs baseline ──
    print(f"\n{'='*80}")
    print("IMPROVEMENT vs SCALARIZED BASELINE")
    print(f"{'='*80}")
    for name, row in points.items():
        if name == "Scalarized Baseline":
            continue
        imp = compute_improvement(row, baseline)
        print(
            f"  {name:<22s} "
            f"Cost: {imp['net_cost']:+.1f}%  "
            f"SoC: {imp['soc_error']:+.1f}%  "
            f"Compliance: {imp['compliance']:+.1f}%"
        )

    # ── LaTeX ──
    tex = to_latex_table(points)
    tex_path = os.path.join(OUT_DIR, "stakeholder_table.tex")
    with open(tex_path, "w") as f:
        f.write(tex + "\n")
    print(f"\nLaTeX table: {tex_path}")

    # ── Plots ──
    plot_comparison(points, OUT_DIR)
    plot_radar(points, OUT_DIR)

    # ── Export selected points ──
    rows = []
    for name, row in points.items():
        r = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        r["stakeholder"] = name
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "stakeholder_points.csv"), index=False)
    print(f"Points CSV: {os.path.join(OUT_DIR, 'stakeholder_points.csv')}")


if __name__ == "__main__":
    main()
