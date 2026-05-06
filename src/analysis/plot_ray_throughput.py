"""
Plot standalone Ray blockchain throughput sweep results.

By default this scans the current working directory for:
    ray_throughput_*_agents.csv

Usage:
    python -m src.analysis.plot_ray_throughput
    python -m src.analysis.plot_ray_throughput --input-dir .
"""

import argparse
import glob
import os
import re
from typing import List

try:
    import matplotlib.pyplot as plt
    import pandas as pd
except ImportError as exc:
    raise SystemExit(
        "Missing plotting dependency. Install with:\n"
        "  pip install matplotlib pandas\n"
        f"Original import error: {exc}"
    ) from exc


AGENT_COUNT_RE = re.compile(r"_(\d+)_agents\.csv$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot throughput from Ray standalone benchmark CSV files."
    )
    parser.add_argument("--input-dir", default=".")
    parser.add_argument("--pattern", default="ray_throughput_*_agents.csv")
    parser.add_argument("--output", default="ray_throughput_scaling.png")
    parser.add_argument("--summary-csv", default="ray_throughput_summary.csv")
    return parser.parse_args()


def infer_agent_count(path: str, df: pd.DataFrame) -> int:
    match = AGENT_COUNT_RE.search(os.path.basename(path))
    if match:
        return int(match.group(1))
    if "n_agents" in df.columns:
        return int(df["n_agents"].iloc[0])
    return int(df["agent_id"].nunique())


def safe_window_s(start_values: pd.Series, end_values: pd.Series) -> float:
    starts = start_values[start_values > 0]
    ends = end_values[end_values > 0]
    if starts.empty or ends.empty:
        return 0.0
    return float(ends.max() - starts.min())


def summarize_file(path: str) -> dict:
    df = pd.read_csv(path)
    n_agents = infer_agent_count(path, df)
    total_tx = int(df["n_blockchain_tx"].sum())

    round_window_s = safe_window_s(df["round_wall_start"], df["round_wall_end"])
    tx_window_s = safe_window_s(df["first_tx_wall_s"], df["last_tx_wall_s"])
    if tx_window_s <= 0:
        tx_window_s = round_window_s

    throughput_tps = total_tx / tx_window_s if tx_window_s > 0 else 0.0
    round_tps = total_tx / round_window_s if round_window_s > 0 else 0.0

    durations = df["round_wall_end"] - df["round_wall_start"]
    first_tx_span = safe_window_s(df["first_tx_wall_s"], df["first_tx_wall_s"])
    last_tx_span = safe_window_s(df["last_tx_wall_s"], df["last_tx_wall_s"])

    row = {
        "file": os.path.basename(path),
        "n_agents": n_agents,
        "rows": len(df),
        "successful_tx": total_tx,
        "tx_window_s": tx_window_s,
        "round_window_s": round_window_s,
        "throughput_tps": throughput_tps,
        "round_tps": round_tps,
        "mean_agent_duration_s": float(durations.mean()),
        "p95_agent_duration_s": float(durations.quantile(0.95)),
        "first_tx_span_s": first_tx_span,
        "last_tx_span_s": last_tx_span,
        "unique_pids": int(df["pid"].nunique()),
    }

    if "mean_tx_latency_ms" in df.columns:
        successful = df[df["n_blockchain_tx"] > 0]
        row.update(
            {
                "mean_tx_latency_ms": float(successful["mean_tx_latency_ms"].mean()),
                "p50_tx_latency_ms": float(successful["p50_tx_latency_ms"].median()),
                "p95_tx_latency_ms": float(successful["p95_tx_latency_ms"].quantile(0.95)),
                "max_tx_latency_ms": float(successful["max_tx_latency_ms"].max()),
            }
        )

    if "mean_gas_used" in df.columns:
        successful = df[df["n_blockchain_tx"] > 0]
        row.update(
            {
                "mean_gas_used": float(successful["mean_gas_used"].mean()),
                "total_gas_used": float(df["total_gas_used"].sum()),
            }
        )

    return row


def load_summaries(input_dir: str, pattern: str) -> pd.DataFrame:
    paths: List[str] = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No CSV files matched {os.path.join(input_dir, pattern)}"
        )

    rows = [summarize_file(path) for path in paths]
    return pd.DataFrame(rows).sort_values("n_agents")


def plot_throughput(summary: pd.DataFrame, output: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.plot(
        summary["n_agents"],
        summary["throughput_tps"],
        marker="o",
        linewidth=2,
        color="#1565C0",
        label="Blockchain throughput",
    )

    ax.set_xscale("log", base=2)
    ax.set_xticks(summary["n_agents"])
    ax.set_xticklabels([str(int(x)) for x in summary["n_agents"]])
    ax.set_xlabel("Concurrent Ray agents")
    ax.set_ylabel("Successful transactions / second")
    ax.set_title("Throughput")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(loc="best")

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    pdf_output = os.path.splitext(output)[0] + ".pdf"
    plt.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output}")
    print(f"Saved plot: {pdf_output}")


def plot_optional_metric(
    summary: pd.DataFrame,
    column: str,
    ylabel: str,
    title: str,
    output: str,
) -> None:
    if column not in summary.columns:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        summary["n_agents"],
        summary[column],
        marker="o",
        linewidth=2,
        color="#2E7D32",
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks(summary["n_agents"])
    ax.set_xticklabels([str(int(x)) for x in summary["n_agents"]])
    ax.set_xlabel("Concurrent Ray agents")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.45)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    pdf_output = os.path.splitext(output)[0] + ".pdf"
    plt.savefig(pdf_output, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output}")
    print(f"Saved plot: {pdf_output}")


def main() -> None:
    args = parse_args()
    summary = load_summaries(args.input_dir, args.pattern)
    summary.to_csv(args.summary_csv, index=False)
    print(summary[["n_agents", "successful_tx", "tx_window_s", "throughput_tps"]])
    print(f"Saved summary: {args.summary_csv}")
    plot_throughput(summary, args.output)
    base, _ = os.path.splitext(args.output)
    plot_optional_metric(
        summary,
        "p95_tx_latency_ms",
        "p95 latency (ms)",
        "Latency",
        f"{base}_latency.png",
    )
    plot_optional_metric(
        summary,
        "mean_gas_used",
        "mean gas / transaction",
        "Gas",
        f"{base}_gas.png",
    )


if __name__ == "__main__":
    main()
