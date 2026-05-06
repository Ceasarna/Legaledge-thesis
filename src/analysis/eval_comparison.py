"""
Comparison 1: Does contract awareness improve compliance without destroying performance?
=======================================================================================

Evaluates 3 pairs of models (no-contract vs contract) on the same neutral environment.
Measures: reward, final SoC, charging cost, compliance rate, peak/off-peak energy.

Pairs:
  1. Local RL (single EV): Tesla Model 3 LR only
  2. Centralized RL (all EVs): all 4 EV models mixed
  3. FRL (4 clients, 1 EV each): shared model via FedAvg
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.stats as stats
from stable_baselines3 import PPO
import sys

# ── Adjust this to your project root if needed ──
PROJECT_ROOT = r"C:\Users\toek3476\FRL\Thesis_2\Thesis_code"
sys.path.insert(0, os.path.join(PROJECT_ROOT, "quickstart-pytorch"))

from pytorchexample.charging_env import SmartChargingEnv

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SEED = 1
NUM_EPISODES = 1000
EVAL_SEEDS = [1, 2, 3, 42, 100, 202, 303, 404, 505, 606]

METRIC_KEYS = [
    "rewards", "final_socs", "soc_deficits",
    "charge_costs", "discharge_revenues", "net_costs",
    "peak_kwhs", "offpeak_kwhs", "compliances", "total_energies",
]

RESULTS_DIR = os.path.join(
    PROJECT_ROOT, "quickstart-pytorch", "pytorchexample",
    "eval", "individual_governance_results",
)

PPO_KWARGS = dict(
    verbose=0,
    learning_rate=3e-4,
    batch_size=240,
    n_steps=4800,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    device="cpu",
)

# Model pairs: (name, no_contract_path, contract_path)
# Filename conventions produced by testing.py training functions:
#   _oneEV: Baseline_RL_{no_contract|contract}_non_IID_mimic_FRL_{model_name}_{agent_id}.pt
#   multi-EV: Baseline_RL_{no_contract|contract}_non_IID_mimic_FRL_{agent_id}.pt
# agent_id convention (420_*):
#   420_0_0..3   -> Local RL (oneEV), no contract, one slot per EV model
#   420_0_10..13 -> Local RL (oneEV), with contract, one slot per EV model
#   420_1        -> Centralized RL (all 4 EVs), no contract
#   420_2        -> Centralized RL (all 4 EVs), with contract
# The Individual Governance comparison uses Volvo XC90 PHEV at i=0 slot of
# the oneEV loop (so agent_ids 42000 / 420010).
MODEL_PAIRS = [
    {
        "name": "Local RL (Volvo)",
        "no_contract": os.path.join(PROJECT_ROOT,
                                     "Baseline_RL_no_contract_non_IID_mimic_FRL_Volvo XC90 PHEV_42000.pt"),
        "contract":    os.path.join(PROJECT_ROOT,
                                     "Baseline_RL_contract_non_IID_mimic_FRL_Volvo XC90 PHEV_420010.pt"),
    },
    {
        "name": "Centralized RL (All EVs)",
        "no_contract": os.path.join(PROJECT_ROOT,
                                     "Baseline_RL_no_contract_non_IID_mimic_FRL_4201.pt"),
        "contract":    os.path.join(PROJECT_ROOT,
                                     "Baseline_RL_contract_non_IID_mimic_FRL_4202.pt"),
    },
    {
        "name": "FRL (4 clients)",
        "no_contract": os.path.join(PROJECT_ROOT, "quickstart-pytorch",
                                     "final_model_not_contract_aware_04_16_1.pt"),
        "contract":    os.path.join(PROJECT_ROOT, "quickstart-pytorch",
                                     "final_model_single_contract_aware_04_16_1.pt"),
    },
]


def verify_model_pairs():
    """Print each resolved path and whether it exists on disk."""
    print("\nResolving MODEL_PAIRS:")
    print("-" * 70)
    all_ok = True
    for pair in MODEL_PAIRS:
        print(f"  {pair['name']}")
        for variant in ("no_contract", "contract"):
            path = pair[variant]
            exists = os.path.exists(path)
            status = "OK " if exists else "MISSING"
            print(f"    [{status}] {variant:<12} → {path}")
            if not exists:
                all_ok = False
    print("-" * 70)
    if not all_ok:
        print("WARNING: Some model files are missing; they will be skipped in evaluation.")
    else:
        print("All model files resolved successfully.")
    return all_ok

# ═══════════════════════════════════════════════════════════════════════════
# Evaluation function
# ═══════════════════════════════════════════════════════════════════════════

def make_eval_env(model_name, seed=None):
    return SmartChargingEnv(
        rl_model="ppo",
        data_source=2024,
        model_name=model_name,
        random_duration=False,
        random_arrival=True,
        seed=seed,
    )


def load_model(model_path, env):
    """Load a trained PPO model from a .pt state dict."""
    model = PPO("MlpPolicy", env, **PPO_KWARGS)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(state_dict, strict=True)
    model.policy.eval()
    return model


def evaluate(model, env, num_episodes, seed=42):
    rng = np.random.RandomState(seed)

    rewards, final_socs, soc_deficits = [], [], []
    charge_costs, discharge_revenues, net_costs = [], [], []
    peak_kwhs, offpeak_kwhs = [], []
    compliances, total_energies = [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        done = False
        ep_reward = 0.0
        ep_charge_cost = 0.0
        ep_discharge_revenue = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            energy = info["actual_energy_kwh"]
            price = info["price_eur_mwh"]

            if energy > 0:
                ep_charge_cost += energy * price / 1000
            elif energy < 0:
                ep_discharge_revenue += abs(energy) * price / 1000

            done = terminated or truncated

        power_log = info.get("power_log", {})
        peak = sum(v for h, v in power_log.items() if 17 <= h <= 20 and v > 0)
        offpeak = sum(v for h, v in power_log.items() if not (17 <= h <= 20) and v > 0)
        final_soc = info.get("final_soc", 0.0)
        target_soc = info.get("target_soc", 80.0)

        rewards.append(ep_reward)
        final_socs.append(final_soc)
        soc_deficits.append(max(0, target_soc - final_soc))
        charge_costs.append(ep_charge_cost)
        discharge_revenues.append(ep_discharge_revenue)
        net_costs.append(ep_charge_cost - ep_discharge_revenue)
        peak_kwhs.append(peak)
        offpeak_kwhs.append(offpeak)
        compliances.append(float(peak == 0.0))
        total_energies.append(sum(power_log.values()))

    return {
        "rewards":              np.array(rewards),
        "final_socs":           np.array(final_socs),
        "soc_deficits":         np.array(soc_deficits),
        "charge_costs":         np.array(charge_costs),
        "discharge_revenues":   np.array(discharge_revenues),
        "net_costs":            np.array(net_costs),
        "peak_kwhs":            np.array(peak_kwhs),
        "offpeak_kwhs":         np.array(offpeak_kwhs),
        "compliances":          np.array(compliances),
        "total_energies":       np.array(total_energies),
    }

def evaluate_multi_seed(model, env, num_episodes, seeds):
    """Run evaluate() once per seed; return dict[metric] -> 1-D array of per-seed means.

    Each value in the returned dict is a length-len(seeds) vector where entry i
    is the mean of that metric over the num_episodes evaluated under seed[i].
    Downstream code uses .mean()/.std() across this vector to get across-seed
    statistics.
    """
    per_seed = {k: [] for k in METRIC_KEYS}
    for s in seeds:
        r = evaluate(model, env, num_episodes, seed=s)
        for k in METRIC_KEYS:
            per_seed[k].append(float(np.mean(r[k])))
    return {k: np.asarray(v) for k, v in per_seed.items()}


def _sig_stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def summarize(results, label=""):
    """Print per-seed mean ± std across seeds.

    `results` here is produced by evaluate_multi_seed, so each metric is a
    length-n_seeds vector of per-seed means. .mean() is the across-seed mean
    and .std() is the across-seed std.
    """
    r = results
    print(f"\n  {label}")
    print(f"  {'─' * 50}")
    print(f"  Avg Reward:          {r['rewards'].mean():>8.2f}  (±{r['rewards'].std():.2f})")
    print(f"  Avg Final SoC:       {r['final_socs'].mean():>8.1f}%  (±{r['final_socs'].std():.1f})")
    print(f"  Avg SoC Deficit:     {r['soc_deficits'].mean():>8.1f}%  (±{r['soc_deficits'].std():.1f})")
    print(f"  Avg Charge Cost:     {r['charge_costs'].mean():>8.4f}  (±{r['charge_costs'].std():.4f}) EUR")
    print(f"  Avg V2G Revenue:     {r['discharge_revenues'].mean():>8.4f}  (±{r['discharge_revenues'].std():.4f}) EUR")
    print(f"  Avg Net Cost:        {r['net_costs'].mean():>8.4f}  (±{r['net_costs'].std():.4f}) EUR")
    print(f"  Compliance Rate:     {r['compliances'].mean()*100:>8.1f}%  (±{r['compliances'].std()*100:.1f})")
    print(f"  Avg Peak kWh:        {r['peak_kwhs'].mean():>8.3f}  (±{r['peak_kwhs'].std():.3f})")
    print(f"  Avg Off-peak kWh:    {r['offpeak_kwhs'].mean():>8.3f}  (±{r['offpeak_kwhs'].std():.3f})")
    print(f"  Avg Total Energy:    {r['total_energies'].mean():>8.2f}  (±{r['total_energies'].std():.2f}) kWh")


# ═══════════════════════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════════════════════

def plot_comparison(all_results, save_path="comparison_contract_awareness.png", title = "comparison"):
    """
    Bar chart comparing no-contract vs contract for each pair.
    Metrics: Reward, Final SoC, Cost, Compliance, Peak kWh.
    """

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(f"{title}", fontsize=14, fontweight="bold")
    
    metrics = [
        ("Avg Reward",        "rewards",              "mean"),
        ("Final SoC (%)",     "final_socs",           "mean"),
        ("Compliance (%)",    "compliances",           "mean_pct"),
        ("Peak kWh",          "peak_kwhs",             "mean"),
        ("Charge Cost (EUR)", "charge_costs",          "mean"),
        ("V2G Revenue (EUR)", "discharge_revenues",    "mean"),
        ("Net Cost (EUR)",    "net_costs",             "mean"),
    ]
    axes_flat = axes.flatten()
    axes_flat[7].set_visible(False)     

    pair_names = [p["name"] for p in MODEL_PAIRS]
    n_pairs = len(pair_names)
    n_metrics = len(metrics)

    x = np.arange(n_pairs)
    width = 0.35

    for ax, (title, key, agg) in zip(axes_flat, metrics):
        no_contract_vals = []
        contract_vals = []

        for pair_name in pair_names:
            nc = all_results[f"{pair_name} (No Contract)"]
            sc = all_results[f"{pair_name} (Contract)"]

            if agg == "mean_pct":
                no_contract_vals.append(nc[key].mean() * 100)
                contract_vals.append(sc[key].mean() * 100)
            else:
                no_contract_vals.append(nc[key].mean())
                contract_vals.append(sc[key].mean())

        bars1 = ax.bar(x - width/2, no_contract_vals, width, label="No Contract",
                       color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width/2, contract_vals, width, label="Contract",
                       color="tomato", alpha=0.8)

        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace(" (", "\n(") for n in pair_names],
                           fontsize=7, ha="center")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar in bars1:
            ax.annotate(f"{bar.get_height():.2f}",
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=6)
        for bar in bars2:
            ax.annotate(f"{bar.get_height():.2f}",
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha="center", fontsize=6)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot → {save_path}")
    plt.show()


def write_six_config_table(all_results, tex_path, env_name, n_seeds, n_episodes):
    """Write the six-configuration LaTeX table using across-seed mean ± std.

    Each metric cell reports the mean of the ten per-seed means, with ± the
    standard deviation across those ten per-seed means.
    """
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Six-configuration comparison on \\textit{{{env_name}}}. "
        f"Each cell reports mean $\\pm$ std across {n_seeds} seeds "
        f"($n = {n_episodes}$ episodes per seed).}}",
        r"  \label{tab:six-configuration}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l c c c c c c c}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Reward} & \textbf{Final SoC (\%)} "
        r"& \textbf{Charge (EUR)} & \textbf{V2G (EUR)} & \textbf{Net Cost (EUR)} "
        r"& \textbf{Compliance (\%)} & \textbf{Peak (kWh)} \\",
        r"    \midrule",
    ]

    for name, r in all_results.items():
        rew = r["rewards"]
        soc = r["final_socs"]
        ch  = r["charge_costs"]
        v2g = r["discharge_revenues"]
        net = r["net_costs"]
        comp = r["compliances"] * 100.0
        pk  = r["peak_kwhs"]
        lines.append(
            f"    {name} "
            f"& ${rew.mean():.2f} \\pm {rew.std():.2f}$ "
            f"& ${soc.mean():.1f} \\pm {soc.std():.1f}$ "
            f"& ${ch.mean():.3f} \\pm {ch.std():.3f}$ "
            f"& ${v2g.mean():.3f} \\pm {v2g.std():.3f}$ "
            f"& ${net.mean():.3f} \\pm {net.std():.3f}$ "
            f"& ${comp.mean():.1f} \\pm {comp.std():.1f}$ "
            f"& ${pk.mean():.3f} \\pm {pk.std():.3f}$ \\\\"
        )

    lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]

    # Console echo
    print("\n" + "=" * 100)
    print(f"Six-configuration LaTeX Table — {env_name}  (mean ± std across {n_seeds} seeds)")
    print("=" * 100)
    print("\n".join(lines))

    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nWrote six-configuration table → {tex_path}")


def write_welch_ttest(all_results, tex_path, env_name, n_seeds,
                       pair_a_key="FRL (4 clients) (Contract)",
                       pair_b_key="FRL (4 clients) (No Contract)"):
    """Welch's t-test on FRL Contract-Aware vs Baseline using the ten-seed vectors.

    For each metric, the test is run on the length-n_seeds arrays of per-seed
    means (so n = n_seeds per group, not n_seeds × n_episodes).
    """
    if pair_a_key not in all_results or pair_b_key not in all_results:
        print(f"[Welch] Skipping — required pair missing "
              f"(have: {list(all_results.keys())})")
        return

    metrics_to_test = [
        ("Compliance (\\%)", "compliances", True),
        ("Final SoC (\\%)",  "final_socs",  False),
        ("Net Cost (EUR)",   "net_costs",   False),
        ("Reward",           "rewards",     False),
        ("Peak (kWh)",       "peak_kwhs",   False),
    ]

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        f"  \\caption{{Welch's $t$-test: Contract-Aware vs.\\ Baseline "
        f"(FRL, 4 clients) on \\textit{{{env_name}}} "
        f"($n = {n_seeds}$ per-seed means per group).}}",
        r"  \label{tab:welch-t-test-frl}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l c c c c c}",
        r"    \toprule",
        r"    \textbf{Metric} & \textbf{Contract-Aware} & \textbf{Baseline} "
        r"& \textbf{$t$} & \textbf{$p$} & \textbf{Sig.} \\",
        r"    \midrule",
    ]

    console_rows = []
    for label, key, as_pct in metrics_to_test:
        a = all_results[pair_a_key][key]
        b = all_results[pair_b_key][key]
        if as_pct:
            a = a * 100.0
            b = b * 100.0

        t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
        stars = _sig_stars(p_val)

        fmt_a = f"${np.mean(a):.3f} \\pm {np.std(a):.3f}$"
        fmt_b = f"${np.mean(b):.3f} \\pm {np.std(b):.3f}$"
        p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "$< 0.0001$"
        lines.append(
            f"    {label} & {fmt_a} & {fmt_b} "
            f"& ${t_stat:.2f}$ & {p_str} & {stars} \\\\"
        )
        console_rows.append((label, t_stat, p_val, stars))

    lines += [
        r"    \bottomrule",
        r"    \multicolumn{6}{l}{\footnotesize "
        r"${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$ (Welch's $t$-test, two-tailed)} \\",
        r"  \end{tabular}}",
        r"\end{table}",
    ]

    os.makedirs(os.path.dirname(tex_path), exist_ok=True)
    with open(tex_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  Welch's t-test — {pair_a_key} vs {pair_b_key} on {env_name}:")
    for label, t_stat, p_val, stars in console_rows:
        # Strip LaTeX \% for console
        plain = label.replace("\\%", "%")
        print(f"    {plain:20s}  t={t_stat:+.2f}  p={p_val:.4f} {stars}")
    print(f"  Wrote Welch's t-test table → {tex_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def _env_slug(env_name: str) -> str:
    return (env_name.lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""))


# The canonical environment whose tables land at the unsuffixed paths
# requested in Task 5/6 (welch_ttest.tex, six_configuration_table.tex).
CANONICAL_ENV_NAME = "All 4 EVs"


def main():
    print("=" * 70)
    print("  Comparison 1: Contract Awareness vs Performance")
    print(f"  Episodes per seed: {NUM_EPISODES}  |  Seeds: {EVAL_SEEDS}")
    print(f"  Total episodes per configuration: {NUM_EPISODES * len(EVAL_SEEDS)}")
    print("=" * 70)

    verify_model_pairs()

    EVAL_ENVS = {
        "Volvo Only": "Volvo XC90 PHEV",
        "All 4 EVs": ["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"],
        "Unseen EV (Model S)": "Tesla Model S",
    }

    for env_name, model_name in EVAL_ENVS.items():
        print(f"\n{'='*70}")
        print(f"  Environment: {env_name}")
        print(f"{'='*70}")

        env = make_eval_env(model_name=model_name, seed=SEED)
        all_results = {}  # key -> dict[metric] -> per-seed-mean vector (len = n_seeds)

        for pair in MODEL_PAIRS:
            name = pair["name"]

            if os.path.exists(pair["no_contract"]):
                print(f"\n>>> Evaluating: {name} (No Contract) on {env_name}  "
                      f"[{len(EVAL_SEEDS)} seeds × {NUM_EPISODES} episodes]")
                model_nc = load_model(pair["no_contract"], env)
                results_nc = evaluate_multi_seed(model_nc, env, NUM_EPISODES, EVAL_SEEDS)
                all_results[f"{name} (No Contract)"] = results_nc
                summarize(results_nc, f"{name} — No Contract — {env_name}")
            else:
                print(f"\n[SKIP] {name} (no_contract): file not found at\n  {pair['no_contract']}")

            if os.path.exists(pair["contract"]):
                print(f"\n>>> Evaluating: {name} (Contract) on {env_name}  "
                      f"[{len(EVAL_SEEDS)} seeds × {NUM_EPISODES} episodes]")
                model_sc = load_model(pair["contract"], env)
                results_sc = evaluate_multi_seed(model_sc, env, NUM_EPISODES, EVAL_SEEDS)
                all_results[f"{name} (Contract)"] = results_sc
                summarize(results_sc, f"{name} — Contract — {env_name}")
            else:
                print(f"\n[SKIP] {name} (contract): file not found at\n  {pair['contract']}")

            nc_key = f"{name} (No Contract)"
            sc_key = f"{name} (Contract)"
            if nc_key in all_results and sc_key in all_results:
                nc = all_results[nc_key]
                sc = all_results[sc_key]
                print(f"\n  Δ (Contract − No Contract) for {name} on {env_name}  "
                      f"[across-seed means]:")
                print(f"  {'─' * 50}")
                print(f"  Reward:          {sc['rewards'].mean() - nc['rewards'].mean():>+8.2f}")
                print(f"  Final SoC:       {sc['final_socs'].mean() - nc['final_socs'].mean():>+8.1f}%")
                print(f"  Charge Cost:     {sc['charge_costs'].mean() - nc['charge_costs'].mean():>+8.4f} EUR")
                print(f"  V2G Revenue:     {sc['discharge_revenues'].mean() - nc['discharge_revenues'].mean():>+8.4f} EUR")
                print(f"  Net Cost:        {sc['net_costs'].mean() - nc['net_costs'].mean():>+8.4f} EUR")
                print(f"  Compliance:      {(sc['compliances'].mean() - nc['compliances'].mean())*100:>+8.1f}%")
                print(f"  Peak kWh:        {sc['peak_kwhs'].mean() - nc['peak_kwhs'].mean():>+8.3f}")

        if len(all_results) >= 2:
            print(f"\n{'='*70}")
            print(f"  Summary Table — {env_name}  (across-seed mean ± std)")
            print(f"{'='*70}")
            print(f"\n{'Model':<35} {'Reward':>16} {'SoC%':>14} {'Net':>18} {'Compl%':>14} {'Peak':>14}")
            print("-" * 115)
            for name, r in all_results.items():
                print(
                    f"{name:<35} "
                    f"{r['rewards'].mean():>7.2f} ± {r['rewards'].std():<6.2f} "
                    f"{r['final_socs'].mean():>6.1f} ± {r['final_socs'].std():<5.1f} "
                    f"{r['net_costs'].mean():>8.3f} ± {r['net_costs'].std():<7.3f} "
                    f"{r['compliances'].mean()*100:>6.1f} ± {r['compliances'].std()*100:<5.1f} "
                    f"{r['peak_kwhs'].mean():>6.3f} ± {r['peak_kwhs'].std():<5.3f}"
                )
            print("-" * 115)

            plot_comparison(
                all_results,
                save_path=f"comparison1_{_env_slug(env_name)}.png",
                title=f"Contract Awareness — {env_name}",
            )

            # Per-env suffixed tex files
            slug = _env_slug(env_name)
            six_tex = os.path.join(RESULTS_DIR, f"six_configuration_table_{slug}.tex")
            welch_tex = os.path.join(RESULTS_DIR, f"welch_ttest_{slug}.tex")
            write_six_config_table(
                all_results, six_tex, env_name,
                n_seeds=len(EVAL_SEEDS), n_episodes=NUM_EPISODES,
            )
            write_welch_ttest(
                all_results, welch_tex, env_name, n_seeds=len(EVAL_SEEDS),
            )

            # Canonical unsuffixed tex files (paths requested by Tasks 5 & 6)
            if env_name == CANONICAL_ENV_NAME:
                canonical_six = os.path.join(RESULTS_DIR, "six_configuration_table.tex")
                canonical_welch = os.path.join(RESULTS_DIR, "welch_ttest.tex")
                write_six_config_table(
                    all_results, canonical_six, env_name,
                    n_seeds=len(EVAL_SEEDS), n_episodes=NUM_EPISODES,
                )
                write_welch_ttest(
                    all_results, canonical_welch, env_name, n_seeds=len(EVAL_SEEDS),
                )


if __name__ == "__main__":
    main()