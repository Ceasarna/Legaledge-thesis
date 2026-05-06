"""
Evaluate two pre-trained collective models (contract-aware vs not-contract-aware)
using the CheckpointGovernance smart contract with multi-agent sync.

Runs NUM_SEEDS independent evaluation seeds per model so we can report
mean ± std and a (seed-level) Welch's t-test for compliance.

Usage:
    1. Make sure Hardhat node is running:  npx hardhat node
    2. Deploy CheckpointGovernance contract:  npx hardhat ignition deploy ...
    3. Run:  python -m src.analysis.eval_collective

TensorBoard logs go to ./logs/eval_collective/<model_tag>/seed_<seed>/agent_<id>
View with:  tensorboard --logdir ./logs/eval_collective
"""

import sys
import os

# Monkey-patch config BEFORE any env/task imports
import src.federated.config as cfg
cfg.COLLECTIVE_SCENARIO_EPISODIC = True
cfg.COLLECTIVE_SCENARIO_SERVER_ROUND = False
cfg.SMART_CONTRACT_AWARE = True
cfg.PARETO_REWARD = False

import threading
import torch
import numpy as np
from scipy import stats
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3 import PPO

from src.envs.charging_env import SmartChargingEnv
from src.governance.contract_bridge import CheckpointGovernanceBridge

# ── Configuration ────────────────────────────────────────────────────
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
NUM_AGENTS = 4
NUM_EVAL_EPISODES = 50  # per seed
NUM_SEEDS = 10
SEED_LIST = list(range(42, 42 + NUM_SEEDS))  # 42..51
EV_ROSTER = ["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"]

MODEL_PATHS = {
    "contract_aware": "final_model_collective_contract_aware_episodic4_04_22.pt",
    "not_contract_aware": "final_model_collective_contract_aware_episodic2.pt",
}

# Round-id allocation: each (seed, model) pair gets its own round so contract
# state never collides — a (round_id, episode_id) the contract has already seen
# is treated as finalized, which would corrupt the eval.
EVAL_ROUND_ID_BASE = 900

def _round_id(seed_idx: int, model_idx: int) -> int:
    return EVAL_ROUND_ID_BASE + seed_idx * len(MODEL_PATHS) + model_idx


def eval_agent(
    agent_id: int,
    model_weights: dict,
    bridge,
    num_episodes: int,
    tb_writer: SummaryWriter,
    episode_barrier: threading.Barrier,
    checkpoint_barrier: threading.Barrier,
    results: dict,
    model_tag: str,
    round_id: int,
    seed_base: int,
):
    """Run deterministic eval episodes for one agent. Syncs with other agents via barriers."""
    ev_model = EV_ROSTER[agent_id % len(EV_ROSTER)]

    env = SmartChargingEnv(
        rl_model="ppo",
        data_source=2024,
        bridge=bridge,
        model_name=ev_model,
        random_duration=False,
        random_arrival=True,
        agent_id=agent_id,
        round_id=round_id,
    )

    net_arch = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    model = PPO("MlpPolicy", env, device="cpu", verbose=0, policy_kwargs=net_arch)
    model.policy.load_state_dict(model_weights, strict=True)

    rewards_list = []
    compliance_list = []
    final_soc_list = []
    energy_list = []
    peak_energy_list = []

    for ep in range(num_episodes):
        env.episode_slot = ep
        ep_seed = seed_base + ep
        obs, _ = env.reset(seed=ep_seed)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        # Extract episode metrics
        final_soc = info.get("final_soc", 0.0)
        total_charge = info.get("total_charge", 0.0)
        collective_util = info.get("collective_utilization", 0.0)
        power_log = info.get("power_log", {})
        peak_energy = sum(v for h, v in power_log.items() if 17 <= h <= 20 and v > 0)

        locally_compliant = collective_util <= 1.0

        # Barrier: wait for all agents to finish this episode before finalizing
        episode_barrier.wait()

        # Finalize episode on the contract
        if bridge is not None:
            try:
                session_len = env.ev.departure_time - env.ev.arrival_time
                final_checkpoint = session_len // env.checkpoint_interval
                result = bridge.finalize_episode(round_id, ep, final_checkpoint)
                locally_compliant = not result["penalized"]
            except Exception as e:
                print(f"[Agent {agent_id}] Finalize error: {e}")

        # Barrier: wait for finalization before starting next episode
        episode_barrier.wait()

        rewards_list.append(ep_reward)
        compliance_list.append(float(locally_compliant))
        final_soc_list.append(final_soc)
        energy_list.append(total_charge)
        peak_energy_list.append(peak_energy)

        # TensorBoard logging
        tb_writer.add_scalar("eval/episode_reward", ep_reward, ep)
        tb_writer.add_scalar("eval/final_soc", final_soc, ep)
        tb_writer.add_scalar("eval/energy_kwh", total_charge, ep)
        tb_writer.add_scalar("eval/peak_energy_kwh", peak_energy, ep)
        tb_writer.add_scalar("eval/compliant", float(locally_compliant), ep)
        tb_writer.add_scalar("eval/collective_utilization", collective_util, ep)

        # Rolling averages
        tb_writer.add_scalar("avg50/reward", np.mean(rewards_list[-50:]), ep)
        tb_writer.add_scalar("avg50/compliance_rate", np.mean(compliance_list[-50:]), ep)
        tb_writer.add_scalar("avg50/final_soc", np.mean(final_soc_list[-50:]), ep)
        tb_writer.add_scalar("avg50/energy_kwh", np.mean(energy_list[-50:]), ep)

        if (ep + 1) % 50 == 0:
            tb_writer.flush()
            print(
                f"  [{model_tag}] Agent {agent_id} | ep {ep+1}/{num_episodes} | "
                f"compliance={np.mean(compliance_list[-50:]):.2f} | "
                f"soc={np.mean(final_soc_list[-50:]):.1f}% | "
                f"energy={np.mean(energy_list[-50:]):.2f} kWh"
            )

    tb_writer.flush()
    tb_writer.close()

    results[agent_id] = {
        "rewards": rewards_list,
        "compliance": compliance_list,
        "final_soc": final_soc_list,
        "energy": energy_list,
        "peak_energy": peak_energy_list,
        "mean_reward": float(np.mean(rewards_list)),
        "std_reward": float(np.std(rewards_list)),
        "compliance_rate": float(np.mean(compliance_list)),
        "mean_final_soc": float(np.mean(final_soc_list)),
        "std_final_soc": float(np.std(final_soc_list)),
        "mean_energy_kwh": float(np.mean(energy_list)),
        "std_energy_kwh": float(np.std(energy_list)),
        "mean_peak_energy": float(np.mean(peak_energy_list)),
    }


def eval_model(model_tag: str, model_path: str, round_id: int, seed: int, use_contract: bool = True):
    """Evaluate a single model across NUM_AGENTS agents in parallel for one seed."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_tag} ({model_path})")
    print(f"Contract: {'ON' if use_contract else 'OFF'} | Episodes: {NUM_EVAL_EPISODES} "
          f"| seed={seed} | round_id={round_id}")
    print(f"{'='*60}")

    weights = torch.load(model_path, map_location="cpu", weights_only=True)

    episode_barrier = threading.Barrier(NUM_AGENTS)
    checkpoint_barrier = threading.Barrier(NUM_AGENTS)
    results = {}

    threads = []
    for agent_id in range(NUM_AGENTS):
        if use_contract:
            bridge = CheckpointGovernanceBridge(CONTRACT_ADDRESS, agent_id=agent_id)
        else:
            bridge = None

        log_dir = f"./logs/eval_collective/{model_tag}/seed_{seed}/agent_{agent_id}"
        tb_writer = SummaryWriter(log_dir=log_dir)

        t = threading.Thread(
            target=eval_agent,
            args=(
                agent_id, weights, bridge, NUM_EVAL_EPISODES,
                tb_writer, episode_barrier, checkpoint_barrier, results,
                model_tag, round_id, seed,
            ),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Print summary
    print(f"\n--- {model_tag} Summary ---")
    all_compliance = []
    all_soc = []
    all_energy = []
    for aid in sorted(results.keys()):
        r = results[aid]
        all_compliance.append(r["compliance_rate"])
        all_soc.append(r["mean_final_soc"])
        all_energy.append(r["mean_energy_kwh"])
        print(
            f"  Agent {aid}: compliance={r['compliance_rate']:.3f} | "
            f"soc={r['mean_final_soc']:.1f}% | "
            f"energy={r['mean_energy_kwh']:.2f} kWh | "
            f"reward={r['mean_reward']:.2f}"
        )
    print(
        f"  OVERALL: compliance={np.mean(all_compliance):.3f} | "
        f"soc={np.mean(all_soc):.1f}% | "
        f"energy={np.mean(all_energy):.2f} kWh"
    )
    return results


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    print(f"Working directory: {os.getcwd()}")

    # all_results[tag] = list of seed runs.
    # Each entry: {"seed": int, "round_id": int, "results": {agent_id -> stats}}
    all_results = {tag: [] for tag in MODEL_PATHS}

    for seed_idx, seed in enumerate(SEED_LIST):
        for model_idx, (tag, path) in enumerate(MODEL_PATHS.items()):
            if not os.path.exists(path):
                print(f"WARNING: Model not found: {path} — skipping")
                continue
            rid = _round_id(seed_idx, model_idx)
            print(f"\n>>> Seed {seed} ({seed_idx+1}/{NUM_SEEDS}) | "
                  f"Model {tag} | round_id={rid}")
            seed_results = eval_model(tag, path, round_id=rid, seed=seed, use_contract=True)
            all_results[tag].append({"seed": seed, "round_id": rid, "results": seed_results})

    # Drop tags with no successful runs
    all_results = {tag: runs for tag, runs in all_results.items() if runs}

    # ── Stats helpers ──────────────────────────────────────────────
    def per_seed_means(runs, key):
        """For each seed, pool across agents and return the per-seed mean."""
        out = []
        for run in runs:
            pooled = []
            for r in run["results"].values():
                pooled.extend(r[key])
            out.append(float(np.mean(pooled)))
        return np.array(out)

    def per_agent_per_seed_means(runs, agent_id, key):
        """For one agent across seeds: list of per-seed means."""
        out = []
        for run in runs:
            r = run["results"].get(agent_id)
            if r is not None:
                out.append(float(np.mean(r[key])))
        return np.array(out)

    def _sig_stars(p):
        if p < 0.001: return "***"
        if p < 0.01:  return "**"
        if p < 0.05:  return "*"
        return ""

    out_dir = "./logs/eval_collective"
    os.makedirs(out_dir, exist_ok=True)

    PRETTY_NAMES = {
        "contract_aware": "Contract-Aware",
        "not_contract_aware": "Baseline (No Contract)",
    }

    # ── Table 1: Comparison (seed-level mean ± std) ────────────────
    latex_lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Evaluation comparison: contract-aware vs.\ baseline model "
        f"({NUM_EVAL_EPISODES} episodes per seed, {NUM_SEEDS} seeds, "
        f"{NUM_AGENTS} agents). Values are mean $\\pm$ std across seeds.}}",
        r"  \label{tab:eval-comparison}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l c c c c}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Compliance} & \textbf{Final SoC (\%)} "
        r"& \textbf{Energy (kWh)} & \textbf{Reward} \\",
        r"    \midrule",
    ]
    for tag, runs in all_results.items():
        name = PRETTY_NAMES.get(tag, tag)
        comp = per_seed_means(runs, "compliance")
        soc  = per_seed_means(runs, "final_soc")
        eng  = per_seed_means(runs, "energy")
        rew  = per_seed_means(runs, "rewards")
        latex_lines.append(
            f"    {name} & ${np.mean(comp):.3f} \\pm {np.std(comp):.3f}$ "
            f"& ${np.mean(soc):.1f} \\pm {np.std(soc):.1f}$ "
            f"& ${np.mean(eng):.2f} \\pm {np.std(eng):.2f}$ "
            f"& ${np.mean(rew):.2f} \\pm {np.std(rew):.2f}$ \\\\"
        )
    latex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]

    # ── Table 2: Per-agent breakdown (mean ± std across seeds) ────
    latex_lines += [
        "",
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Per-agent evaluation breakdown "
        f"(mean $\\pm$ std across {NUM_SEEDS} seeds).}}",
        r"  \label{tab:eval-per-agent}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{l l l c c c}",
        r"    \toprule",
        r"    \textbf{Model} & \textbf{Agent} & \textbf{EV} "
        r"& \textbf{Compliance} & \textbf{Final SoC (\%)} & \textbf{Energy (kWh)} \\",
        r"    \midrule",
    ]
    for tag, runs in all_results.items():
        name = PRETTY_NAMES.get(tag, tag)
        for aid in range(NUM_AGENTS):
            comp = per_agent_per_seed_means(runs, aid, "compliance")
            soc  = per_agent_per_seed_means(runs, aid, "final_soc")
            eng  = per_agent_per_seed_means(runs, aid, "energy")
            if len(comp) == 0:
                continue
            ev = EV_ROSTER[aid % len(EV_ROSTER)]
            latex_lines.append(
                f"    {name} & {aid} & {ev} "
                f"& ${np.mean(comp):.3f} \\pm {np.std(comp):.3f}$ "
                f"& ${np.mean(soc):.1f} \\pm {np.std(soc):.1f}$ "
                f"& ${np.mean(eng):.2f} \\pm {np.std(eng):.2f}$ \\\\"
            )
    latex_lines += [
        r"    \bottomrule",
        r"  \end{tabular}}",
        r"\end{table}",
    ]

    # ── Table 3: Welch's t-test (seed-level, n = NUM_SEEDS) ────────
    tags = list(all_results.keys())
    metrics_to_test = [
        ("Compliance",      "compliance"),
        ("Final SoC (\\%)", "final_soc"),
        ("Energy (kWh)",    "energy"),
        ("Reward",          "rewards"),
    ]

    if len(tags) == 2:
        tag_a, tag_b = tags
        name_a = PRETTY_NAMES.get(tag_a, tag_a)
        name_b = PRETTY_NAMES.get(tag_b, tag_b)

        latex_lines += [
            "",
            r"\begin{table}[htbp]",
            r"  \centering",
            f"  \\caption{{Welch's $t$-test on seed-level means: {name_a} vs.\\ {name_b} "
            f"($n = {NUM_SEEDS}$ seeds per group).}}",
            r"  \label{tab:welch-t-test}",
            r"  \resizebox{\textwidth}{!}{%",
            r"  \begin{tabular}{l c c c c c}",
            r"    \toprule",
            f"    \\textbf{{Metric}} & \\textbf{{{name_a}}} & \\textbf{{{name_b}}} "
            r"& \textbf{$t$} & \textbf{$p$} & \textbf{Sig.} \\",
            r"    \midrule",
        ]

        for metric_label, key in metrics_to_test:
            a = per_seed_means(all_results[tag_a], key)
            b = per_seed_means(all_results[tag_b], key)
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            stars = _sig_stars(p_val)

            if key == "compliance":
                fmt_a = f"${np.mean(a):.3f} \\pm {np.std(a):.3f}$"
                fmt_b = f"${np.mean(b):.3f} \\pm {np.std(b):.3f}$"
            else:
                fmt_a = f"${np.mean(a):.2f} \\pm {np.std(a):.2f}$"
                fmt_b = f"${np.mean(b):.2f} \\pm {np.std(b):.2f}$"

            p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "$< 0.0001$"
            latex_lines.append(
                f"    {metric_label} & {fmt_a} & {fmt_b} "
                f"& ${t_stat:.2f}$ & {p_str} & {stars} \\\\"
            )

        latex_lines += [
            r"    \bottomrule",
            r"    \multicolumn{6}{l}{\footnotesize "
            r"${}^{*}p<0.05$, ${}^{**}p<0.01$, ${}^{***}p<0.001$ (Welch's $t$-test, two-tailed)} \\",
            r"  \end{tabular}}",
            r"\end{table}",
        ]

    tex_path = f"{out_dir}/eval_tables.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(latex_lines) + "\n")

    # ── Console summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"COMPARISON  ({NUM_SEEDS} seeds × {NUM_EVAL_EPISODES} episodes × {NUM_AGENTS} agents)")
    print(f"{'='*60}")
    for tag, runs in all_results.items():
        comp = per_seed_means(runs, "compliance")
        soc  = per_seed_means(runs, "final_soc")
        eng  = per_seed_means(runs, "energy")
        rew  = per_seed_means(runs, "rewards")
        print(f"  {tag:30s} | compliance={np.mean(comp):.3f}±{np.std(comp):.3f} | "
              f"soc={np.mean(soc):.1f}±{np.std(soc):.1f}% | "
              f"energy={np.mean(eng):.2f}±{np.std(eng):.2f} kWh | "
              f"reward={np.mean(rew):.2f}±{np.std(rew):.2f}")

    if len(tags) == 2:
        print(f"\n  Welch's t-test (seed-level, n={NUM_SEEDS}):")
        for metric_label, key in metrics_to_test:
            a = per_seed_means(all_results[tags[0]], key)
            b = per_seed_means(all_results[tags[1]], key)
            t_stat, p_val = stats.ttest_ind(a, b, equal_var=False)
            print(f"    {metric_label:20s}  t={t_stat:+.2f}  p={p_val:.4f} {_sig_stars(p_val)}")

    print(f"\nExported: {tex_path}")
    print(f"TensorBoard: tensorboard --logdir ./logs/eval_collective")


if __name__ == "__main__":
    main()
