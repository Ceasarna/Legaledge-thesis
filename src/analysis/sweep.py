"""
Pareto Frontier Sweep
=====================
After training a weight-conditioned Pareto model, this script:
  1. Generates weight combinations on the 3-simplex (w_price, w_soc, w_compliance)
  2. Runs N episodes per weight setting with the trained global model
  3. Records: avg charging cost, final SoC error, compliance rate, peak/offpeak energy
  4. Identifies the Pareto-optimal (non-dominated) points
  5. Plots 2D Pareto frontiers and calculates hypervolume
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from pytorchexample.charging_env import SmartChargingEnv


# ── Config ──────────────────────────────────────────────────────────────────
MODEL_PATH_1 = "C:/Users/toek3476/FRL/Thesis_2/Thesis_code/quickstart-pytorch/final_model_single_contract_aware_pareto8_noterminalsoc.pt"
MODEL_PATH = "C:/Users/toek3476/FRL/Thesis_2/Thesis_code/quickstart-pytorch/final_model_single_not_contract_aware_pareto1.pt"


SEED = 42
NUM_EPISODES = 240          # episodes per weight setting
GRID_RESOLUTION = 11        # points per axis on the simplex (11 → ~66 combos)

PPO_KWARGS = dict(
    verbose=0,
    learning_rate=3e-4,
    batch_size=240,
    n_steps=4800,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
    device="cpu",
)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results")


# ── Weight generation ───────────────────────────────────────────────────────
def generate_simplex_weights(resolution: int = GRID_RESOLUTION) -> np.ndarray:
    """Generate evenly-spaced weight vectors on the 3-simplex that sum to 1."""
    weights = []
    for i in range(resolution):
        for j in range(resolution - i):
            k = resolution - 1 - i - j
            w = np.array([i, j, k], dtype=np.float64) / (resolution - 1)
            weights.append(w)
    return np.array(weights)


# ── Environment factory ─────────────────────────────────────────────────────
def make_env(seed: int) -> SmartChargingEnv:
    return SmartChargingEnv(
        rl_model="ppo",
        data_source=2024,
        model_name=["Tesla Model 3 LR"],
        random_duration=False,
        random_arrival=True,
        seed=seed,
        bridge=None,  # no blockchain needed for sweep
    )


# ── Pareto dominance ────────────────────────────────────────────────────────
def is_dominated(point, others):
    """Check if `point` is dominated by any row in `others`.
    Assumes all objectives are to be MINIMIZED."""
    for o in others:
        if np.all(o <= point) and np.any(o < point):
            return True
    return False


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


# ── Hypervolume (2D) ────────────────────────────────────────────────────────
def hypervolume_2d(points: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume (area dominated by `points` w.r.t. `ref`).
    Points and ref in minimization space."""
    pts = points[points[:, 0].argsort()]
    hv = 0.0
    prev_y = ref[1]
    for p in pts:
        if p[0] < ref[0] and p[1] < ref[1]:
            hv += (ref[0] - p[0]) * (prev_y - p[1])
            prev_y = min(prev_y, p[1])
    return hv


# ── Run sweep ───────────────────────────────────────────────────────────────
def run_sweep(model_path: str, weights_list: np.ndarray, num_episodes: int, seed: int):
    env = make_env(seed)
    model = PPO("MlpPolicy", env, **PPO_KWARGS)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    model.policy.load_state_dict(state_dict, strict=True)

    results = []
    total = len(weights_list)

    for idx, (w_price, w_soc, w_comp) in enumerate(weights_list):
        ep_charge_costs, ep_discharge_revenues, ep_net_costs = [], [], []
        ep_soc_errors, ep_compliances = [], []
        ep_peak, ep_offpeak, ep_rewards, ep_final_socs = [], [], [], []

        for ep in range(num_episodes):
            obs, _ = env.reset(seed=seed + ep)

            # Override the Dirichlet-sampled weights with the sweep weights
            env.unwrapped.w_price = w_price
            env.unwrapped.w_soc = w_soc
            env.unwrapped.w_compliance = w_comp
            # Rebuild obs so the weight channels match
            obs = env.unwrapped._get_observation_pareto()

            done = False
            ep_charge_cost = 0.0
            ep_discharge_rev = 0.0
            ep_reward = 0.0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                price_before = env.unwrapped._get_price()  # EUR/MWh
                obs, reward, terminated, truncated, info = env.step(action)
                energy = info.get("actual_energy_kwh", 0.0)
                if energy > 0:
                    ep_charge_cost += price_before * energy / 1000  # EUR
                elif energy < 0:
                    ep_discharge_rev += abs(energy) * price_before / 1000  # EUR
                ep_reward += reward
                done = terminated or truncated

            final_soc = info.get("final_soc", 0.0)
            target_soc = info.get("target_soc", 80.0)
            soc_error = abs(target_soc - final_soc)

            power_log = info.get("power_log", {})
            peak = sum(v for h, v in power_log.items() if 17 <= h <= 20 and v > 0)
            offpeak = sum(v for h, v in power_log.items() if not (17 <= h <= 20) and v > 0)

            ep_charge_costs.append(ep_charge_cost)
            ep_discharge_revenues.append(ep_discharge_rev)
            ep_net_costs.append(ep_charge_cost - ep_discharge_rev)
            ep_soc_errors.append(soc_error)
            ep_compliances.append(float(peak == 0.0))
            ep_peak.append(peak)
            ep_offpeak.append(offpeak)
            ep_rewards.append(ep_reward)
            ep_final_socs.append(final_soc)

        results.append({
            "w_price": w_price,
            "w_soc": w_soc,
            "w_compliance": w_comp,
            "avg_charge_cost": np.mean(ep_charge_costs),
            "std_charge_cost": np.std(ep_charge_costs),
            "avg_discharge_revenue": np.mean(ep_discharge_revenues),
            "std_discharge_revenue": np.std(ep_discharge_revenues),
            "avg_net_cost": np.mean(ep_net_costs),
            "std_net_cost": np.std(ep_net_costs),
            "avg_soc_error": np.mean(ep_soc_errors),
            "std_soc_error": np.std(ep_soc_errors),
            "avg_compliance": np.mean(ep_compliances),
            "ci_compliance": np.std(ep_compliances),
            "avg_peak_kwh": np.mean(ep_peak),
            "avg_offpeak_kwh": np.mean(ep_offpeak),
            "avg_reward": np.mean(ep_rewards),
            "avg_final_soc": np.mean(ep_final_socs),
        })

        print(f"[{idx+1:3d}/{total}]  w=({w_price:.2f}, {w_soc:.2f}, {w_comp:.2f})  "
              f"charge={results[-1]['avg_charge_cost']:7.4f}  v2g={results[-1]['avg_discharge_revenue']:7.4f}  "
              f"net={results[-1]['avg_net_cost']:+7.4f}  soc_err={results[-1]['avg_soc_error']:5.1f}%  "
              f"compliance={results[-1]['avg_compliance']:5.1%}  reward={results[-1]['avg_reward']:7.2f}")

    env.close()
    return results


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_pareto(results: list, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    net_costs = np.array([r["avg_net_cost"] for r in results])
    discharge_revs = np.array([r["avg_discharge_revenue"] for r in results])
    soc_errors = np.array([r["avg_soc_error"] for r in results])
    non_compliance = 1.0 - np.array([r["avg_compliance"] for r in results])
    # 95% confidence intervals (std / sqrt(n) * 1.96)
    ci_scale = 1.96 / np.sqrt(NUM_EPISODES)
    ci_net_costs = np.array([r["std_net_cost"] for r in results]) * ci_scale
    ci_soc_errors = np.array([r["std_soc_error"] for r in results]) * ci_scale
    ci_compliance = np.array([r["ci_compliance"] for r in results]) * ci_scale
    w_price = np.array([r["w_price"] for r in results])
    w_soc = np.array([r["w_soc"] for r in results])
    w_comp = np.array([r["w_compliance"] for r in results])

    # ── 1. Net Cost vs Non-Compliance (2D Pareto) ──
    obj_2d = np.column_stack([net_costs, non_compliance])
    mask_2d = pareto_front_mask(obj_2d)
    front_2d = obj_2d[mask_2d]
    front_2d = front_2d[front_2d[:, 0].argsort()]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(net_costs[~mask_2d], non_compliance[~mask_2d],
                xerr=ci_net_costs[~mask_2d], yerr=ci_compliance[~mask_2d],
                fmt="none", ecolor="gray", alpha=0.15, zorder=1)
    scatter = ax.scatter(net_costs[~mask_2d], non_compliance[~mask_2d],
                         c=w_comp[~mask_2d], cmap="coolwarm", alpha=0.4, s=40,
                         edgecolors="gray", linewidth=0.3, label="Dominated", zorder=2)
    ax.errorbar(net_costs[mask_2d], non_compliance[mask_2d],
                xerr=ci_net_costs[mask_2d], yerr=ci_compliance[mask_2d],
                fmt="none", ecolor="black", alpha=0.3, zorder=3)
    ax.scatter(net_costs[mask_2d], non_compliance[mask_2d],
               c=w_comp[mask_2d], cmap="coolwarm", s=100,
               edgecolors="black", linewidth=1.2, marker="D", label="Pareto-optimal", zorder=4)
    ax.plot(front_2d[:, 0], front_2d[:, 1], "k--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Avg Net Cost (EUR) [charge - V2G revenue]", fontsize=12)
    ax.set_ylabel("Non-Compliance Rate (1 - compliance)", fontsize=12)
    ax.set_title("Pareto Frontier: Net Cost vs Contract Compliance", fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("w_compliance", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_cost_vs_compliance.png"), dpi=150)
    plt.close(fig)

    # ── 2. Net Cost vs SoC Error (2D Pareto) ──
    obj_soc = np.column_stack([net_costs, soc_errors])
    mask_soc = pareto_front_mask(obj_soc)
    front_soc = obj_soc[mask_soc]
    front_soc = front_soc[front_soc[:, 0].argsort()]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.errorbar(net_costs[~mask_soc], soc_errors[~mask_soc],
                xerr=ci_net_costs[~mask_soc], yerr=ci_soc_errors[~mask_soc],
                fmt="none", ecolor="gray", alpha=0.15, zorder=1)
    scatter = ax.scatter(net_costs[~mask_soc], soc_errors[~mask_soc],
                         c=w_soc[~mask_soc], cmap="viridis", alpha=0.4, s=40,
                         edgecolors="gray", linewidth=0.3, label="Dominated", zorder=2)
    ax.errorbar(net_costs[mask_soc], soc_errors[mask_soc],
                xerr=ci_net_costs[mask_soc], yerr=ci_soc_errors[mask_soc],
                fmt="none", ecolor="black", alpha=0.3, zorder=3)
    ax.scatter(net_costs[mask_soc], soc_errors[mask_soc],
               c=w_soc[mask_soc], cmap="viridis", s=100,
               edgecolors="black", linewidth=1.2, marker="D", label="Pareto-optimal", zorder=4)
    ax.plot(front_soc[:, 0], front_soc[:, 1], "k--", alpha=0.6, linewidth=1)
    ax.set_xlabel("Avg Net Cost (EUR) [charge - V2G revenue]", fontsize=12)
    ax.set_ylabel("Avg SoC Error (%)", fontsize=12)
    ax.set_title("Pareto Frontier: Net Cost vs SoC Accuracy", fontsize=14)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("w_soc", fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_cost_vs_soc_error.png"), dpi=150)
    plt.close(fig)

    # ── 3. 3D scatter (all three objectives) ──
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    obj_3d = np.column_stack([net_costs, soc_errors, non_compliance])
    mask_3d = pareto_front_mask(obj_3d)
    ax.scatter(net_costs[~mask_3d], soc_errors[~mask_3d], non_compliance[~mask_3d],
               alpha=0.3, s=20, c="gray", label="Dominated")
    ax.scatter(net_costs[mask_3d], soc_errors[mask_3d], non_compliance[mask_3d],
               s=80, c="red", marker="D", edgecolors="black", label="Pareto-optimal")
    ax.set_xlabel("Net Cost (EUR)")
    ax.set_ylabel("SoC Error (%)")
    ax.set_zlabel("Non-Compliance")
    ax.set_title("3D Pareto Frontier", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "pareto_3d.png"), dpi=150)
    plt.close(fig)

    # ── 4. Weight-space heatmap ──
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    metrics = [
        (net_costs, "Avg Net Cost (EUR)", "RdYlGn_r"),
        (discharge_revs, "Avg V2G Revenue (EUR)", "Greens"),
        (soc_errors, "Avg SoC Error (%)", "Oranges"),
        (non_compliance, "Non-Compliance Rate", "Blues"),
    ]
    for ax, (vals, title, cmap) in zip(axes, metrics):
        sc = ax.scatter(w_price, w_soc, c=vals, cmap=cmap, s=60, edgecolors="gray", linewidth=0.3)
        plt.colorbar(sc, ax=ax)
        ax.set_xlabel("w_price")
        ax.set_ylabel("w_soc")
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Metrics across Weight Space (w_compliance = 1 - w_price - w_soc)", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "weight_space_heatmap.png"), dpi=150)
    plt.close(fig)

    print(f"\nPlots saved to {out_dir}/")


# ── Hypervolume & summary ───────────────────────────────────────────────────
def compute_metrics(results: list):
    net_costs = np.array([r["avg_net_cost"] for r in results])
    non_compliance = 1.0 - np.array([r["avg_compliance"] for r in results])
    soc_errors = np.array([r["avg_soc_error"] for r in results])

    # 2D hypervolume: Net Cost vs Non-Compliance
    obj_cn = np.column_stack([net_costs, non_compliance])
    mask_cn = pareto_front_mask(obj_cn)
    front_cn = obj_cn[mask_cn]
    ref_cn = np.array([net_costs.max() * 1.1, 1.1])  # reference point slightly beyond worst
    hv_cn = hypervolume_2d(front_cn, ref_cn)

    # 2D hypervolume: Net Cost vs SoC Error
    obj_cs = np.column_stack([net_costs, soc_errors])
    mask_cs = pareto_front_mask(obj_cs)
    front_cs = obj_cs[mask_cs]
    ref_cs = np.array([net_costs.max() * 1.1, soc_errors.max() * 1.1])
    hv_cs = hypervolume_2d(front_cs, ref_cs)

    # Find the "knee" — closest to the ideal point (min net cost, 0 non-compliance)
    ideal = np.array([net_costs.min(), 0.0])
    dists = np.linalg.norm(obj_cn[mask_cn] - ideal, axis=1)
    knee_idx_in_front = np.argmin(dists)
    # Map back to original results
    front_indices = np.where(mask_cn)[0]
    knee_idx = front_indices[knee_idx_in_front]
    knee = results[knee_idx]

    # Sparsity of the cost-compliance front
    if len(front_cn) > 2:
        sorted_front = front_cn[front_cn[:, 0].argsort()]
        gaps = np.diff(sorted_front, axis=0)
        sparsity = np.mean(np.linalg.norm(gaps, axis=1))
    else:
        sparsity = float("inf")

    # ── Best balanced point (all 3 objectives) ──
    # Normalize each objective to [0, 1] using min-max, then find the point
    # with the smallest Euclidean distance to the utopia (all-zero) corner.
    # All objectives are minimized: net_cost, soc_error, non_compliance.
    obj_3d = np.column_stack([net_costs, soc_errors, non_compliance])
    obj_min = obj_3d.min(axis=0)
    obj_max = obj_3d.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0  # avoid division by zero if an objective is constant
    obj_norm = (obj_3d - obj_min) / obj_range  # each column in [0, 1]
    dist_to_utopia = np.linalg.norm(obj_norm, axis=1)  # distance to (0, 0, 0)
    best_balanced_idx = int(np.argmin(dist_to_utopia))
    best_balanced = results[best_balanced_idx]

    print("\n" + "=" * 70)
    print("PARETO SWEEP SUMMARY")
    print("=" * 70)
    print(f"Total weight settings evaluated:  {len(results)}")
    print(f"Pareto-optimal (net cost vs compliance): {mask_cn.sum()}")
    print(f"Pareto-optimal (net cost vs SoC error):  {mask_cs.sum()}")
    print(f"\nHypervolume (net cost vs compliance): {hv_cn:.2f}")
    print(f"Hypervolume (net cost vs SoC error):  {hv_cs:.2f}")
    print(f"Sparsity (net cost vs compliance):    {sparsity:.4f}")
    print(f"\nKnee point (best cost-compliance balance, 2D):")
    print(f"  Weights:      w_price={knee['w_price']:.2f}, w_soc={knee['w_soc']:.2f}, w_compliance={knee['w_compliance']:.2f}")
    print(f"  Charge Cost:  {knee['avg_charge_cost']:.4f} EUR")
    print(f"  V2G Revenue:  {knee['avg_discharge_revenue']:.4f} EUR")
    print(f"  Net Cost:     {knee['avg_net_cost']:.4f} EUR")
    print(f"  SoC Error:    {knee['avg_soc_error']:.1f}%")
    print(f"  Compliance:   {knee['avg_compliance']:.1%}")
    print(f"  Reward:       {knee['avg_reward']:.2f}")
    print(f"\nBest balanced point (all 3 objectives, normalized L2 to utopia):")
    print(f"  Weights:      w_price={best_balanced['w_price']:.2f}, w_soc={best_balanced['w_soc']:.2f}, w_compliance={best_balanced['w_compliance']:.2f}")
    print(f"  Charge Cost:  {best_balanced['avg_charge_cost']:.4f} EUR")
    print(f"  V2G Revenue:  {best_balanced['avg_discharge_revenue']:.4f} EUR")
    print(f"  Net Cost:     {best_balanced['avg_net_cost']:.4f} EUR")
    print(f"  SoC Error:    {best_balanced['avg_soc_error']:.1f}%")
    print(f"  Compliance:   {best_balanced['avg_compliance']:.1%}")
    print(f"  Final SoC:    {best_balanced['avg_final_soc']:.1f}%")
    print(f"  Reward:       {best_balanced['avg_reward']:.2f}")
    print("=" * 70)

    return {
        "hv_cost_compliance": hv_cn,
        "hv_cost_soc": hv_cs,
        "sparsity": sparsity,
        "knee": knee,
        "best_balanced": best_balanced,
        "n_pareto_cn": int(mask_cn.sum()),
        "n_pareto_cs": int(mask_cs.sum()),
    }


# ── Save raw results to CSV ────────────────────────────────────────────────
def save_csv(results: list, out_dir: str):
    import csv
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "sweep_results.csv")
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Raw results saved to {path}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    weights = generate_simplex_weights(GRID_RESOLUTION)
    print(f"Pareto Sweep: {len(weights)} weight settings x {NUM_EPISODES} episodes each")
    print(f"Model: {model_path}\n")

    results = run_sweep(model_path, weights, NUM_EPISODES, SEED)

    save_csv(results, OUT_DIR)
    summary = compute_metrics(results)
    plot_pareto(results, OUT_DIR)

    return results, summary


if __name__ == "__main__":
    main()
