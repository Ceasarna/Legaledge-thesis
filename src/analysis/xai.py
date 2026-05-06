"""
XAI Dashboard — Explainable FRL with Contract Compliance
=========================================================
Runs a single full episode, computes SHAP values at every timestep,
and produces two figures:

  Figure 1 — Episode Dashboard
    • Charging action + electricity price (peak hours shaded)
    • SoC trajectory
    • SHAP feature-importance heatmap over the episode

  Figure 2 — Counterfactual Waterfall
    • Explains the most contract-relevant step:
        - If non-compliant: "Why did the agent charge during peak hours?"
        - If compliant:     "What restrained the agent during peak hours?"
"""

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from stable_baselines3 import PPO
from src.envs.charging_env import SmartChargingEnv

# ── Configuration ─────────────────────────────────────────────────────────────
MODEL_NAME = "Tesla Model 3 LR"
AGENT_ID   = 1000
MODEL_PATH = "final_model_single_contract_aware.pt"

PEAK_START, PEAK_END = 17, 20   # peak-hour window (inclusive)
BG_STEPS   = 40000              # background steps for SHAP baseline
BG_KMEANS  = 25                 # kmeans clusters to speed up KernelExplainer

FEATURE_NAMES = [
    "Price (norm)",
    "SoC Δ",
    "Time Remaining",
    "Hour of Day",
    "Forecast +1h", "Forecast +2h", "Forecast +3h",
    "Forecast +4h", "Forecast +5h", "Forecast +6h",
]

# Environment & model
env = SmartChargingEnv(
    rl_model="ppo",
    data_source=2024,
    model_name=MODEL_NAME,
    random_duration=False,
    random_arrival=True,
)

model = PPO(
    "MlpPolicy", env,
    device="cpu", verbose=0,
    policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
)
model.policy.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.policy.eval()
print(f"Loaded: {MODEL_PATH}")

# Background data for SHAP baseline
print(f"Collecting {BG_STEPS} background steps...")
bg_obs = []
obs, _ = env.reset()
for _ in range(BG_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    bg_obs.append(obs)
    obs, _, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
bg_obs = np.array(bg_obs)

# SHAP wrapper + explainer
def agent_predict_wrapper(observations):
    actions, _ = model.predict(observations, deterministic=True)
    return actions.squeeze(-1) if actions.ndim > 1 else actions

print("Building KernelExplainer...")
explainer = shap.KernelExplainer(
    agent_predict_wrapper,
    shap.kmeans(bg_obs, BG_KMEANS),
)
base_value = float(np.array(explainer.expected_value).squeeze())

# Run one full episode, collect per-step data
print("Running episode...")
episode_obs     = []
episode_actions = []
episode_prices  = []
episode_socs    = []
episode_hours   = []
episode_rewards = []

obs, _ = env.reset(seed=4)
raw_env = env.unwrapped
terminated = truncated = False

while not (terminated or truncated):
    step_idx = raw_env.current_step
    hour = (raw_env.arrival_time + step_idx) % raw_env.steps_per_day

    action, _ = model.predict(obs, deterministic=True)
    price = raw_env._get_price(step_idx)

    episode_obs.append(obs.copy())
    episode_actions.append(float(action.squeeze()))
    episode_prices.append(price)
    episode_hours.append(int(hour))
    episode_socs.append(raw_env.ev.get_soc())

    obs, reward, terminated, truncated, info = env.step(action)
    episode_rewards.append(reward)

episode_obs     = np.array(episode_obs)       # (T, n_features)
episode_actions = np.array(episode_actions)   # (T,)  — normalised [-1, 1]
episode_prices  = np.array(episode_prices)
episode_socs    = np.array(episode_socs)
episode_hours   = np.array(episode_hours)
T = len(episode_obs)
steps = np.arange(T)

print(f"Episode length: {T} steps  |  Final SoC: {episode_socs[-1]:.1f}%")

# SHAP values for every timestep
print(f"Computing SHAP for all {T} timesteps")
shap_values = explainer.shap_values(episode_obs)   # (T, n_features)

# Identify the "most interesting" step for the counterfactual─
peak_mask  = (episode_hours >= PEAK_START) & (episode_hours <= PEAK_END)
peak_steps = np.where(peak_mask)[0]

# Non-compliant if the agent charged (positive action) during peak hours
# Discharging during peak is fine (and actually beneficial for the grid)
peak_charging = episode_actions[peak_mask] if len(peak_steps) else np.array([])
non_compliant_steps = peak_steps[peak_charging > 0.05] if len(peak_steps) else np.array([], dtype=int)

if len(non_compliant_steps):
    # Pick the peak step with the highest charging power
    cf_step = non_compliant_steps[np.argmax(episode_actions[non_compliant_steps])]
    cf_label = f"Step {cf_step} (Hour {episode_hours[cf_step]:02d}:00) — Non-compliant peak charging"
    cf_color = "#d62728"
else:
    # All peak hours are compliant — pick the step with most restraint (lowest action)
    if len(peak_steps):
        cf_step = peak_steps[np.argmin(episode_actions[peak_steps])]
        cf_label = f"Step {cf_step} (Hour {episode_hours[cf_step]:02d}:00) — Compliant: agent restrained during peak"
    else:
        cf_step = int(np.argmax(episode_actions))
        cf_label = f"Step {cf_step} — Highest charging action"
    cf_color = "#2ca02c"

print(f"Counterfactual step: {cf_label}")

# FIGURE 1 — Episode Dashboard
fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.2, 2.5]})
fig.suptitle(
    f"XAI Episode Dashboard — {MODEL_NAME}  |  "
    f"Arrival H{episode_hours[0]:02d}  |  "
    f"Final SoC {episode_socs[-1]:.1f}%",
    fontsize=13, fontweight="bold",
)

# ── shade peak hours on all axes ──
def shade_peak(ax):
    for s in steps:
        if peak_mask[s]:
            ax.axvspan(s - 0.5, s + 0.5, color="#ff7f0e", alpha=0.12, zorder=0)

# Panel A: Price + Charging/Discharging Action
ax0 = axes[0]
shade_peak(ax0)
color_price     = "#1f77b4"
color_charge    = "#d62728"    # positive action  → charging
color_discharge = "#17becf"    # negative action  → discharging (V2G)

bar_colors = np.where(episode_actions >= 0, color_charge, color_discharge)

ax0_r = ax0.twinx()
ax0.bar(steps, episode_actions, color=bar_colors, alpha=0.65)
ax0.axhline(0, color="gray", linewidth=0.8, linestyle="-")
ax0_r.plot(steps, episode_prices, color=color_price, linewidth=1.8, label="Price (€/MWh)")
ax0_r.set_ylabel("Price (€/MWh)", color=color_price)
ax0.set_ylabel("Action  (−1 = full discharge  /  +1 = full charge)")
ax0.set_ylim(-1.1, 1.1)
ax0.set_title("Charging / Discharging Behaviour vs. Electricity Price", fontsize=10)

ax0.axvline(cf_step, color=cf_color, linewidth=2, linestyle="--")
patch_peak      = mpatches.Patch(color="#ff7f0e", alpha=0.35, label="Peak hours (17–20)")
patch_charge    = mpatches.Patch(color=color_charge,    alpha=0.7, label="Charge (+)")
patch_discharge = mpatches.Patch(color=color_discharge, alpha=0.7, label="Discharge / V2G (−)")
line_price      = plt.Line2D([0], [0], color=color_price, linewidth=1.8, label="Price")
leg0 = ax0_r.legend(handles=[patch_peak, patch_charge, patch_discharge, line_price],
                  fontsize=8, loc="upper left", framealpha=1.0)
leg0.set_zorder(100)

# Panel B: SoC Trajectory 
ax1 = axes[1]
shade_peak(ax1)
ax1.plot(steps, episode_socs, color="#9467bd", linewidth=2, label="SoC (%)")
target_soc = raw_env.ev.get_target_soc()
ax1.axhline(target_soc, color="#9467bd", linestyle=":", linewidth=1.5, label=f"Target SoC {target_soc:.0f}%")
ax1.set_ylabel("SoC (%)")
ax1.set_ylim(0, 105)
ax1.set_title("State of Charge", fontsize=10)
leg1 = ax1.legend(fontsize=8, loc="lower right", framealpha=1.0)
leg1.set_zorder(100)
ax1.axvline(cf_step, color=cf_color, linewidth=2, linestyle="--")

# Panel C: SHAP Heatmap (feature × timestep)
ax2 = axes[2]
heatmap = shap_values.T 
vmax = np.abs(heatmap).max()
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
im = ax2.imshow(
    heatmap, aspect="auto", cmap="RdBu_r", norm=norm,
    extent=[-0.5, T - 0.5, len(FEATURE_NAMES) - 0.5, -0.5],
    interpolation="nearest",
)
ax2.set_yticks(range(len(FEATURE_NAMES)))
ax2.set_yticklabels(FEATURE_NAMES, fontsize=8)
ax2.set_xlabel("Timestep (hour of session)")
ax2.set_title("SHAP Feature Importance per Timestep  (red = push charge ↑, blue = push charge ↓)", fontsize=10)

# overlay peak shading on heatmap
for s in steps:
    if peak_mask[s]:
        ax2.axvspan(s - 0.5, s + 0.5, color="#ff7f0e", alpha=0.15, zorder=2)
ax2.axvline(cf_step, color=cf_color, linewidth=2, linestyle="--", zorder=3)

cbar = fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.02, pad=0.01)
cbar.set_label("SHAP value/n(impact on action)", fontsize=8)

axes[2].set_xticks(steps)
axes[2].set_xticklabels([f"{h:02d}h" for h in episode_hours], rotation=45, fontsize=7)

plt.tight_layout()
plt.savefig("xai_episode_dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved xai_episode_dashboard.png")

# FIGURE 2 — Counterfactual Waterfall
print(f"/nRendering counterfactual waterfall: {cf_label}")

explanation = shap.Explanation(
    values      = shap_values[cf_step],
    base_values = base_value,
    data        = episode_obs[cf_step],
    feature_names = FEATURE_NAMES,
)

plt.figure(figsize=(9, 5))
shap.waterfall_plot(explanation, show=False)
plt.title(cf_label, fontsize=11, color=cf_color, pad=12)
plt.tight_layout()
plt.savefig("xai_counterfactual_waterfall.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved xai_counterfactual_waterfall.png")

# FIGURE 3 — The "What-If" Counterfactual
print("\n--- Generating Counterfactual ---")

altered_obs = episode_obs[cf_step].copy()

# FEATURE_NAMES[0] is "Price (norm)". Artificially negative (super cheap)
altered_obs[0] = 1

# See what the agent WOULD have done with this new fake price
new_action, _ = model.predict(altered_obs, deterministic=True)
new_action_value = float(new_action.squeeze())

print(f"Original Action: {episode_actions[cf_step]:.2f}")
print(f"New 'What-If' Action: {new_action_value:.2f}")

# Compute new SHAP values for this altered reality
cf_shap_values = explainer.shap_values(np.array([altered_obs])) 

cf_explanation = shap.Explanation(
    values      = cf_shap_values[0], # Take the first (and only) result
    base_values = base_value,
    data        = altered_obs,
    feature_names = FEATURE_NAMES,
)

plt.figure(figsize=(9, 5))
shap.waterfall_plot(cf_explanation, show=False)
plt.title(f"COUNTERFACTUAL: What if price was raised to max (1)?\nNew Action: {new_action_value:.2f}", fontsize=11, color="purple", pad=12)
plt.tight_layout()
plt.savefig("xai_counterfactual_what_if.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved xai_counterfactual_what_if.png")