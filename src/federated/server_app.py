"""FLWR-TD3: A Flower / PyTorch app."""

# Default FLWR imports
import time
import torch

from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg
from tensorboardX import SummaryWriter
import csv, os
from datetime import datetime
import numpy as np

# Specfic imports for this app
from stable_baselines3 import PPO
from src.envs.charging_env import SmartChargingEnv
from src.federated.config import COLLECTIVE_SCENARIO_EPISODIC, SMART_CONTRACT_AWARE, PARETO_REWARD

# Create ServerApp
app = ServerApp()

class CustomFedAvg(FedAvg):
    def __init__(self, train_config: ConfigRecord, run_id: str = "", **kwargs):
        super().__init__(**kwargs)
        self._train_config = train_config

        contract_tag = "contract" if SMART_CONTRACT_AWARE else "no_contract"
        run_dir = (
            f"./logs/benchmark/federated/{contract_tag}/{run_id}"
            if run_id else
            f"./logs/benchmark/federated/{contract_tag}"
        )
        self.tb_writer = SummaryWriter(log_dir=f"{run_dir}/server")
        self._csv_path = f"{run_dir}/server_metrics.csv"

        # Throughput CSV (round-level summary)
        self._throughput_csv_path = f"{run_dir}/throughput_metrics.csv"
        os.makedirs(run_dir, exist_ok=True)
        with open(self._throughput_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "n_agents", "rl_time_s", "blockchain_time_s",
                "fl_overhead_s", "aggregation_s", "rl_pct", "blockchain_pct",
                "mean_tx_latency_ms", "total_episodes",
                "total_blockchain_tx", "blockchain_window_s", "blockchain_window_tps",
                "round_window_s", "system_window_tps",
            ])

        self._concurrency_csv_path = f"{run_dir}/concurrency_log.csv"
        with open(self._concurrency_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "round", "agent_id", "pid", "round_wall_start", "round_wall_end",
                "first_tx_wall_s", "last_tx_wall_s", "n_blockchain_tx", "rss_mb",
            ])

    def aggregate_train(self, server_round, replies):
        _agg_start = time.perf_counter()

        round_times, rl_times, bc_times, fl_overheads = [], [], [], []
        first_tx_walls, last_tx_walls, agent_tx_counts = [], [], []
        round_wall_starts, round_wall_ends = [], []
        concurrency_rows = []

        for reply in replies:
            metrics = reply.content.get("metrics", MetricRecord({}))
            rt = float(metrics.get("round_time_s", 0.0))
            if rt > 0:
                round_times.append(rt)
                rl_times.append(float(metrics.get("rl_time_s", 0.0)))
                bc_times.append(float(metrics.get("blockchain_time_s", 0.0)))
                fl_overheads.append(float(metrics.get("fl_overhead_s", 0.0)))

            n_tx = int(metrics.get("n_blockchain_tx", 0))
            agent_tx_counts.append(n_tx)
            rws = float(metrics.get("round_wall_start", 0.0))
            rwe = float(metrics.get("round_wall_end", 0.0))
            if rws > 0 and rwe > 0:
                round_wall_starts.append(rws)
                round_wall_ends.append(rwe)
            if n_tx > 0:
                first_tx_walls.append(float(metrics.get("first_tx_wall_s", 0.0)))
                last_tx_walls.append(float(metrics.get("last_tx_wall_s", 0.0)))
            concurrency_rows.append((
                int(metrics.get("agent_id", -1)),
                int(metrics.get("pid", 0)),
                rws, rwe,
                float(metrics.get("first_tx_wall_s", 0.0)),
                float(metrics.get("last_tx_wall_s", 0.0)),
                n_tx,
                float(metrics.get("rss_mb", 0.0)),
            ))

        # Normal FedAvg aggregation
        arrays, aggregated_metrics = super().aggregate_train(server_round, replies)

        if aggregated_metrics:
            m = dict(aggregated_metrics)
            train_reward     = m.get("train_reward", 0.0)
            train_final_soc  = m.get("train_final_soc", 0.0)
            train_energy_kwh = m.get("mean_energy_kwh", 0.0)
            train_compliance = m.get("train_compliance", 0.0)

            self.tb_writer.add_scalar("train/train_reward",       train_reward,     server_round)
            self.tb_writer.add_scalar("train/train_final_soc",    train_final_soc,  server_round)
            self.tb_writer.add_scalar("train/mean_energy_kwh",    train_energy_kwh, server_round)

        if COLLECTIVE_SCENARIO_EPISODIC:
            agent_energies = {}
            for reply in replies:
                metrics = reply.content.get("metrics", MetricRecord({}))
                agent_id = int(metrics.get("agent_id", -1))
                energy = float(metrics.get("mean_energy_kwh", 0.0))
                if agent_id >= 0:
                    agent_energies[agent_id] = energy

            total_mean_energy = sum(agent_energies.values())
            print(f"[Episodic] Round {server_round} — per-agent energy: {agent_energies}, "
                  f"total: {total_mean_energy:.2f} kWh")

            self.tb_writer.add_scalar("train/collective_mean_energy_kwh", total_mean_energy, server_round)
            for aid, energy in agent_energies.items():
                self.tb_writer.add_scalar(f"train/agent_{aid}_energy_kwh", energy, server_round)
            self.tb_writer.flush()

            self._train_config["server_round"] = server_round + 1

        _agg_elapsed = time.perf_counter() - _agg_start
        if round_times:
            n_agents = len(round_times)
            mean_rl = np.mean(rl_times)
            mean_bc = np.mean(bc_times)
            mean_fl = np.mean(fl_overheads)
            mean_round = np.mean(round_times)

            episodes_per_agent = [
                float(reply.content.get("metrics", MetricRecord({})).get("train_episodes", 0))
                for reply in replies
            ]
            total_episodes = sum(episodes_per_agent)

            # Per-tx latency: how long does one blockchain call take on average?
            # (blockchain_time / episodes) per agent, then averaged
            per_tx_latencies = []
            for bc_t, ep_count in zip(bc_times, episodes_per_agent):
                if ep_count > 0 and bc_t > 0.001:
                    per_tx_latencies.append(bc_t / ep_count * 1000)  # in ms
            mean_tx_latency_ms = np.mean(per_tx_latencies) if per_tx_latencies else 0.0

            self.tb_writer.add_scalar("throughput/mean_round_time_s",   mean_round,    server_round)
            self.tb_writer.add_scalar("throughput/mean_rl_time_s",      mean_rl,       server_round)
            self.tb_writer.add_scalar("throughput/mean_blockchain_time_s", mean_bc,     server_round)
            self.tb_writer.add_scalar("throughput/mean_fl_overhead_s",  mean_fl,       server_round)
            self.tb_writer.add_scalar("throughput/aggregation_time_s",  _agg_elapsed,  server_round)
            self.tb_writer.add_scalar("throughput/blockchain_pct",
                                      (mean_bc / mean_round * 100) if mean_round > 0 else 0, server_round)
            self.tb_writer.add_scalar("throughput/rl_pct",
                                      (mean_rl / mean_round * 100) if mean_round > 0 else 0, server_round)
            # ── Window-based TPS (independent of chain timestamps) ──
            total_blockchain_tx = sum(agent_tx_counts)
            if first_tx_walls and last_tx_walls:
                blockchain_window_s = max(last_tx_walls) - min(first_tx_walls)
            else:
                blockchain_window_s = 0.0
            if round_wall_starts and round_wall_ends:
                round_window_s = max(round_wall_ends) - min(round_wall_starts)
            else:
                round_window_s = 0.0
            blockchain_window_tps = (
                total_blockchain_tx / blockchain_window_s if blockchain_window_s > 0 else 0.0
            )
            system_window_tps = (
                total_episodes / round_window_s if round_window_s > 0 else 0.0
            )

            self.tb_writer.add_scalar("throughput/mean_tx_latency_ms",  mean_tx_latency_ms, server_round)
            self.tb_writer.add_scalar("throughput/total_episodes",      total_episodes,     server_round)
            self.tb_writer.add_scalar("throughput/n_agents",            n_agents,      server_round)
            self.tb_writer.add_scalar("throughput/total_blockchain_tx", total_blockchain_tx, server_round)
            self.tb_writer.add_scalar("throughput/blockchain_window_s", blockchain_window_s, server_round)
            self.tb_writer.add_scalar("throughput/blockchain_window_tps", blockchain_window_tps, server_round)
            self.tb_writer.add_scalar("throughput/round_window_s",      round_window_s,    server_round)
            self.tb_writer.add_scalar("throughput/system_window_tps",   system_window_tps, server_round)
            self.tb_writer.flush()

            rl_pct = (mean_rl / mean_round * 100) if mean_round > 0 else 0
            bc_pct = (mean_bc / mean_round * 100) if mean_round > 0 else 0

            with open(self._throughput_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    server_round, n_agents, f"{mean_rl:.4f}", f"{mean_bc:.4f}",
                    f"{mean_fl:.4f}", f"{_agg_elapsed:.4f}", f"{rl_pct:.2f}",
                    f"{bc_pct:.2f}", f"{mean_tx_latency_ms:.2f}", f"{total_episodes:.0f}",
                    total_blockchain_tx, f"{blockchain_window_s:.4f}",
                    f"{blockchain_window_tps:.4f}",
                    f"{round_window_s:.4f}", f"{system_window_tps:.4f}",
                ])

            with open(self._concurrency_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for (aid, pid, rws, rwe, ftw, ltw, ntx, rss) in concurrency_rows:
                    writer.writerow([
                        server_round, aid, pid,
                        f"{rws:.6f}", f"{rwe:.6f}",
                        f"{ftw:.6f}", f"{ltw:.6f}",
                        ntx, f"{rss:.2f}",
                    ])

        return arrays, aggregated_metrics
    
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    batch_size: int = context.run_config["batch-size"]
    n_steps: int = context.run_config["n-steps"]

    model_type = context.run_config.get("model", "ppo").lower()

    if model_type == "ppo":
        global_agent = PPO("MlpPolicy", 
                           SmartChargingEnv(), 
                           learning_rate=lr, 
                           batch_size=batch_size,
                           n_steps=n_steps,
                           policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
                           device="cpu")
    else:
        raise ValueError(f"Model {model_type} not supported!")

    # Resume from a saved model, otherwise start fresh
    RESUME_PATH = None
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        saved = torch.load(RESUME_PATH, map_location="cpu", weights_only=True)
        global_agent.policy.load_state_dict(saved, strict=True)
        print(f"[Server] Resumed from {RESUME_PATH}")

    state_dict = global_agent.policy.state_dict()
    initial_arrays = ArrayRecord.from_torch_state_dict(state_dict)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    train_config = ConfigRecord({
        "lr":                     lr,
        "model":                  model_type,
        "server_round":           0,
        "run_id":                 run_id,
    })

    # Initialize FedAvg strategy
    strategy = CustomFedAvg(
        train_config=train_config,
        fraction_train=fraction_train,
        run_id=run_id,
    )

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=initial_arrays,
        train_config=train_config,
        num_rounds=num_rounds,
    )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    if SMART_CONTRACT_AWARE:
        if COLLECTIVE_SCENARIO_EPISODIC:
            torch.save(state_dict, "final_model_collective_contract_aware_episodic.pt")
            print("\nSuccessfully saved to final_model_collective_contract_aware_episodic.pt.")
        elif PARETO_REWARD:
            torch.save(state_dict, "final_model_single_contract_aware_pareto.pt")
            print("\nSuccessfully saved to final_model_single_contract_aware_pareto.pt")
        else:
            torch.save(state_dict, "final_model_single_contract_aware.pt")
            print("\nSuccessfully saved to final_model_single_contract_aware.pt")

    else:
        if COLLECTIVE_SCENARIO_EPISODIC:
            torch.save(state_dict, "final_model_collective_episodic_not_contract_aware.pt")
            print("\nSuccessfully saved to final_model_collective_episodic_not_contract_aware.pt.")
        elif PARETO_REWARD:
            torch.save(state_dict, "final_model_single_not_contract_aware_pareto.pt")
            print("\nSuccessfully saved to final_model_single_not_contract_aware_pareto.pt")
        else:
            torch.save(state_dict, "final_model_not_contract_aware.pt")
            print("\nSuccessfully saved to final_model_not_contract_aware.pt")

