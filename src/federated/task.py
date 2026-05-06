"""FLWR-TD3: A Flower / PyTorch app."""
import time
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter
import csv
import os

# This is only for Single-Scenario
class TrainCallback(BaseCallback):
    def __init__(
        self,
        bridge=None,
        agent_id: int = 0,
        verbose=0,
        non_compliance_penalty: float = -1.0,
        server_round: int = 1,
        episode_offset: int = 0,
        tb_log_dir: str = None,
        federated: bool = False,

    ):
        super().__init__(verbose)
        self.bridge = bridge
        self.agent_id = agent_id
        self.server_round = server_round
        self.episode_offset = episode_offset

        if self.bridge is None:
            if federated:
                log_dir = tb_log_dir or f"./logs/benchmark/federated/agent_{agent_id}"
            else:
                log_dir = tb_log_dir or f"./logs/benchmark/agent_{agent_id}"
        else:
            if federated:
                log_dir = tb_log_dir or f"./logs/benchmark/federated/agent_{agent_id}_contract_aware"
            else:
                log_dir = tb_log_dir or f"./logs/benchmark/agent_{agent_id}_contract_aware"
        self.tb_writer = SummaryWriter(log_dir=log_dir)
        self.episode_rewards = []
        self.current_reward = 0.0

        # Price metrics
        self.episode_price_costs = []
        self.current_price_cost = 0.0

        self.episode_penalties = []
        self.episode_compliance = []
        self.episode_gas = []
        self.total_kwh = []
        self.latency_ms = []

        self.total_energy_kwh = 0.0
        self.peak_charge_counts = []
        self.offpeak_charge_counts = []
        self.episode_final_soc = []
        self.episode_soc_target_delta = []
        self.total_charge_counts = []

        # Logging for reward components
        self.total_price_reward_counts = []
        self.total_soc_reward_counts = []
        self.total_compliance_reward_counts = []

        # Logging for running compliance_rate
        self.total_compliance_counts = []

        self.non_compliance_penalty = non_compliance_penalty

        # ── Throughput timing ──
        self.total_blockchain_time_s = 0.0
        self.total_rl_time_s = 0.0
        self._episode_rl_start = time.perf_counter()

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]

        # Price Metrics
        info = self.locals["infos"][0]
        actual_energy = info.get("actual_energy_kwh", 0.0)
        env = self.training_env.envs[0].unwrapped
        current_price = env._get_price(env.current_step - 1)
        self.current_price_cost += actual_energy * current_price

        if self.locals["dones"][0]:
            power_log  = self.locals["infos"][0].get("power_log", {})
            self.total_energy_kwh += sum(power_log.values())

            # Price Metrics
            self.episode_price_costs.append(self.current_price_cost)
            self.current_price_cost = 0.0

            peak_energy    = sum(v for h, v in power_log.items() if 17 <= h <= 20 and v > 0)
            offpeak_energy = sum(v for h, v in power_log.items() if not (17 <= h <= 20) and v > 0)
            self.peak_charge_counts.append(peak_energy)
            self.offpeak_charge_counts.append(offpeak_energy)

            total_charge = info["total_charge"]
            self.total_charge_counts.append(total_charge)
            
            self.total_price_reward_counts.append(info.get("reward_price", 0.0))
            self.total_soc_reward_counts.append(info.get("reward_soc", 0.0))
            self.total_compliance_reward_counts.append(info.get("reward_compliance", 0.0))

            locally_compliant = peak_energy == 0.0

            ep_info    = self.locals["infos"][0]
            final_soc  = ep_info.get("final_soc")
            target_soc = ep_info.get("target_soc")
            if final_soc is not None and target_soc is not None:
                self.episode_final_soc.append(final_soc)
                self.episode_soc_target_delta.append(final_soc - target_soc)
                

            # Time: RL portion ends, blockchain begins 
            self.total_rl_time_s += time.perf_counter() - self._episode_rl_start
            
            if self.bridge is not None:
                _bc_start = time.perf_counter()
                try:
                    result = self.bridge.record_episode(self.agent_id, power_log)
                    #result = None
                    if result is not None:
                        compliant = result["compliant"]
                        penalty   = 0.0 if compliant else self.non_compliance_penalty
                        self.locals["rewards"][0] += penalty
                        self.current_reward       += penalty
                        self.episode_penalties.append(penalty)
                        self.episode_compliance.append(compliant)
                        self.episode_gas.append(result["gasUsed"])
                        self.total_kwh.append(result["totalEnergy"])
                        self.latency_ms.append(result["latency"])
                except Exception as e:
                    print(f"[ContractBridge] Warning: {e}")
                self.total_blockchain_time_s += time.perf_counter() - _bc_start
            else:
                self.episode_compliance.append(locally_compliant)
            
            if self.bridge is not None:
                # Primal Dual for contract compliance
                CONTRACT_PENALTY = 5.0  # positive magnitude
                
                if peak_energy > 0:
                    env.lambda_peak = min(
                        env.lambda_max,
                        env.lambda_peak + env.lambda_lr * CONTRACT_PENALTY
                    )
                else:
                    env.lambda_peak *= 0.99
            #self.episode_compliance.append(float(locally_compliant))
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0

            global_ep = self.episode_offset + len(self.episode_rewards) - 1
            self.tb_writer.add_scalar("avg100/peak_charge_kwh",    np.mean(self.peak_charge_counts[-100:]),      global_ep)
            self.tb_writer.add_scalar("avg100/offpeak_charge_kwh", np.mean(self.offpeak_charge_counts[-100:]),   global_ep)
            if self.episode_final_soc:
                self.tb_writer.add_scalar("avg100/final_soc",         np.mean(self.episode_final_soc[-100:]),        global_ep)
            if self.episode_compliance:
                self.tb_writer.add_scalar("avg100/compliance_rate", np.mean(self.episode_compliance[-100:]),     global_ep)
            if len(self.episode_rewards) % 100 == 0:
                self.tb_writer.flush()

            # Logging for reward components
            self.tb_writer.add_scalar("avg100/total_price_reward",    np.mean(self.total_price_reward_counts[-100:]),      global_ep)
            self.tb_writer.add_scalar("avg100/total_soc_rward", np.mean(self.total_soc_reward_counts[-100:]),   global_ep)
            self.tb_writer.add_scalar("avg100/total_compliance_reward", np.mean(self.total_compliance_reward_counts[-100:]),   global_ep)

            self.tb_writer.add_scalar("avg100/total_charge", np.mean(self.total_charge_counts[-100:]), global_ep)

            self.tb_writer.add_scalar("avg100/episode_price_cost", np.mean(self.episode_price_costs[-100:]), global_ep)
            if self.episode_price_costs[-1] != 0:
                total_energy_ep = sum(self.locals["infos"][0].get("power_log", {}).values())
                if total_energy_ep > 0:
                    self.tb_writer.add_scalar("avg100/price_per_kwh", 
                        np.mean(self.episode_price_costs[-100:]) / max(total_energy_ep, 1e-8), global_ep)
    
            self._episode_rl_start = time.perf_counter()
        return True

def train(
    model,
    total_timesteps,
    bridge=None,
    agent_id: int = 0,
    non_compliance_penalty: float = -1.0,
    server_round: int = 1,
    episode_offset: int = 0,
    tb_log_dir: str = None,
    federated: bool = False,
):
    callback = TrainCallback(
        bridge=bridge,
        agent_id=agent_id,
        non_compliance_penalty=non_compliance_penalty,
        server_round=server_round,
        episode_offset=episode_offset,
        tb_log_dir=tb_log_dir,
        federated = federated,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=False)
    callback.tb_writer.flush()
    callback.tb_writer.close()

    if callback.episode_rewards:
        mean_reward    = np.mean(callback.episode_rewards)
        min_reward     = np.min(callback.episode_rewards)
        max_reward     = np.max(callback.episode_rewards)
        ep_count       = len(callback.episode_rewards)
        mean_penalty      = np.mean(callback.episode_penalties) if callback.episode_penalties else 0.0
        compliance_rate   = np.mean(callback.episode_compliance) if callback.episode_compliance else 1.0
        mean_gas       = np.mean(callback.episode_gas) if callback.episode_gas else 0.0
        mean_latency   = np.mean(callback.latency_ms) if callback.latency_ms else 0.0
        mean_total_kwh = np.mean(callback.total_kwh) if callback.total_kwh else 0.0
    else:
        mean_reward = min_reward = max_reward = mean_penalty = 0.0
        compliance_rate = 1.0
        ep_count = 0

    return {
        "train_reward":          float(mean_reward),
        "train_min":             float(min_reward),
        "train_max":             float(max_reward),
        "train_episodes":        int(ep_count),
        "mean_contract_penalty": float(mean_penalty),
        "compliance_rate":       float(compliance_rate),
        "total_energy_kwh":      float(callback.total_energy_kwh),
        "mean_gas_used":         float(mean_gas),
        "mean_latency_ms":       float(mean_latency),
        "mean_total_kwh":        float(mean_total_kwh),
        "total_rl_time_s":         float(callback.total_rl_time_s),
        "total_blockchain_time_s": float(callback.total_blockchain_time_s),
    }


# Episodic Collective Callback — checkpoint-based - Collective Governance
class TrainCallbackCollectiveEpisodic(BaseCallback):
    def __init__(self, bridge=None, agent_id=0, verbose=0,
                 non_compliance_penalty=-1.0, server_round=1,
                 episode_offset=0, tb_log_dir=None, federated=False):
        super().__init__(verbose)
        self.bridge = bridge
        self.agent_id = agent_id
        self.server_round = server_round
        self.episode_offset = episode_offset
        self.episode_slot = 0
 
        tag = "contract_aware" if self.bridge else ""
        base = "federated" if federated else ""
        log_dir = tb_log_dir or f"./logs/benchmark/{base}/agent_{agent_id}{'_' + tag if tag else ''}"
        self.tb_writer = SummaryWriter(log_dir=log_dir)
 
        self.episode_rewards = []
        self.current_reward = 0.0
        self.episode_price_costs = []
        self.current_price_cost = 0.0
        self.episode_penalties = []
        self.episode_compliance = []
        self.episode_energy_kwh = []
        self.peak_charge_counts = []
        self.offpeak_charge_counts = []
        self.episode_final_soc = []
        self.total_price_reward_counts = []
        self.total_soc_reward_counts = []
        self.total_compliance_reward_counts = []
        self.non_compliance_penalty = non_compliance_penalty

        os.makedirs(log_dir, exist_ok=True)
        self.csv_path = os.path.join(log_dir, f"agent_{agent_id}_sync.csv")
        
        with open(self.csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["global_step", "episode_slot", "checkpoint_id", "own_energy", "collective_total"])

    def _on_training_start(self) -> None:
            """Syncs the environment's episode ID with the global offset before step 1."""
            env = self.training_env.envs[0].unwrapped

            env.episode_slot = self.episode_offset

    def _on_step(self) -> bool:
        self.current_reward += self.locals["rewards"][0]
        info = self.locals["infos"][0]
        actual_energy = info.get("actual_energy_kwh", 0.0)
        env = self.training_env.envs[0].unwrapped
        current_price = env._get_price(env.current_step - 1)
        self.current_price_cost += actual_energy * current_price

        #  Catch Checkpoint Data Mid-Episode
        if info.get("checkpoint_triggered", False):
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    self.num_timesteps, 
                    self.episode_slot, 
                    info.get("checkpoint_id"), 
                    info.get("own_submitted_energy", 0.0), 
                    info.get("blockchain_collective_total", 0.0),
                    info.get("contribution_ratio", 0.0),

                ])

        if self.locals["dones"][0]:
            total_charge = info["total_charge"]
            self.episode_energy_kwh.append(total_charge)
            self.episode_price_costs.append(self.current_price_cost)
            self.current_price_cost = 0.0
 
            self.total_price_reward_counts.append(info.get("reward_price", 0.0))
            self.total_soc_reward_counts.append(info.get("reward_soc", 0.0))
            self.total_compliance_reward_counts.append(info.get("reward_compliance", 0.0))
 
            final_soc = info.get("final_soc")
            target_soc = info.get("target_soc")
            if final_soc is not None and target_soc is not None:
                self.episode_final_soc.append(final_soc)

            # Finalize episode on contract & determine compliance
            collective_util = info.get("collective_utilization", 0.0)
            locally_compliant = collective_util <= 1.0
            global_ep = self.episode_offset + len(self.episode_rewards) - 1

            if self.bridge is not None:
                try:
                    # Finalize using the last checkpoint
                    session_len = env.ev.departure_time - env.ev.arrival_time
                    final_checkpoint = session_len // env.checkpoint_interval

                    global_ep_id = self.episode_offset + self.episode_slot
                    result = self.bridge.finalize_episode(
                        self.server_round, global_ep_id, final_checkpoint)
                    
                    locally_compliant = not result["penalized"]
 
                    self.tb_writer.add_scalar("contract/penalized",
                        float(result["penalized"]), self.episode_slot*global_ep)
                    self.tb_writer.add_scalar("contract/total_energy_kwh",
                        result["total_energy"], self.episode_slot*global_ep)
                    self.tb_writer.add_scalar("contract/collective_util",
                        collective_util, self.episode_slot*global_ep)
                except Exception as e:
                    print(f"[Agent {self.agent_id}] Finalize error: {e}")
 
                self.episode_slot += 1
                # Also advance the env's episode_slot for the next episode
                env.episode_slot += 1
 
            self.episode_compliance.append(float(locally_compliant))
            self.episode_rewards.append(self.current_reward)
            self.current_reward = 0.0
 
            # ── Logging ──
            if self.episode_final_soc:
                self.tb_writer.add_scalar("avg100/final_soc", np.mean(self.episode_final_soc[-100:]), global_ep)
            if self.episode_compliance:
                self.tb_writer.add_scalar("avg100/compliance_rate", np.mean(self.episode_compliance[-100:]), global_ep)
            self.tb_writer.add_scalar("avg100/episode_energy_kwh", np.mean(self.episode_energy_kwh[-100:]), global_ep)
            self.tb_writer.add_scalar("avg100/total_price_reward", np.mean(self.total_price_reward_counts[-100:]), global_ep)
            self.tb_writer.add_scalar("avg100/total_soc_reward", np.mean(self.total_soc_reward_counts[-100:]), global_ep)
            self.tb_writer.add_scalar("avg100/total_compliance_reward", np.mean(self.total_compliance_reward_counts[-100:]), global_ep)
            self.tb_writer.add_scalar("live/collective_utilization", collective_util, global_ep)
            self.tb_writer.add_scalar("live/budget_usage", total_charge / max(env.energy_budget_kwh, 1e-8), global_ep)
 
            if len(self.episode_rewards) % 100 == 0:
                self.tb_writer.flush()
        return True

def train_collective_episodic_task(
    model,
    total_timesteps,
    bridge=None,
    agent_id: int = 0,
    non_compliance_penalty: float = -1.0,
    server_round: int = 1,
    episode_offset: int = 0,
    tb_log_dir: str = None,
    federated: bool = False,
):
    callback = TrainCallbackCollectiveEpisodic(
        bridge=bridge,
        agent_id=agent_id,
        non_compliance_penalty=non_compliance_penalty,
        server_round=server_round,
        episode_offset=episode_offset,
        tb_log_dir=tb_log_dir,
        federated=federated,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback, reset_num_timesteps=False, progress_bar=True)
    callback.tb_writer.flush()
    callback.tb_writer.close()

    if callback.episode_rewards:
        mean_reward    = np.mean(callback.episode_rewards)
        min_reward     = np.min(callback.episode_rewards)
        max_reward     = np.max(callback.episode_rewards)
        ep_count       = len(callback.episode_rewards)
        mean_penalty      = np.mean(callback.episode_penalties) if callback.episode_penalties else 0.0
        compliance_rate   = np.mean(callback.episode_compliance) if callback.episode_compliance else 1.0
        mean_energy_kwh   = np.mean(callback.episode_energy_kwh) if callback.episode_energy_kwh else 0.0
    else:
        mean_reward = min_reward = max_reward = mean_penalty = 0.0
        compliance_rate = 1.0
        mean_energy_kwh = 0.0
        ep_count = 0

    return {
        "train_reward":          float(mean_reward),
        "train_min":             float(min_reward),
        "train_max":             float(max_reward),
        "train_episodes":        int(ep_count),
        "mean_contract_penalty": float(mean_penalty),
        "compliance_rate":       float(compliance_rate),
        "mean_energy_kwh":       float(mean_energy_kwh),
    }