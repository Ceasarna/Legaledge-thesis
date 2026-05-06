"""FLWR-TD3: A Flower / PyTorch app."""

import os
import time
import torch
import numpy as np

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.common import ConfigRecord
from flwr.clientapp import ClientApp

from src.federated.task import train_collective_episodic_task
from src.federated.task import train as train_fn
from src.governance.contract_bridge import (
    ContractBridge,
    CheckpointGovernanceBridge,
)
from src.federated.config import SMART_CONTRACT_ADDRESS, COLLECTIVE_SCENARIO_EPISODIC, SMART_CONTRACT_AWARE

from stable_baselines3 import PPO
from src.envs.charging_env import SmartChargingEnv

EV_ROSTER = ["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"]

def _get_ev_model(partition_id: int) -> str:
    """Deterministic EV assignment from Flower's partition-id (0, 1, 2, 3)."""
    return EV_ROSTER[partition_id % len(EV_ROSTER)]

# Flower ClientApp
app = ClientApp()

def _get_bridge(agent_id: int = 0):
    if SMART_CONTRACT_ADDRESS is None or not SMART_CONTRACT_AWARE:
        return None
    try:
        return ContractBridge(SMART_CONTRACT_ADDRESS, agent_id=agent_id)
    except Exception as e:
        print(f"[ContractBridge] Could not connect: {e}")
        return None

def _get_checkpoint_bridge(agent_id: int):
    """Connect to the CheckpointGovernance contract for episodic collective."""
    if SMART_CONTRACT_ADDRESS is None or not COLLECTIVE_SCENARIO_EPISODIC:
        return None
    try:
        return CheckpointGovernanceBridge(SMART_CONTRACT_ADDRESS, agent_id=agent_id)
    except Exception as e:
        print(f"[CheckpointBridge] Could not connect: {e}")
        return None

# Individual Scenario
def train_single_scenario(msg: Message, context: Context):
    _round_start = time.perf_counter()
    _round_wall_start = time.time()
    lr = context.run_config.get("lr", 3e-4)
    local_epochs = context.run_config["local-epochs"]
    n_step = int(context.run_config.get("n-steps", 2400))
    batch_size = int(context.run_config.get("batch-size", 240))
    model_type = context.run_config.get("model", "ppo").lower()
    agent_id = int(context.node_config["partition-id"])

    bridge = _get_bridge(agent_id)

    train_cfg = msg.content.get("config", ConfigRecord({}))
    server_round = int(train_cfg.get("server_round", 1))
    run_id = str(train_cfg.get("run_id", ""))

    ev_model_name = _get_ev_model(agent_id)
    print(f"[Agent {agent_id}] EV: {ev_model_name}")

    env = SmartChargingEnv(rl_model="ppo",
                           data_source=(2024),
                           bridge=bridge,
                           model_name=ev_model_name,
                           random_duration=False,
                           random_arrival=True)

    neural_network_size_ppo = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    contract_tag = "contract" if SMART_CONTRACT_AWARE else "no_contract"
    tb_log_dir = (
        f"./logs/benchmark/federated/{contract_tag}/{run_id}/agent_{agent_id}"
        if run_id else
        f"./logs/benchmark/federated/{contract_tag}/agent_{agent_id}"
    )

    if model_type == "ppo":
        model = PPO("MlpPolicy",
                    env,
                    device="cpu",
                    verbose=0,
                    learning_rate=lr,
                    batch_size=batch_size,
                    n_steps=n_step,
                    tensorboard_log=tb_log_dir,
                    policy_kwargs=neural_network_size_ppo)
    else:
        raise ValueError(f"Model {model_type} not supported!")
    
    arrays_record = msg.content["arrays"]
    model_state_dict = arrays_record.to_torch_state_dict()
    model.policy.load_state_dict(model_state_dict, strict=True)


    if "train_state" not in context.state:
        context.state["train_state"] = ConfigRecord({"num_timesteps": 0, "num_episodes": 0})
    model.num_timesteps = int(context.state["train_state"]["num_timesteps"])
    episode_offset = int(context.state["train_state"]["num_episodes"])

    train_metrics = train_fn(
        model,
        local_epochs,
        bridge=bridge,
        agent_id=agent_id,
        server_round=server_round,
        episode_offset=episode_offset,
        tb_log_dir=tb_log_dir,
        federated = True,
    )

    # Pack weights to return
    updated_state_dict = model.policy.state_dict()
    new_model_record = ArrayRecord.from_torch_state_dict(updated_state_dict)

    _round_elapsed = time.perf_counter() - _round_start
    _round_wall_end = time.time()
    _rl_time = train_metrics.get("total_rl_time_s", 0.0)
    _bc_time = train_metrics.get("total_blockchain_time_s", 0.0)
    _fl_overhead = max(0.0, _round_elapsed - _rl_time - _bc_time)

    metrics_dict = {
        "train_reward":           train_metrics.get("train_reward", 0.0),
        "train_min":              train_metrics.get("train_min", 0.0),
        "train_max":              train_metrics.get("train_max", 0.0),
        "train_episodes":         train_metrics.get("train_episodes", 0),
        "mean_contract_penalty":  train_metrics.get("mean_contract_penalty", 0.0),
        "compliance_rate":         train_metrics.get("compliance_rate", 1.0),
        "total_energy_kwh":       train_metrics.get("total_energy_kwh", 0.0),
        "num-examples":           local_epochs,
        "round_time_s":           _round_elapsed,
        "rl_time_s":              _rl_time,
        "blockchain_time_s":      _bc_time,
        "fl_overhead_s":          _fl_overhead,
    }
    metric_record = MetricRecord(metrics_dict)

    content = RecordDict({"arrays": new_model_record, "metrics": metric_record})
    return content

# Episodic collective — checkpoint-based - Collective Governance Scenario
def train_collective_scenario_episodic(msg: Message, context: Context):
    _round_start = time.perf_counter()
    _round_wall_start = time.time()
    lr = context.run_config.get("lr", 3e-4)
    local_epochs = context.run_config["local-epochs"]
    n_step = int(context.run_config.get("n-steps", 2400))
    batch_size = int(context.run_config.get("batch-size", 240))
    model_type = context.run_config.get("model", "ppo").lower()
    agent_id = int(context.node_config["partition-id"])
 
    train_cfg = msg.content.get("config", ConfigRecord({}))
    server_round = int(train_cfg.get("server_round", 1))
    run_id = str(train_cfg.get("run_id", ""))
 
    # Use the checkpoint bridge (not the old episodic bridge)
    if SMART_CONTRACT_AWARE:
        bridge = _get_checkpoint_bridge(agent_id)
    else: 
        bridge = None
        
    ev_model_name = _get_ev_model(agent_id)
    print(f"[Agent {agent_id}] EV: {ev_model_name} | Checkpoint bridge: {'connected' if bridge else 'none'}")
 
    # Compute per-agent budget from the contract capacity
    if bridge is not None:
        total_capacity_kwh = bridge.max_capacity_scaled / 1000.0  # SCALE=1000
        per_agent_budget = total_capacity_kwh / bridge.num_agents
    else:
        per_agent_budget = 25.0
 
    env = SmartChargingEnv(
        rl_model="ppo", data_source=2024, bridge=bridge,
        model_name=ev_model_name, random_duration=False, random_arrival=True,
        agent_id=agent_id, round_id=server_round,
        energy_budget_kwh=per_agent_budget,
    )
 
 
    neural_network_size_ppo = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    contract_tag = "checkpoint_collective"
    tb_log_dir = (
        f"./logs/benchmark/federated/{contract_tag}/{run_id}/agent_{agent_id}"
        if run_id else f"./logs/benchmark/federated/{contract_tag}/agent_{agent_id}"
    )
 
    model = PPO(
        "MlpPolicy", env, policy_kwargs=neural_network_size_ppo,
        learning_rate=lr, batch_size=batch_size, n_steps=n_step,
        device="cpu", verbose=0, tensorboard_log=tb_log_dir,
    )
 
    arrays_record = msg.content["arrays"]
    model.policy.load_state_dict(arrays_record.to_torch_state_dict(), strict=True)
 
    if "train_state" not in context.state:
        context.state["train_state"] = ConfigRecord({"num_timesteps": 0, "num_episodes": 0})
    model.num_timesteps = int(context.state["train_state"]["num_timesteps"])
    episode_offset = int(context.state["train_state"]["num_episodes"])
 
    train_metrics = train_collective_episodic_task(
        model, local_epochs, bridge=bridge, agent_id=agent_id,
        server_round=server_round, episode_offset=episode_offset,
        tb_log_dir=tb_log_dir, federated=True,
    )
 
    updated_state_dict = model.policy.state_dict()
    new_model_record = ArrayRecord.from_torch_state_dict(updated_state_dict)
 
    _round_elapsed = time.perf_counter() - _round_start
    _round_wall_end = time.time()
    metrics_dict = {
        "num-examples": local_epochs,
        "train_reward": train_metrics.get("train_reward", 0.0),
        "train_min": train_metrics.get("train_min", 0.0),
        "train_max": train_metrics.get("train_max", 0.0),
        "train_episodes": train_metrics.get("train_episodes", 0),
        "compliance_rate": train_metrics.get("compliance_rate", 1.0),
        "mean_energy_kwh": train_metrics.get("mean_energy_kwh", 0.0),
        "round_time_s": _round_elapsed,
    }
    metric_record = MetricRecord(metrics_dict)
    return RecordDict({"arrays": new_model_record, "metrics": metric_record})


@app.train()

def train(msg: Message, context: Context):
    if COLLECTIVE_SCENARIO_EPISODIC:
        content = train_collective_scenario_episodic(msg, context)
    else:
        content = train_single_scenario(msg, context)

    return Message(content=content, reply_to=msg)

# Flower framework evaluate function. However, external evaluation is done. This function is not used in data collecion.
@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the global model on the local environment."""
    model_type = context.run_config.get("model", "ppo").lower()
    agent_id = int(context.node_config["partition-id"])
    n_eval_episodes = int(context.run_config.get("n-eval-episodes", 5))

    ev_model_name = _get_ev_model(agent_id)

    eval_env = SmartChargingEnv(
        rl_model=model_type,
        data_source=2024,
        model_name=ev_model_name,
        random_duration=False,
        random_arrival=True,
    )

    neural_network_size_ppo = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if model_type == "ppo":
        model = PPO("MlpPolicy", eval_env, verbose=0,
                    policy_kwargs=neural_network_size_ppo, device="cpu")
    else:
        raise ValueError(f"Model {model_type} not supported for evaluation.")

    arrays_record = msg.content["arrays"]
    model_state_dict = arrays_record.to_torch_state_dict()
    model.policy.load_state_dict(model_state_dict, strict=True)

    # Eval uses the single-agent ContractBridge only when NOT in collective mode
    if SMART_CONTRACT_AWARE and not COLLECTIVE_SCENARIO_EPISODIC:
        bridge = _get_bridge(agent_id)
    else:
        bridge = None

    rewards, final_socs, energy_kwhs = [], [], []
    compliances, contract_compliances = [], []

    for _ in range(n_eval_episodes):
        obs, _ = eval_env.reset()
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated

        rewards.append(ep_reward)
        final_socs.append(info.get("final_soc", 0.0))
        power_log = info.get("power_log", {})
        energy_kwhs.append(sum(power_log.values()))

        # Local compliance check (peak hours 17-20)
        peak_energy = sum(v for h, v in power_log.items() if 17 <= h <= 20 and v > 0)
        compliances.append(float(peak_energy == 0.0))

        # On-chain compliance when contract bridge is available
        if bridge is not None:
            try:
                result = bridge.record_episode(agent_id, power_log)
                if result is not None:
                    contract_compliances.append(float(result["compliant"]))
            except Exception:
                pass

    metrics = {
        "test_reward":       float(np.mean(rewards)),
        "test_final_soc":    float(np.mean(final_socs)),
        "test_energy_kwh":   float(np.mean(energy_kwhs)),
        "test_compliance":   float(np.mean(compliances)),
        "num-examples":      n_eval_episodes,
    }
    if contract_compliances:
        metrics["test_contract_compliance"] = float(np.mean(contract_compliances))

    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
