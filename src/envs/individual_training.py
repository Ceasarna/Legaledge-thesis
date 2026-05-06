import numpy as np
import shap
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env # type: ignore
from src.federated.config import SMART_CONTRACT_ADDRESS
from src.governance.contract_bridge import ContractBridge
from src.envs.charging_env import SmartChargingEnv

from tensorboardX import SummaryWriter
from src.federated.task import train
from src.governance.contract_bridge import ContractBridge # type: ignore

# Centralized training. Mimics the configs of FRL. 
def train_and_test_single_agent_no_contract_non_IID_mimic_FRL(
    contract_address: str = None,
    agent_id: int = 0,
    total_timesteps: int = 1_000_000,
    tb_log_dir: str = "logs/benchmark",
    batch_size: int = 50,
    n_steps: int = 500,
    learning_rate: float = 3e-4,
    smart_contract_aware: bool = False,
):
    
    bridge = None
    if smart_contract_aware and (contract_address or SMART_CONTRACT_ADDRESS):
        try:
            bridge = ContractBridge(contract_address or SMART_CONTRACT_ADDRESS, agent_id=agent_id)
            print(f"[ContractBridge] Connected for agent {agent_id}")
        except Exception as e:
            print(f"[ContractBridge] Could not connect: {e}")

    env = SmartChargingEnv(rl_model="ppo",
                           data_source=(2024),
                           bridge=bridge,
                           model_name=["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"],
                           random_duration=False,
                           random_arrival=True)

    agent_tb_dir = f"{tb_log_dir}/agent_{agent_id}"
    model = PPO("MlpPolicy",
                env,
                device="cpu",
                verbose=0,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                tensorboard_log=agent_tb_dir,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))

    train(model,
          total_timesteps=total_timesteps,
          bridge=bridge,
          agent_id=agent_id,
          tb_log_dir=agent_tb_dir)

    state_dict = model.policy.state_dict()
    model_filename = (
        f"Baseline_RL_contract_non_IID_mimic_FRL_{agent_id}.pt"
        if smart_contract_aware else
        f"Baseline_RL_no_contract_non_IID_mimic_FRL_{agent_id}.pt"
    )
    print(f"Saving model to {model_filename}")
    torch.save(state_dict, model_filename)

    return model, env

# Centralized training. Mimics the config of FRL. Only one EV model present.
def train_and_test_single_agent_no_contract_non_IID_mimic_FRL_oneEV(
    contract_address: str = None,
    agent_id: int = 0,
    total_timesteps: int = 1_000_000,
    tb_log_dir: str = "logs/benchmark",
    batch_size: int = 50,
    n_steps: int = 500,
    learning_rate: float = 3e-4,
    smart_contract_aware: bool = False,
    model_name: str = "CustomEV",
):
    bridge = None
    if smart_contract_aware and (contract_address or SMART_CONTRACT_ADDRESS):
        try:
            bridge = ContractBridge(contract_address or SMART_CONTRACT_ADDRESS, agent_id=agent_id)
            print(f"[ContractBridge] Connected for agent {agent_id}")
        except Exception as e:
            print(f"[ContractBridge] Could not connect: {e}")

    model_list = ["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"]

    env = SmartChargingEnv(rl_model="ppo",
                           data_source=(2024),
                           bridge=bridge,
                           model_name=model_name,
                           random_duration=False,
                           random_arrival=True)

    agent_tb_dir = f"{tb_log_dir}/agent_{agent_id}"
    model = PPO("MlpPolicy",
                env,
                device="cpu",
                verbose=0,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                tensorboard_log=agent_tb_dir,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))

    train(model,
          total_timesteps=total_timesteps,
          bridge=bridge,
          agent_id=agent_id,
          tb_log_dir=agent_tb_dir)

    state_dict = model.policy.state_dict()
    model_filename = (
        f"Baseline_RL_contract_non_IID_mimic_FRL_{model_name}_{agent_id}.pt"
        if smart_contract_aware else
        f"Baseline_RL_no_contract_non_IID_mimic_FRL_{model_name}_{agent_id}.pt"
    )
    print(f"Saving model to {model_filename}")
    torch.save(state_dict, model_filename)

    return model, env



if __name__ == "__main__":
    CONTRACT = SMART_CONTRACT_ADDRESS
    TIMESTEPS = 1_920_000
    # ["Tesla Model 3 LR", "Nissan Leaf", "Renault Zoe", "Volvo XC90 PHEV"],

    models = {0: "Tesla Model 3 LR", 1: "Nissan Leaf", 2: "Renault Zoe", 3: "Volvo XC90 PHEV"}
    models1 = {0: "Volvo XC90 PHEV"}

    train_and_test_single_agent_no_contract_mimic_FRL_flag = False
    train_and_test_single_agent_contract_mimic_FRL_flag = True
    train_and_test_single_agent_contract_mimic_FRL_flag_oneEV = False
    train_and_test_single_agent_contract_mimic_FRL_flag_oneEV_with_contract = False

    env = SmartChargingEnv(rl_model="ppo",
                           data_source=(2024),
                           model_name="Tesla Model 3 LR",
                           random_duration=False,
                           random_arrival=True)

    model = PPO("MlpPolicy",
                env,
                device="cpu",
                verbose=0,
                policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])))

    #plot_single_agent_pareto(model, env)

    if train_and_test_single_agent_no_contract_mimic_FRL_flag:
        train_and_test_single_agent_no_contract_non_IID_mimic_FRL(contract_address=None,
                                                 agent_id=421_1,
                                                 total_timesteps=TIMESTEPS*4,
                                                 tb_log_dir="logs/benchmark",
                                                 batch_size=240,
                                                 n_steps=4800,
                                                 learning_rate=3e-4,
                                                 smart_contract_aware=False
                                                 )

    if train_and_test_single_agent_contract_mimic_FRL_flag:
        train_and_test_single_agent_no_contract_non_IID_mimic_FRL(contract_address=SMART_CONTRACT_ADDRESS,
                                                 agent_id=421_2,
                                                 total_timesteps=TIMESTEPS*4,
                                                 tb_log_dir="logs/benchmark",
                                                 batch_size=240,
                                                 n_steps=4800,
                                                 learning_rate=3e-4,
                                                 smart_contract_aware=True
                                                 )
        
    if not train_and_test_single_agent_contract_mimic_FRL_flag_oneEV:
        for i in range(4):
            print(models[i])
            train_and_test_single_agent_no_contract_non_IID_mimic_FRL_oneEV(contract_address=None,
                                                    agent_id=420_0_0+i,
                                                    total_timesteps=TIMESTEPS,
                                                    tb_log_dir="logs/benchmark",
                                                    batch_size=240,
                                                    n_steps=4800,
                                                    learning_rate=2e-4,
                                                    smart_contract_aware=False,
                                                    model_name=models[i]
                                                    )

    if train_and_test_single_agent_contract_mimic_FRL_flag_oneEV_with_contract:
        for i in range(4):
            print(models[i])
            train_and_test_single_agent_no_contract_non_IID_mimic_FRL_oneEV(contract_address=SMART_CONTRACT_ADDRESS,
                                                    agent_id=420_0_10+i,
                                                    total_timesteps=TIMESTEPS,
                                                    tb_log_dir="logs/benchmark",
                                                    batch_size=240,
                                                    n_steps=4800,
                                                    learning_rate=2e-4,
                                                    smart_contract_aware=True,
                                                    model_name=models[i]
                                                    )
