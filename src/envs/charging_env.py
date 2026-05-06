import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.envs.helper.price_parser import parse_price_data
from src.envs.helper.EV import EV_MODELS

from src.federated.config import COLLECTIVE_SCENARIO_EPISODIC, PARETO_REWARD

PRICE_FORECAST_HORIZON = 6
DEGRADATION_COST_PER_KWH = 0.05
CC_CV_THRESHOLD_SOC = 0.80

# How often to submit a checkpoint to the smart contract (every N timesteps)
CHECKPOINT_INTERVAL = 12


class SmartChargingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data_source: int = 2024,
        model_name: list = ["CustomEV"],
        rl_model: str = "ppo",
        random_duration: bool = False,
        random_arrival: bool = False,

        # Multi-objective weights
        w_price: float = 0.2,
        w_soc: float = 0.6,
        w_compliance: float = 0.2,

        # Contract bridge
        bridge=None,
        agent_id: int = 0,
        round_id: int = 0,

        # Server_Round (legacy)
        last_round_penalized: int = 0,
        my_last_energy_share: float = 0,
        compliant: bool = False,

        # Collective budget (capacity / num_agents)
        energy_budget_kwh: float = 25.0,
    ):
        super().__init__()
        self.counter = 0

        self.data_source = data_source
        self.market_prices = parse_price_data(data_source)
        self.MAX_PRICE = float(self.market_prices["SE3 Price (EUR)"].max())
        self.MIN_PRICE = float(self.market_prices["SE3 Price (EUR)"].min())
        self.PRICE_RANGE = max(self.MAX_PRICE - self.MIN_PRICE, 1e-8)
        self._all_prices = self.market_prices["SE3 Price (EUR)"].to_numpy(dtype=np.float32)
        self._all_hours = self.market_prices["Delivery Start (CET)"].dt.hour.to_numpy(dtype=np.int32)

        if isinstance(model_name, str):
            model_name = [model_name]
        for name in model_name:
            if name not in EV_MODELS:
                raise ValueError(f"Unknown model '{name}'. Available: {list(EV_MODELS.keys())}")
        self.model_names = model_name

        self.ev = EV_MODELS[self.model_names[0]](ev_id="EV_1")
        self.rl_model = rl_model
        self.steps_per_day = 24
        self.random_duration = random_duration
        self.random_arrival = random_arrival

        self.total_charge = 0.0

        # Multi-objective weights
        self.w_price = w_price
        self.w_soc = w_soc
        self.w_compliance = w_compliance

        # Contract bridge
        self.bridge = bridge
        self.agent_id = agent_id
        self.round_id = round_id
        self.episode_slot = 0

        # Peak-hour window — fetched from contract on reset when bridge is available,
        # otherwise falls back to defaults matching the contract's initial values.
        self.peak_hour_start = 17
        self.peak_hour_end = 20

        # Checkpoint-based collective tracking
        self.energy_budget_kwh = energy_budget_kwh
        self.collective_utilization = 0.0
        self.contribution_ratio = 0.0
        self.checkpoint_interval = CHECKPOINT_INTERVAL

        # Legacy fields
        self.last_round_penalized = last_round_penalized
        self.my_last_energy_share = my_last_energy_share
        self.my_last_total_energy = 0.0
        self.compliant = compliant

        # Action
        if self.rl_model in ["ppo", "sac", "td3"]:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(1,), dtype=np.float32
            )
        elif self.rl_model == "dqn":
            self.action_space = spaces.Discrete(3)
        else:
            raise ValueError(f"RL model '{self.rl_model}' not supported.")

        # Observation spaces
        if COLLECTIVE_SCENARIO_EPISODIC:
            # [price, soc_delta, time_left, hour, budget_usage, collective_util, own_battery_util] + forecast
            low = np.array(
                [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0] + [0.0] * PRICE_FORECAST_HORIZON,
                dtype=np.float32)
            high = np.array(
                [1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0] + [1.0] * PRICE_FORECAST_HORIZON,
                dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        elif PARETO_REWARD:
            TOTAL_WEIGHTS = 3
            low = np.array([0.0, -1.0, 0.0, 0.0] + [0.0] * TOTAL_WEIGHTS + [0.0] * PRICE_FORECAST_HORIZON, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * TOTAL_WEIGHTS + [1.0] * PRICE_FORECAST_HORIZON, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            low = np.array([0.0, -1.0, 0.0, 0.0] + [0.0] * PRICE_FORECAST_HORIZON, dtype=np.float32)
            high = np.array([1.0, 1.0, 1.0, 1.0] + [1.0] * PRICE_FORECAST_HORIZON, dtype=np.float32)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.counter += 1

        if len(self.model_names) > 1:
            chosen = self.model_names[self.np_random.integers(len(self.model_names))]
            self.ev = EV_MODELS[chosen](ev_id="EV_1")
        else:
            self.ev = EV_MODELS[self.model_names[0]](ev_id="EV_1")

        if self.random_duration:
            session_steps = int(np.clip(self.np_random.normal(12.0, 4.0), 4.0, 24.0))
        else:
            session_steps = 24

        initial_soc = float(self.np_random.uniform(0.1, 0.4))
        target_soc = 0.80

        self.total_charge = 0.0
        self.collective_utilization = 0.0
        self.contribution_ratio = 0.0
        self.own_battery_util = 0.0
        self.current_step = 0

        duration_needed = session_steps + PRICE_FORECAST_HORIZON + 1

        if self.random_arrival:
            arrival = int(self.np_random.integers(0, 24))
            valid_starts = np.where(
                (self._all_hours == arrival) &
                (np.arange(len(self._all_prices)) <= len(self._all_prices) - duration_needed)
            )[0]
        else:
            valid_starts = np.where(
                (self._all_hours == 0) &
                (np.arange(len(self._all_prices)) <= len(self._all_prices) - duration_needed)
            )[0]

        start_idx = int(self.np_random.choice(valid_starts))
        self._price_array = self._all_prices[start_idx: start_idx + duration_needed]
        self._hour_array = self._all_hours[start_idx: start_idx + duration_needed]

        self.arrival_time = int(self._hour_array[0])
        departure_time = self.arrival_time + session_steps

        self.ev.reset_session(self.arrival_time, departure_time, initial_soc, target_soc)
        total_duration = self.ev.get_time_remaining(self.arrival_time)

        self.price_window_max = float(self._price_array.max())
        self.price_window_min = float(self._price_array.min())
        self.current_price = self._get_price()
        self.power_log: dict = {}

        self._min_soc_trajectory = self._build_min_soc_trajectory(total_duration)

        # Logging
        self.total_price_reward = 0.0
        self.total_soc_reward = 0.0
        self.total_compliance_reward = 0.0

        if PARETO_REWARD:
            weights = self.np_random.dirichlet((1.0, 1.0, 1.0))
            self.w_price = weights[0]
            self.w_soc = weights[1]
            self.w_compliance = weights[2]


        if self.bridge is not None and hasattr(self.bridge, "get_peak_hours"):
            try:
                self.peak_hour_start, self.peak_hour_end = self.bridge.get_peak_hours()
            except Exception as e:
                print(f"[Env] Could not fetch peak hours from contract: {e}")

        if PARETO_REWARD:
            return self._get_observation_pareto(), {}

        return self._get_observation(), {}

    def step(self, action):
        max_power_kw = self.ev.max_power_kw

        requested_power_kw = float(action[0]) * max_power_kw

        current_price = self._get_price()
        normalized_current_price = self._norm_price(current_price)

        actual_energy_kwh = self.ev.transfer_energy(requested_power_kw)
        self.total_charge += actual_energy_kwh

        current_slot = self._get_hour()
        self.power_log[current_slot] = actual_energy_kwh

        self.current_step += 1
        duration = self.ev.departure_time - self.ev.arrival_time
        terminated = bool(self.current_step >= duration)

        # Info
        info = {"actual_energy_kwh": actual_energy_kwh,
                "price_eur_mwh": current_price}

        # Checkpoint submission - Collective Governance
        if COLLECTIVE_SCENARIO_EPISODIC and self.bridge is not None:
            if self.current_step % self.checkpoint_interval == 0 or terminated:
                checkpoint_id = self.current_step // self.checkpoint_interval
                own_submitted = max(0.0, self.total_charge)
                try:
                    self.collective_utilization, self.contribution_ratio = self.bridge.submit_checkpoint(
                        self.round_id, self.episode_slot, checkpoint_id,
                        self.agent_id, own_submitted
                    )
                    
                    info["checkpoint_triggered"] = True
                    info["checkpoint_id"] = checkpoint_id
                    info["own_submitted_energy"] = own_submitted
                    info["blockchain_collective_total"] = self.collective_utilization
                    info["contribution_ratio"] = self.contribution_ratio    
                except Exception as e:
                    print(f"[Checkpoint] Error: {e}")
        
        # Reward
        if PARETO_REWARD:
            reward = self.reward_function_pareto(
                actual_energy_kwh, normalized_current_price, terminated)
        elif COLLECTIVE_SCENARIO_EPISODIC:
            reward = self.reward_function_collaborative_episodic(
                actual_energy_kwh, normalized_current_price, terminated)
        else:
            reward = self.reward_function_collaborative(
                actual_energy_kwh, normalized_current_price, terminated)

        
        if terminated:
            info["power_log"] = self.power_log
            info["final_soc"] = self.ev.get_soc()
            info["target_soc"] = self.ev.get_target_soc()
            info["ev_class"] = type(self.ev).__name__
            info["total_charge"] = self.total_charge
            info["collective_utilization"] = self.collective_utilization
            info["reward_price"] = self.total_price_reward
            info["reward_soc"] = self.total_soc_reward
            info["reward_compliance"] = self.total_compliance_reward

        if PARETO_REWARD:
            obs = self._get_observation_pareto()
        else:
            obs = self._get_observation()
        return obs, float(reward), terminated, False, info

    # Reward: Episodic Collective - Collective Governance
    def reward_function_collaborative_episodic(self, actual_energy_kwh, normalized_current_price, terminated):
        BUDGET_PENALTY_SCALE = 1000.0

        power_frac = actual_energy_kwh / self.ev.max_power_kw

        # 1. Price objective
        reward = -power_frac * normalized_current_price
        self.total_price_reward += reward

        # 2. SoC terminal penalty
        if terminated:
            deficit = max(0.0, self.ev.get_target_soc() - self.ev.get_soc()) / 100.0
            soc_pen = -(deficit ** 2) * 50
            reward += soc_pen
            self.total_soc_reward += soc_pen

        # 3. Collective budget penalty — based on live utilization from contract
        if power_frac > 0 and self.current_step % self.checkpoint_interval == 0:
            my_ratio  = self.contribution_ratio
            if self.collective_utilization > 1.0:
                overage = self.collective_utilization - 1.0
                pen = overage * my_ratio * BUDGET_PENALTY_SCALE
                reward -= pen

                self.total_compliance_reward += pen
            
        return reward

    # Reward: Pareto - Used for both Individual Scenario and Pareto Scenario
    def reward_function_pareto(self, actual_energy_kwh, normalized_current_price, terminated):

        # 1. Price objective
        power_frac = actual_energy_kwh / self.ev.max_power_kw
        price_reward = -power_frac * normalized_current_price
        self.total_price_reward += price_reward

        # 2. SoC — terminal
        soc_reward = 0.0
        if terminated:
            deficit = max(0.0, self.ev.get_target_soc() - self.ev.get_soc()) / 100.0
            soc_reward = -(deficit ** 2) * 15.0

        # 3. Compliance
        compliance_reward = 0.0
        is_peak = self.peak_hour_start <= self._get_hour() <= self.peak_hour_end
        if is_peak and power_frac > 0:
            compliance_reward = -(power_frac ** 2) * 10.0
            if power_frac > 0.02:
                compliance_reward -= 1.0

        if PARETO_REWARD:
            reward = (
                self.w_price * price_reward +
                self.w_soc * soc_reward +
                self.w_compliance * compliance_reward
            )
        else:
            reward = price_reward + soc_reward + compliance_reward
        return reward



    # Observations 
    def _get_observation(self):
        current_price = self._get_price()
        normalized_current_price = self._norm_price(current_price)
        soc_delta = (self.ev.get_soc() - self.ev.get_target_soc()) / 100.0
        session_len = self.ev.departure_time - self.ev.arrival_time
        time_left = float(
            self.ev.get_time_remaining(self.arrival_time + self.current_step)
        ) / max(session_len, 1)
        current_hour = self._get_hour()

        forecast = np.array([
            self._norm_price(self._get_price(self.current_step + h + 1))
            for h in range(PRICE_FORECAST_HORIZON)
        ], dtype=np.float32)


        if COLLECTIVE_SCENARIO_EPISODIC:
            
            collective_util = np.clip(self.collective_utilization, 0.0, 2.0)
            contribution_ratio = np.clip(self.contribution_ratio, 0.0, 1.0)
            own_battery_util = np.clip(self.total_charge / self.ev.battery_capacity_kwh, 0.0, 1.0)

            obs = np.concatenate([
            np.array([
                normalized_current_price,
                soc_delta,
                time_left,
                current_hour / 23,
                contribution_ratio,     # "Am I the one stressing the grid?"
                collective_util,        # "How stressed is the grid?"
                own_battery_util,       # "How much of my own capacity have I used?"
            ], dtype=np.float32),
            forecast
        ])
            return obs

        else:
            obs = np.concatenate([
                np.array([
                    normalized_current_price, soc_delta, time_left,
                    current_hour / 23,
                ], dtype=np.float32),
                forecast,
            ])
        return obs

    def _get_observation_pareto(self):
        current_price = self._get_price()
        normalized_current_price = self._norm_price(current_price)
        soc_delta = (self.ev.get_soc() - self.ev.get_target_soc()) / 100.0
        session_len = self.ev.departure_time - self.ev.arrival_time
        time_left = float(
            self.ev.get_time_remaining(self.arrival_time + self.current_step)
        ) / max(session_len, 1)
        current_hour = self._get_hour()

        forecast = np.array([
            self._norm_price(self._get_price(self.current_step + h + 1))
            for h in range(PRICE_FORECAST_HORIZON)
        ], dtype=np.float32)

        obs = np.concatenate([
            np.array([
                normalized_current_price, soc_delta, time_left,
                current_hour / 23,
                self.w_price, self.w_soc, self.w_compliance,
            ], dtype=np.float32),
            forecast,
        ])
        return obs

    # Helpers 
    def _get_hour(self, current_step: int = -1) -> int:
        idx = int(np.clip(
            self.current_step if current_step == -1 else current_step,
            0, len(self._hour_array) - 1))
        return int(self._hour_array[idx])

    def _get_price(self, current_step: int = -1) -> float:
        if current_step == -1:
            idx = np.clip(self.current_step, 0, len(self._price_array) - 1)
            return float(self._price_array[idx])
        idx = np.clip(current_step, 0, len(self._price_array) - 1)
        return float(self._price_array[idx])

    def _build_min_soc_trajectory(self, total_duration: int, ramp_hours: int = 6) -> np.ndarray:
        initial_soc = self.ev.get_soc()
        target_soc = self.ev.get_target_soc()
        flat_steps = max(0, total_duration - ramp_hours)
        flat_part = np.full(flat_steps, initial_soc)
        ramp_part = np.linspace(initial_soc, target_soc, total_duration - flat_steps + 1)
        return np.concatenate([flat_part, ramp_part])

    def _norm_price(self, price: float) -> float:
        return (price - self.price_window_min) / (self.price_window_max - self.price_window_min + 1e-8)