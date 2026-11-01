

import importlib
import itertools

import gymnasium as gym
import numpy as np

from ._legacy import kernel as k
from ._legacy.native import UlpBatchedWarmupNative


class LaneNative(UlpBatchedWarmupNative):
    

    def __init__(self, lanes=3, max_dl=None):
        if lanes not in (3, 4, 5):
            raise ValueError("lanes must be 3, 4 or 5")
        self.total_lanes = lanes
        self.max_dl = lanes - 1 if max_dl is None else max_dl
        if not 0 <= self.max_dl < lanes:
            raise ValueError("At least one general lane must remain")
        self.kernel = importlib.import_module(f"cav_dl.envs.kernels.lane{lanes}")
        super().__init__()
        self.lane_num = lanes
        self.action_table = list(itertools.product(range(self.max_dl + 1), repeat=4))
        self.action_space = gym.spaces.Discrete(len(self.action_table))

    def _allocate_road(self):
        roads = super()._allocate_road()
        for road in roads:
            road.lane_num = self.total_lanes
        return roads

    def get_obs(self):
        return self.kernel.observation(
            self._history, self.t, self.action_recording[self.current_step],
            self.actual_input, self.ramp_entry_index,
        )

    def cal_reward(self, state):
        total, density, capacity, safety = self.kernel.reward_components(
            self._history, self.t, state,
        )
        occupancy = max(state[4], state[9], state[14], state[19], state[24], state[29])
        safety = float(np.float32(safety)) if .5 < occupancy <= 1 else 0.
        return float(total), dict(
            reward_density=density, reward_flow=capacity,
            safety_penalty=safety, base_reward=total,
        )

    def _advance(self, count, lanes=None):
        cumulative = np.array([
            self.cumulative_main_input, self.cumulative_ramp_input,
            self.cumulative_main_output,
        ], dtype=float)
        config = np.zeros(120) if lanes is None else np.asarray(lanes, dtype=float)
        t, performed, early, terminated = self.kernel.advance_ulp(
            self._history, self.input_supply, self.input_cav_supply,
            self.ramp_inflow, self.ramp_cav_ratio, self.ramp, self.actual_input,
            self.cav_input, self.ramp_input_actual_ts, self.t, count, config,
            lanes is not None, self.ramp_entry_index, cumulative,
            self.actual_initial_car_num, self.actual_main_input + self.actual_ramp_input,
            self.terminated,
        )
        self.cumulative_main_input, self.cumulative_ramp_input, self.cumulative_main_output = cumulative
        self.terminated = terminated
        self.early_stop_flag = [False] * 120
        self.early_stop_flag[20:99] = list(self._history[t - 1, 20:99, k.EARLY] > 0)
        self.early_stop = early if lanes is not None else self.early_stop
        if t - 1 >= 3920:
            self.stop_penalty = True
        self.t = t
        self._last_written = max(self._last_written, t + 1)
        
        for _ in range(performed):
            self.sys_time += 15 / 3600
        self._inventory_audit()
        self.cav_mass_error_max = max(
            getattr(self, "cav_mass_error_max", 0.), abs(self.latest_c_error),
        )
        if getattr(self, "strict_cav_conservation", False):
            if abs(self.latest_c_error) >= 1e-5:
                raise ValueError(f"CAV mass ledger error: {self.latest_c_error}")
