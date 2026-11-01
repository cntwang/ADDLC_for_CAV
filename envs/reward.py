

import gymnasium as gym

from ._legacy.unified import UnifiedEnv
from .native import LaneNative

DROP = {"NoDensity": 0, "NoCapacity": 1, "NoSafety": 2, "NoTime": 3, "NoEarlyPenalty": 4}
COMPONENTS = ["density", "capacity", "safety", "time", "early"]


class FiveReward(UnifiedEnv):
    def __init__(self, lanes=3, max_dl=None, arm="Full"):
        if arm not in {"Full", *DROP}:
            raise ValueError(f"Unknown five-term reward variant: {arm}")
        self.variant, self.arm = "Full", arm
        gym.Wrapper.__init__(self, LaneNative(lanes, max_dl))

    def step(self, action):
        obs, reference, done, truncated, info = super().step(action)
        values = info["six_reward_components"]
        reward = reference - values[5]
        if self.arm in DROP:
            reward -= values[DROP[self.arm]]
        if done or truncated:
            episode = info["ablation_episode"]
            episode["six_term_reference_episode"] = episode["J_episode"]
            episode["J_episode"] -= episode["episode_timeout"]
            episode["own_objective_episode"] = episode["J_episode"]
            episode["actual_return"] = float(self.component_totals[:5].sum())
            for count in range(3, self.env.lane_num):
                episode[f"request_{count}"] = int(self.action_counts[count])
            if self.arm in DROP:
                i = DROP[self.arm]
                episode["own_objective_episode"] -= episode[f"episode_{COMPONENTS[i]}"]
                episode["actual_return"] -= float(self.component_totals[i])
        return obs, float(reward), done, truncated, info


def make_env(lanes=3, max_dl=None, arm="Full"):
    return FiveReward(lanes=lanes, max_dl=max_dl, arm=arm)
