

import gymnasium as gym
import numpy as np

from .synthetic import MODES, scene


class Episodes(gym.Wrapper):
    def __init__(self, env, seed, demand=None):
        super().__init__(env)
        self.base = 700000000 + 2000000 * (seed - 202610150)
        if self.base < 0:
            
            self.base = int(np.random.SeedSequence([seed, 1915]).generate_state(1)[0]) % 2000000000
        self.counter = 0
        self.demand = demand

    def reset(self, *, seed=None, options=None):
        case = self.base + self.counter
        self.counter += 1
        rng = np.random.default_rng(np.random.SeedSequence([case, 1915]))
        p = float(rng.uniform())
        if self.demand is None:
            main = MODES[int(rng.integers(6))]
            ramp = MODES[int(rng.integers(6))]
            data = scene(case, p, main, ramp)
            extra = dict(main_mode=main, ramp_mode=ramp, cav_mode=data["cav_mode"])
        else:
            day = self.demand.days[int(rng.integers(len(self.demand.days))) ]
            data = self.demand.scene(case, p, day)
            extra = dict(data_date=day)
        self.meta = dict(
            training_traffic_seed=case, scenario_sha256=data["sha256"],
            offered_arrivals=data["offered_arrivals"],
            arrival_quota_error=data["arrival_quota_error"], **extra,
        )
        obs, info = self.env.reset(options=data, test=True)
        self.env.env.strict_cav_conservation = True
        self.env.episode_p = p
        self.env.skip_reference_identity = True
        info.update(self.meta)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        if done or truncated:
            episode = info["ablation_episode"]
            episode.update(self.meta, cav_mass_error_max=float(self.env.env.cav_mass_error_max))
            info["episode"] = dict(r=episode["actual_return"], l=episode["controls"], t=0.)
        return obs, reward, done, truncated, info
