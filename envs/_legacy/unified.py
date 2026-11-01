import numpy as np
import gymnasium as gym
from cav_dl.envs._legacy.environment import EpisodeInputGenerator
from cav_dl.envs._legacy.warmup import BatchedRewardExperiment
from cav_dl.envs._legacy.core import COMPONENTS, DROP


class UnifiedEnv(BatchedRewardExperiment):
    
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        env = self.env
        self.warm_components = np.array([env.episode_total_density_reward, env.episode_total_flow_reward,
                                         env.episode_total_safety_penalty, 0., 0., 0.], dtype=float)
        assert abs(self.warm_components.sum() - self.warmup_reference) < 1e-6
        obs = obs.copy()
        obs[30:34] = 0.
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = super().step(action)
        obs = obs.copy()
        
        
        obs[30:34] = self.env.decode_discrete_action(int(action))
        if done or trunc:
            ep = info['ablation_episode']
            full = self.component_totals + self.warm_components
            ep.update(J_control=float(self.component_totals.sum()), J_warmup=float(self.warm_components.sum()),
                      J_episode=float(full.sum()), observation_schema='causal38')
            for j, name in enumerate(COMPONENTS):
                ep['episode_' + name] = float(full[j])
            warm_own = self.warm_components.sum()
            if self.variant in DROP:
                warm_own -= self.warm_components[DROP[self.variant]]
            ep['own_objective_episode'] = float(ep['actual_return'] + warm_own)
            
            
            
            if not getattr(self, 'skip_reference_identity', False):
                assert abs(ep['J_episode'] - ep['reference_with_warmup']) < 1e-6
        return obs, reward, done, trunc, info


class TrainingEpisodes(gym.Wrapper):
    
    def __init__(self, env, training_seed):
        super().__init__(env)
        self.base = 30000000 + 2000000 * (training_seed - 202610150)
        self.counter = 0

    def reset(self, *, seed=None, options=None):
        case = self.base + self.counter
        self.counter += 1
        saved = np.random.get_state()
        try:
            np.random.seed(case)
            data = EpisodeInputGenerator().generate_test_reset_config(case, 10600, 6500, [.4, .5, .6, .7, .8, .9])
        finally:
            np.random.set_state(saved)
        obs, info = self.env.reset(options=data, test=True)
        return obs, info

    def step(self, action):
        obs, reward, done, trunc, info = self.env.step(action)
        if done or trunc:
            ep = info['ablation_episode']
            info['episode'] = {'r': ep['actual_return'], 'l': ep['controls'], 't': 0.}
            ep['training_traffic_seed'] = self.base + self.counter - 1
        return obs, reward, done, trunc, info
