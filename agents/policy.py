

import itertools
import pickle
from pathlib import Path

import numpy as np

FIXED = [
    (0, 0, 0, 0), (1, 1, 1, 1), (0, 0, 1, 1), (1, 1, 0, 0), (0, 1, 1, 0),
    (2, 2, 2, 2), (0, 0, 2, 2), (2, 2, 0, 0), (0, 2, 2, 0),
]


def load_policy(algorithm, directory, lanes=3, max_dl=None, transfer=False):
    
    algorithm = algorithm.lower()
    if algorithm not in {"ppo", "dqn", "sac", "rule", *[f"fs{i}" for i in range(1, 10)]}:
        raise ValueError(f"Unknown controller: {algorithm}")
    max_dl = lanes - 1 if max_dl is None else max_dl
    table = list(itertools.product(range(max_dl + 1), repeat=4))
    if algorithm.upper().startswith("FS"):
        request = FIXED[int(algorithm[2:]) - 1]
        return lambda obs: table.index(request)
    if algorithm.lower() == "rule":
        from cav_dl.envs._legacy.kernel import fd

        def rule(obs):
            request = []
            for seg in range(1, 5):
                dl, gl, pd, pg, safety = obs[seg * 5:seg * 5 + 5]
                n = float(obs[29 + seg])
                mass = n * dl + (lanes - n) * gl
                p = (n * dl * pd + (lanes - n) * gl * pg) / mass if mass > 1e-12 else 0.
                p = float(np.clip(p, 0, 1))
                count = int(p >= .55) + int(p >= .85)
                request.append(min(count, max_dl) if mass / lanes >= 1.5 * fd(p)[3]
                               and safety < .9 else 0)
            return table.index(tuple(request))
        return rule

    if directory is None:
        raise ValueError("A model directory is required for PPO, DQN or SAC")
    directory = Path(directory)
    with (directory / "normalizer_final.pkl").open("rb") as stream:
        normalizer = pickle.load(stream)
    normalizer.training = False
    normalizer.norm_reward = False
    normalizer.venv = None
    if algorithm.lower() == "sac":
        import torch
        from .sac import DiscreteSAC
        actions = 81 if transfer else len(table)
        model = DiscreteSAC(actions=actions)
        state = torch.load(directory / "model_final.pt", map_location="cpu", weights_only=False)
        model.actor.load_state_dict(state["actor"])
    else:
        from stable_baselines3 import PPO, DQN
        model = (PPO if algorithm.lower() == "ppo" else DQN).load(
            directory / "model_final.zip", device="cpu",
        )

    def predict(obs):
        projected = np.asarray(obs, dtype=np.float32).copy()
        if transfer:
            projected[34:37] *= 3. / lanes
        normalized = normalizer.normalize_obs(projected)
        if algorithm.lower() == "sac":
            action = model.predict(normalized, deterministic=True)
        else:
            action = int(model.predict(normalized, deterministic=True)[0])
        if transfer:
            request = tuple((action // 3**j) % 3 for j in [3, 2, 1, 0])
            return table.index(request)
        return int(action)
    return predict
