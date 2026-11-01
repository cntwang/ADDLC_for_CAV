

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np


def defaults():
    return json.loads(Path(__file__).with_name("defaults.json").read_text())


def train(algorithm, lanes, seed, steps, output, config=None, pems=None,
          max_dl=None, arm="Full", pems_scale=None):
    import torch
    from torch import nn
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from cav_dl.envs.reward import make_env
    from cav_dl.data.episodes import Episodes
    from cav_dl.data.pems import PeMSDemand
    from cav_dl.agents.schedules import EntCoefSchedulerBySteps

    config = defaults() if config is None else config
    output = Path(output)
    if (output / "manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    demand = PeMSDemand(pems, lanes=lanes, scale=pems_scale) if pems else None
    env = Episodes(make_env(lanes, max_dl, arm), seed, demand)
    vec = VecNormalize(DummyVecEnv([lambda: env]), **config["normalization"])
    manifest = dict(algorithm=algorithm, lanes=lanes, max_dl=env.env.env.max_dl,
                    seed=seed, requested_steps=steps, config=config, reward_variant=arm,
                    demand="pems" if pems else "six_family_synthetic",
                    boundary_scale=demand.scale if demand else None)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    started = time.perf_counter()

    class EpisodeLog(BaseCallback):
        def __init__(self):
            super().__init__()
            self.count = 0
            self.stream = None

        def write(self, step, info):
            if "ablation_episode" not in info:
                return
            self.count += 1
            row = dict(steps=step, episode=self.count, **info["ablation_episode"])
            row["common_reward"] = row["J_episode"]
            row.pop("scenario_sha256", None)
            if self.stream is None:
                self.stream = (output / "episodes.csv").open("w", newline="", encoding="utf-8")
                self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                self.writer.writeheader()
            self.writer.writerow(row)
            self.stream.flush()

        def _on_step(self):
            for info in self.locals["infos"]:
                self.write(self.num_timesteps, info)
            return True

    recorder = EpisodeLog()
    if algorithm == "sac":
        from cav_dl.agents.sac import DiscreteSAC, ThreeStepReplay
        model = DiscreteSAC(seed, actions=vec.action_space.n)
        replay = ThreeStepReplay(seed=seed)
        vec.action_space.seed(seed)
        obs = vec.reset()
        with (output / "updates.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "steps", "actor_loss", "critic_loss", "alpha", "entropy",
            ])
            writer.writeheader()
            for step in range(1, steps + 1):
                raw = vec.get_original_obs()[0].copy()
                action = int(vec.action_space.sample()) if step <= 10000 else model.predict(obs[0])
                following, _, done, info = vec.step(np.array([action]))
                nxt = vec.unnormalize_obs(info[0]["terminal_observation"]) if done[0] else (
                    vec.get_original_obs()[0].copy()
                )
                replay.add(raw, action, float(vec.get_original_reward()[0]), nxt, bool(done[0]))
                obs = following
                recorder.write(step, info[0])
                if step >= 10000 and step % 10 == 0 and replay.size >= 1024:
                    for _ in range(10):
                        batch = replay.sample(vec, 1024)
                        for i in [0, 3]:
                            batch[i] = torch.as_tensor(vec.normalize_obs(batch[i].numpy()),
                                                       dtype=torch.float32)
                        losses = model.update(batch)
                    writer.writerow(dict(steps=step, **losses))
        model.save(output / "model_final.pt", steps)
        actual_steps = steps
    elif algorithm == "ppo":
        c = config["ppo"]
        def lr(progress):
            fraction = min(1., (1 - progress) * steps / c["lr_decay_steps"])
            return c["lr_start"] + (c["lr_end"] - c["lr_start"]) * fraction
        keys = ["gamma", "gae_lambda", "n_steps", "batch_size", "n_epochs", "clip_range",
                "clip_range_vf", "vf_coef", "max_grad_norm", "target_kl"]
        model = PPO("MlpPolicy", vec, learning_rate=lr, ent_coef=c["ent_start"],
                    **{key: c[key] for key in keys}, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(activation_fn=nn.ReLU,
                                       net_arch=dict(pi=[128, 128], vf=[128, 128]),
                                       ortho_init=False))
        schedule = EntCoefSchedulerBySteps(c["ent_start"], c["ent_end"],
                                          c["ent_decay"], c["ent_warmup"])
        model.learn(steps, callback=CallbackList([schedule, recorder]))
        model.save(output / "model_final.zip")
        actual_steps = model.num_timesteps
    else:
        c = {key: value for key, value in config["dqn"].items() if key != "network"}
        model = DQN("MlpPolicy", vec, **c, seed=seed, device="cpu", verbose=0,
                    policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=[128, 128]))
        model.learn(steps, callback=recorder)
        model.save(output / "model_final.zip")
        actual_steps = model.num_timesteps
    vec.save(output / "normalizer_final.pkl")
    if recorder.stream:
        recorder.stream.close()
    vec.close()
    result = dict(steps=actual_steps, episodes=recorder.count,
                  elapsed=time.perf_counter() - started)
    (output / "complete.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=["ppo", "dqn", "sac"], default="ppo")
    parser.add_argument("--lanes", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--seed", type=int, default=202610150)
    parser.add_argument("--steps", type=int, default=1000000)
    parser.add_argument("--arm", default="Full")
    parser.add_argument("--max-dl", type=int)
    parser.add_argument("--pems-data", help="PeMS manifest.json and daily_flows.npz directory")
    parser.add_argument("--pems-scale", type=float, help="Override common PeMS boundary scaling")
    parser.add_argument("--config", help="Optional JSON optimizer configuration")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text()) if args.config else None
    result = train(args.algorithm, args.lanes, args.seed, args.steps, args.output,
                   config, args.pems_data, args.max_dl, args.arm, args.pems_scale)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
