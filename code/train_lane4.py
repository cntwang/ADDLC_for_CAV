"""Train, refine and validate four-lane PPO using the released seed recipes."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import csv
from functools import partial
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd

from cav_dl.data.episodes import Episodes
from cav_dl.data.pems import PeMSDemand
from cav_dl.envs.reward import make_env

PROJECT = Path(__file__).resolve().parents[1]


class FourLaneDemand(PeMSDemand):
    def scene(self, seed, p, day, composition_scale=1.0):
        return super().scene(seed, p, day, composition_scale=composition_scale)


def environment(seed, rank):
    env = Episodes(make_env(4, 3), seed,
                   FourLaneDemand(PROJECT / 'assets/pems', 4, scale=.8))
    env.base += rank * 200000
    return env


def train(seed, folder, recipe, refine=False):
    import torch
    from torch import nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from cav_dl.agents.schedules import EntCoefSchedulerBySteps

    config = json.loads((PROJECT / 'defaults.json').read_text())
    c = config['ppo']
    c['n_steps'] = 512
    stage = folder / ('refinement' if refine else 'initial')
    if (stage / 'complete.json').exists():
        return
    if (stage / 'episodes.csv').exists():
        raise FileExistsError('Use a new output directory for an incomplete run')
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    rank_offset = 5 if refine else 0
    vec = SubprocVecEnv([partial(environment, seed, i + rank_offset) for i in range(4)],
                        start_method='spawn')
    if refine:
        source = folder / 'checkpoints' / f'step_{int(recipe.initial_checkpoint)}'
        vec = VecNormalize.load(source / 'normalizer_final.pkl', vec)
        vec.training, vec.norm_reward = True, True
        model = PPO.load(source / 'model_final.zip', env=vec, device='cpu', seed=seed + 1000,
                         learning_rate=.00005, ent_coef=.001)
        start, budget = model.num_timesteps, int(recipe.additional_steps)
        points = [65536, 131072, 262144] if budget == 262144 else [250000, 500000, 750000, 1000000]
        schedules = []
    else:
        vec = VecNormalize(vec, **config['normalization'])
        start, budget, points = 0, 1000000, [250000, 500000, 750000, 1000000]
        def learning_rate(progress):
            return c['lr_start'] + (c['lr_end'] - c['lr_start']) * min(1., (1 - progress) * budget / c['lr_decay_steps'])
        keys = ('gamma', 'gae_lambda', 'n_steps', 'batch_size', 'n_epochs', 'clip_range',
                'clip_range_vf', 'vf_coef', 'max_grad_norm', 'target_kl')
        model = PPO('MlpPolicy', vec, learning_rate=learning_rate, ent_coef=c['ent_start'],
                    **{key: c[key] for key in keys}, seed=seed, device='cpu', verbose=0,
                    policy_kwargs=dict(activation_fn=nn.ReLU, net_arch=dict(pi=[128, 128], vf=[128, 128]), ortho_init=False))
        schedules = [EntCoefSchedulerBySteps(c['ent_start'], c['ent_end'], c['ent_decay'], c['ent_warmup'])]

    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.index, self.stream, self.last = 0, None, 0

        def _on_step(self):
            local = self.num_timesteps - start
            if refine:
                fraction = min(1., max(0., local - 20000) / (budget - 20000))
                self.model.ent_coef = .001 + (.0001 - .001) * fraction
            for info in self.locals['infos']:
                if 'ablation_episode' not in info:
                    continue
                row = dict(steps=self.num_timesteps, local_steps=local,
                           **{k: v for k, v in info['ablation_episode'].items()
                              if isinstance(v, (str, int, float, bool))})
                if self.stream is None:
                    self.stream = (stage / 'episodes.csv').open('w', newline='', encoding='utf-8')
                    self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
            if local - self.last >= 20000:
                self.last = local
                if self.stream:
                    self.stream.flush()
                print(f'{seed}: {local}/{budget}', flush=True)
            return True

        def snapshot(self):
            dest = folder / 'checkpoints' / f'step_{self.num_timesteps}'
            dest.mkdir(parents=True, exist_ok=True)
            self.model.save(dest / 'model_final.zip')
            vec.save(dest / 'normalizer_final.pkl')
            (dest / 'manifest.json').write_text(json.dumps(dict(seed=seed,
                steps=self.num_timesteps, local_steps=self.num_timesteps - start,
                refinement=refine, source_checkpoint=start, boundary_scale=.8, max_dl=3)))

        def _on_rollout_end(self):
            if self.index < len(points) and self.num_timesteps - start >= points[self.index]:
                self.snapshot()
                self.index += 1

    recorder = Recorder()
    model.learn(budget, callback=CallbackList([*schedules, recorder]), reset_num_timesteps=not refine)
    recorder.snapshot()
    if recorder.stream:
        recorder.stream.close()
    vec.close()
    (stage / 'complete.json').write_text(json.dumps(dict(steps=model.num_timesteps,
        additional_steps=model.num_timesteps - start, reward='unchanged five terms')))


def select(seed, folder, recipe):
    from cav_dl.agents.policy import load_policy
    from cav_dl.evaluation import native_episode, clearance_episode
    cases = pd.read_csv(PROJECT / 'configs/lane4_validation.csv')
    demand = FourLaneDemand(PROJECT / 'assets/pems', 4, scale=.8)
    rows = []
    for checkpoint in sorted((folder / 'checkpoints').glob('step_*'), key=lambda p: int(p.name[5:])):
        policy = load_policy('ppo', checkpoint, 4, 3)
        for case in cases.itertuples():
            data = demand.scene(int(case.traffic_seed), case.p, case.date)
            if data['sha256'] != case.scenario_sha256:
                raise ValueError('Validation input fingerprint mismatch')
            native = native_episode(data, policy, 4, 3)
            physical = clearance_episode(data, policy, 4, 3)
            rows.append(dict(checkpoint=int(checkpoint.name[5:]), case_id=case.case_id,
                             native_episode_reward=native['J_episode'], complete_tts=physical['complete_tts']))
    raw = pd.DataFrame(rows)
    raw.to_csv(folder / 'validation_raw.csv', index=False)
    means = raw.groupby('checkpoint')[['native_episode_reward', 'complete_tts']].mean()
    means.to_csv(folder / 'validation_summary.csv')
    eligible = means[means.complete_tts <= recipe.validation_tts_limit]
    best = (eligible.sort_values('native_episode_reward', ascending=False)
            if len(eligible) else means.sort_values('complete_tts')).index[0]
    (folder / 'selected.json').write_text(json.dumps(dict(seed=seed, checkpoint=int(best),
        validation_tts_gate_passed=bool(len(eligible)))))
    print(f'Selected {folder / "checkpoints" / f"step_{best}"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['train', 'refine', 'select'])
    recipes = pd.read_csv(PROJECT / 'configs/lane4_training.csv').set_index('seed')
    parser.add_argument('--seed', type=int, choices=recipes.index.tolist(), required=True)
    parser.add_argument('--output', type=Path, default=PROJECT / 'runs/lane4')
    args = parser.parse_args()
    folder = args.output / f'PPO_seed{args.seed}'
    recipe = recipes.loc[args.seed]
    if args.phase == 'select':
        select(args.seed, folder, recipe)
    else:
        train(args.seed, folder, recipe, refine=args.phase == 'refine')


if __name__ == '__main__':
    main()
