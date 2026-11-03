import os

for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import random
import shutil
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
R = PROJECT
import numpy as np
import pandas as pd
from cav_dl.data.pems import PeMSDemand
from cav_dl.data.episodes import Episodes
from cav_dl.data.cases import load_case, save_case
from cav_dl.envs.reward import make_env

NAME = "lane4_scale08_corrected_20261103"
RUN = R / "runs" / NAME
EVAL = R / "evaluations" / NAME
OUT = R / "output/pdf" / NAME
SEEDS = [202610150, 202610151, 202610152]
DATA = PROJECT / "assets/pems"
TRANSFER_MODELS = PROJECT / "models/pems3"
if not TRANSFER_MODELS.exists():
    TRANSFER_MODELS = R / "deliverables/PeMS_both06_Reward_panels_moderate100_CI/models"
STEPS = 1000000
METRICS = ["native_episode_reward", "complete_tts", "mean_entry_wait_h", "fixed_departures"]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    for attempt in range(7):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 6:
                raise
            time.sleep(.05 * 2**attempt)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CheckedDemand(PeMSDemand):
    def scene(self, seed, p, day, composition_scale=1.0):
        data = super().scene(seed, p, day, composition_scale)
        raw = np.repeat(self.archive[day.replace("-", "")][:, 36:162] / 20, 20, axis=1)
        np.testing.assert_array_equal(data["input_supply"], raw[0] * .8)
        np.testing.assert_array_equal(data["ramp_inflow"], raw[1] * .8)
        assert abs(data["offered_arrivals"] - raw.sum() * .8) < 1e-7
        return data


class CheckedEpisodes(Episodes):
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.meta.update(boundary_scale=.8, actual_demand_scale_verified=True)
        return obs, info


def environment(seed, rank):
    env = CheckedEpisodes(make_env(4, 3), seed, CheckedDemand(DATA, 4, scale=.8))
    env.base += rank * 200000
    return env


def prepare():
    EVAL.mkdir(parents=True, exist_ok=True)
    if (EVAL / "protocol.json").exists():
        return
    config = json.loads((PROJECT / "configs/training.json").read_text())
    config["ppo"]["n_steps"] = 512
    days = json.loads((DATA / "manifest.json").read_text())["date_splits"]
    validation = []
    for day in days["validation"]:
        for p in np.arange(1, 10) / 10:
            for rep in range(2):
                index = len(validation)
                validation.append(dict(case_id=index, traffic_seed=1600000000 + index,
                                       p=float(p), date=day, rep=rep,
                                       boundary_scale=.8, composition_scale=1.))
    val = pd.DataFrame(validation)
    val.to_csv(EVAL / "validation_cases.csv", index=False)
    test = pd.read_csv(PROJECT / "configs/test_cases_4lane.csv", float_precision="round_trip")
    test.to_csv(EVAL / "test_cases.csv", index=False)
    assert set(val.date).isdisjoint(test.date)
    assert set(val.date).isdisjoint(days["train"])
    assert set(val.traffic_seed).isdisjoint(test.traffic_seed)
    checked = CheckedDemand(DATA, 4, scale=.8)
    for split, frame in [("validation", val), ("test", test)]:
        for row in frame.itertuples():
            scene = checked.scene(int(row.traffic_seed), row.p, row.date)
            if split == "test":
                assert scene["sha256"] == row.scenario_sha256
            save_case(scene, EVAL / split / "cases" / f"case_{row.case_id:05d}")
    for seed in SEEDS:
        for rank in range(4):
            env = environment(seed, rank)
            obs, info = env.reset()
            assert obs.shape == (38,) and env.action_space.n == 256
            for action in [0, 85, 255]:
                _, reward, done, trunc, step_info = env.step(action)
                assert abs(reward - step_info["six_reward_components"][:5].sum()) < 1e-5
                if done or trunc:
                    break
            env.close()
    write_json(EVAL / "training.json", config)
    protocol = dict(name=NAME, lanes=4, actual_boundary_scale=.8,
                    composition="original quota15; composition_scale=1.0; initial stock unchanged",
                    p_train="Uniform(0,1)", seeds=SEEDS, steps_per_seed=STEPS,
                    initialization="fresh PPO; no rule hybrid or extra policy information",
                    vector_envs=4, steps_per_env=512, total_rollout=2048,
                    optimization="original PPO settings; only rollout collection parallelized",
                    training_dates=days["train"], validation_dates=days["validation"],
                    test_dates=sorted(test.date.unique().tolist()), validation_cases=len(val),
                    test_cases=len(test), checkpoint_decisions=[250000, 500000, 750000, 1000000],
                    selection="For each seed: maximum mean native five-term validation reward among checkpoints whose complete validation TTS is no higher than the three-seed frozen transfer mean; if none qualifies use highest reward and report gate failure. No test outcomes used for selection.",
                    reward="unchanged five terms and weights; no timeout penalty",
                    termination="original native termination retained in training/native reward; complete TTS includes both entry queues until clearance",
                    test_cohort="Current configured fixed-seed test cohort; inference limited to this cohort",
                    five_lane="unchanged original frozen models and saved results; scale1.0",
                    source_sha256=sha(Path(__file__)),
                    physics_sha256={str(p.relative_to(PROJECT)): sha(p) for p in
                        [PROJECT / "envs/kernels/lane4.py", PROJECT / "envs/reward.py"]},
                    cases_sha256={s: sha(EVAL / f"{s}_cases.csv") for s in ["validation", "test"]})
    write_json(EVAL / "protocol.json", protocol)
    print("PREPARED: exact 0.8 arrays, five-term identity, 72 validation / 100 unchanged test cases", flush=True)


def train(seed):
    import torch
    from torch import nn
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList
    from cav_dl.agents.schedules import EntCoefSchedulerBySteps

    folder = RUN / f"PPO_seed{seed}"
    if (folder / "complete.json").exists():
        return
    assert not (folder / "manifest.json").exists(), "Existing partial run must be resumed explicitly"
    folder.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    config = json.loads((EVAL / "training.json").read_text())
    c = config["ppo"]
    vec = VecNormalize(SubprocVecEnv([partial(environment, seed, i) for i in range(4)],
                                    start_method="spawn"), **config["normalization"])
    def lr(progress):
        return c["lr_start"] + (c["lr_end"] - c["lr_start"]) * min(1., (1-progress) * STEPS / c["lr_decay_steps"])
    keys = ("gamma", "gae_lambda", "n_steps", "batch_size", "n_epochs", "clip_range",
            "clip_range_vf", "vf_coef", "max_grad_norm", "target_kl")
    model = PPO("MlpPolicy", vec, learning_rate=lr, ent_coef=c["ent_start"],
                **{k: c[k] for k in keys}, policy_kwargs=dict(activation_fn=nn.ReLU,
                net_arch=dict(pi=[128, 128], vf=[128, 128]), ortho_init=False), seed=seed,
                device="cpu", verbose=0)
    write_json(folder / "manifest.json", dict(seed=seed, lanes=4, boundary_scale=.8,
               config=config, fresh=True, test_access=False, reward="Full five terms", max_dl=3))
    started = time.perf_counter()
    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.threshold = 250000
            self.last = 0
            self.stream = None
            self.episodes = 0
            self.snapshots = []
        def _on_step(self):
            for info in self.locals["infos"]:
                if "ablation_episode" not in info:
                    continue
                ep = info["ablation_episode"]
                row = dict(steps=self.num_timesteps, **{k: v for k,v in ep.items()
                           if isinstance(v, (str, int, float, bool))})
                if self.stream is None:
                    self.stream = (folder / "episodes.csv").open("w", newline="", encoding="utf-8")
                    self.writer = csv.DictWriter(self.stream, fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
                self.episodes += 1
            if self.num_timesteps - self.last >= 20000:
                self.last = self.num_timesteps
                if self.stream:
                    self.stream.flush()
                write_json(folder / "progress.json", dict(steps=self.num_timesteps, target=STEPS,
                           episodes=self.episodes, elapsed=time.perf_counter()-started))
                print(f"TRAIN {seed}: {self.num_timesteps}/{STEPS}", flush=True)
            return True
        def _on_rollout_end(self):
            if self.num_timesteps >= self.threshold:
                self.snapshot()
                self.threshold += 250000
        def snapshot(self):
            path = folder / "checkpoints" / f"step_{self.num_timesteps}"
            path.mkdir(parents=True, exist_ok=True)
            self.model.save(path / "model_final.zip")
            vec.save(path / "normalizer_final.pkl")
            write_json(path / "manifest.json", dict(seed=seed, steps=self.num_timesteps,
                       lanes=4, boundary_scale=.8, max_dl=3))
            if self.num_timesteps not in self.snapshots:
                self.snapshots.append(self.num_timesteps)
            write_json(folder / "checkpoints.json", self.snapshots)
    rec = Recorder()
    schedule = EntCoefSchedulerBySteps(c["ent_start"], c["ent_end"], c["ent_decay"], c["ent_warmup"])
    model.learn(STEPS, callback=CallbackList([schedule, rec]))
    model.save(folder / "model_final.zip")
    vec.save(folder / "normalizer_final.pkl")
    rec.snapshot()
    if rec.stream:
        rec.stream.close()
    vec.close()
    write_json(folder / "complete.json", dict(steps=model.num_timesteps, episodes=rec.episodes,
               elapsed=time.perf_counter()-started, actual_scale_verified_every_reset=True))
    print(f"COMPLETE {seed}: {model.num_timesteps}", flush=True)


POLICIES = {}


def prepare_refinement():
    target = EVAL / "refinement_protocol.json"
    if target.exists():
        return
    chosen = json.loads((EVAL / "selected_checkpoints.json").read_text())
    shutil.copy2(EVAL / "selected_checkpoints.json", EVAL / "stage1_selected_checkpoints.json")
    shutil.copy2(EVAL / "validation/summary.csv", EVAL / "validation/stage1_summary.csv")
    write_json(target, dict(source_selection=chosen, extra_decisions_per_seed=262144,
               learning_rate=.00005, entropy_start=.001, entropy_end=.0001,
               entropy_warmup=20000, vector_envs=4, rollout=2048,
               checkpoint_local_steps=[65536,131072,262144],
               initialization="same-seed stage1 validation-selected checkpoint",
               training_demand="same exact0.8, same train dates; disjoint new traffic seeds",
               training_reward="unchanged native five-term reward; no TTS reward",
               selection="same validation reward / TTS gate as stage1; all seeds retained",
               test_access=False, source_sha256=sha(Path(__file__))))


def refine(seed):
    import torch
    from functools import partial
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    protocol = json.loads((EVAL / "refinement_protocol.json").read_text())
    root = RUN / f"PPO_seed{seed}"
    output = root / "refinement"
    if (output / "complete.json").exists():
        return
    assert not (output / "manifest.json").exists(), "Resume a partial refinement explicitly"
    output.mkdir(parents=True, exist_ok=True)
    start = protocol["source_selection"][str(seed)]["checkpoint"]
    source = root / "checkpoints" / f"step_{start}"
    torch.set_num_threads(1)
    vec = VecNormalize.load(source / "normalizer_final.pkl",
                           SubprocVecEnv([partial(environment,seed,i+5) for i in range(4)],start_method="spawn"))
    vec.training = True
    vec.norm_reward = True
    model = PPO.load(source / "model_final.zip", env=vec, device="cpu", seed=seed+1000,
                     learning_rate=.00005, ent_coef=.001)
    assert model.num_timesteps == start
    write_json(output / "manifest.json",dict(seed=seed, optimizer_seed=seed+1000,
               start_checkpoint=start, extra_steps=262144, protocol=protocol,
               source_model_sha256=sha(source/"model_final.zip")))
    began = time.perf_counter()
    class Recorder(BaseCallback):
        def __init__(self):
            super().__init__()
            self.stream = None
            self.last = 0
            self.points = [65536,131072,262144]
            self.index = 0
            self.episodes = 0
        def _on_step(self):
            local = self.num_timesteps-start
            fraction = min(1.,max(0.,local-20000)/(262144-20000))
            self.model.ent_coef = .001+(.0001-.001)*fraction
            for info in self.locals["infos"]:
                if "ablation_episode" not in info:
                    continue
                row=dict(steps=self.num_timesteps, local_steps=local,
                         **{k:v for k,v in info["ablation_episode"].items()
                            if isinstance(v,(str,int,float,bool))})
                if self.stream is None:
                    self.stream=(output/"episodes.csv").open("w",newline="",encoding="utf-8")
                    self.writer=csv.DictWriter(self.stream,fieldnames=list(row))
                    self.writer.writeheader()
                self.writer.writerow(row)
                self.episodes+=1
            if local-self.last>=20000:
                self.last=local
                if self.stream:
                    self.stream.flush()
                write_json(output/"progress.json",dict(local_steps=local,target=262144,
                           elapsed=time.perf_counter()-began,episodes=self.episodes))
                print(f"REFINE {seed}: {local}/262144",flush=True)
            return True
        def snapshot(self):
            dest=root/"checkpoints"/f"step_{self.num_timesteps}"
            dest.mkdir(parents=True,exist_ok=True)
            self.model.save(dest/"model_final.zip")
            vec.save(dest/"normalizer_final.pkl")
            write_json(dest/"manifest.json",dict(seed=seed,steps=self.num_timesteps,
                       source_checkpoint=start,local_steps=self.num_timesteps-start,
                       lanes=4,max_dl=3,boundary_scale=.8,refinement=True))
            points=json.loads((root/"checkpoints.json").read_text())
            write_json(root/"checkpoints.json",sorted(set(points+[self.num_timesteps])))
        def _on_rollout_end(self):
            if self.index<len(self.points) and self.num_timesteps-start>=self.points[self.index]:
                self.snapshot()
                self.index+=1
    rec=Recorder()
    model.learn(262144,callback=rec,reset_num_timesteps=False)
    rec.snapshot()
    if rec.stream:
        rec.stream.close()
    vec.close()
    write_json(output/"complete.json",dict(extra_steps=model.num_timesteps-start,
               retained_model_path_steps=model.num_timesteps,experiment_decisions=1001472+262144,
               elapsed=time.perf_counter()-began,episodes=rec.episodes))
    print(f"REFINEMENT COMPLETE {seed}",flush=True)


def evaluate_job(spec):
    import torch
    torch.set_num_threads(1)
    from cav_dl.agents.policy import load_policy
    from cav_dl.evaluation import native_episode, clearance_episode
    split, case_id, arm, seed, checkpoint = spec
    key = (arm, seed, checkpoint)
    if key not in POLICIES:
        if arm == "Transfer":
            folder = TRANSFER_MODELS / f"PPO_seed{seed}"
        elif arm == "Retrained":
            folder = RUN / f"PPO_seed{seed}/checkpoints/step_{checkpoint}"
        else:
            folder = None
        POLICIES[key] = load_policy("FS1" if arm == "FS1" else "ppo", folder, 4, 3, arm == "Transfer")
    data = load_case(EVAL / split / "cases" / f"case_{case_id:05d}")
    policy = POLICIES[key]
    native = native_episode(data, policy, 4, 3)
    physical = clearance_episode(data, policy, 4, 3)
    return dict(case_id=case_id, arm=arm, training_seed=seed, checkpoint=checkpoint,
                p=float(data["p"]), date=data["date"], traffic_seed=int(data["seed"]),
                scenario_sha256=data["sha256"], lanes=4, boundary_scale=.8,
                native_episode_reward=native["J_episode"], native_reason=native["reason"], **physical)


def evaluate(split):
    folder = EVAL / split
    journal = folder / "journal.jsonl"
    rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
    cases = pd.read_csv(EVAL / f"{split}_cases.csv")
    variants = [("Transfer", s, -1) for s in SEEDS] + [("FS1", -1, -1)]
    if split == "validation":
        for seed in SEEDS:
            points = json.loads((RUN / f"PPO_seed{seed}/checkpoints.json").read_text())
            variants.extend(("Retrained", seed, n) for n in points)
    else:
        chosen = json.loads((EVAL / "selected_checkpoints.json").read_text())
        variants.extend(("Retrained", s, chosen[str(s)]["checkpoint"]) for s in SEEDS)
    seen = {(r["case_id"], r["arm"], r["training_seed"], r["checkpoint"]) for r in rows}
    specs = [(split, int(c), a, s, n) for a,s,n in variants for c in cases.case_id
             if (int(c), a, s, n) not in seen]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn")) as pool:
        for row in pool.map(evaluate_job, specs, chunksize=3):
            rows.append(row)
            with journal.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            if len(rows) % 50 == 0:
                print(f"{split.upper()}: {len(rows)}/{len(cases)*len(variants)}", flush=True)
            if len(rows)%25 == 0 or len(rows)==len(cases)*len(variants):
                write_json(folder / "progress.json", dict(done=len(rows), expected=len(cases)*len(variants)))
    frame = pd.DataFrame(rows)
    assert len(frame) == len(cases) * len(variants)
    assert not frame.duplicated(["case_id", "arm", "training_seed", "checkpoint"]).any()
    assert frame.success.all()
    assert frame.groupby("case_id").scenario_sha256.nunique().eq(1).all()
    frame.to_csv(folder / "raw.csv", index=False)
    summary = frame.groupby(["arm", "training_seed", "checkpoint"])[METRICS].mean().reset_index()
    summary.to_csv(folder / "summary.csv", index=False)
    if split == "validation":
        reference = summary[summary.arm.eq("Transfer")].complete_tts.mean()
        chosen = {}
        for seed in SEEDS:
            options = summary[summary.arm.eq("Retrained") & summary.training_seed.eq(seed)]
            eligible = options[options.complete_tts <= reference]
            best = (eligible.sort_values(["native_episode_reward","checkpoint"],ascending=[False,True])
                    if len(eligible) else options.sort_values(["complete_tts","checkpoint"],ascending=[True,True])).iloc[0]
            chosen[str(seed)] = dict(checkpoint=int(best.checkpoint),
                                    tts_gate_passed=bool(len(eligible)),
                                    validation_reward=float(best.native_episode_reward),
                                    validation_tts=float(best.complete_tts))
        write_json(EVAL / "selected_checkpoints.json", chosen)
        print(json.dumps(chosen), flush=True)
    else:
        output(frame)


def output(frame):
    OUT.mkdir(parents=True, exist_ok=True)
    lane4 = frame.groupby(["case_id", "arm"])[METRICS].mean().groupby("arm").mean().reset_index()
    lane4["lanes"] = 4
    lane5 = pd.read_csv(PROJECT / "data/fig15.csv")
    lane5 = lane5[lane5.lanes.eq(5)]
    pd.concat([lane4, lane5], ignore_index=True)[["lanes", "arm", *METRICS]].to_csv(OUT / "fig15.csv", index=False)
    frame.to_csv(OUT / "lane4_raw.csv", index=False)
    for filename in ["protocol.json", "validation_cases.csv", "test_cases.csv", "selected_checkpoints.json",
                     "refinement_protocol.json", "selection_amendment.json", "frozen_models.json"]:
        if (EVAL / filename).exists():
            shutil.copy2(EVAL / filename, OUT / filename)
    new = lane4.set_index("arm")
    changes = {}
    for arm in ["Transfer", "FS1"]:
        changes[arm] = {m: float(100*(new.loc["Retrained",m]/new.loc[arm,m]-1))
                        for m in METRICS}
    write_json(OUT / "comparison.json", changes)
    print(lane4.to_string(index=False), flush=True)
    print(json.dumps(changes), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=["prepare", "train", "refine-prepare", "refine", "validation", "test"])
    parser.add_argument("--seed", type=int, choices=SEEDS)
    args = parser.parse_args()
    if args.phase == "prepare":
        prepare()
    elif args.phase == "train":
        train(args.seed)
    elif args.phase == "refine-prepare":
        prepare_refinement()
    elif args.phase == "refine":
        refine(args.seed)
    else:
        evaluate(args.phase)
