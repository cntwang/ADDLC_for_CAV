

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from cav_dl.agents.policy import load_policy
from cav_dl.data.cases import load_case
from cav_dl.envs._legacy import kernel as k
from cav_dl.envs.clearance import ClearanceNative
from cav_dl.envs.reward import make_env
from cav_dl.utils.io import export_table


def native_episode(data, policy, lanes=3, max_dl=None):
    env = make_env(lanes, max_dl)
    obs, _ = env.reset(options=data, test=True)
    env.skip_reference_identity = True
    env.episode_p = float(data["p"])
    env.env.strict_cav_conservation = True
    for _ in range(100):
        obs, _, done, truncated, info = env.step(policy(obs))
        if done or truncated:
            result = info["ablation_episode"]
            env.close()
            return result
    env.close()
    raise RuntimeError("Native episode exceeded the documented control horizon")


def clearance_episode(data, policy, lanes=3, max_dl=None, trajectory=None, compliance=1., recorder=None):
    if not 0 <= compliance <= 1:
        raise ValueError("HV compliance must be in [0, 1]")
    if compliance < 1:
        from cav_dl.envs.noncompliance import StayNative
        env = StayNative(compliance, lanes, max_dl)
    else:
        env = ClearanceNative(lanes, max_dl)
    env.reset(options=data, test=True)
    obs = env.observe()
    cutoff = len(data["input_supply"])
    if cutoff % 60:
        raise ValueError("Demand horizon must align with the 60-microstep control period")
    score = 0.
    fixed = departures = switches = rejects = 0.
    previous = np.zeros(4)
    max_n = max_c = 0.
    while env.t < 28800 and not (env.t >= cutoff and env.residual() < 1e-7):
        action = int(policy(obs))
        request = env.decode_discrete_action(action)
        config = np.zeros(120)
        for s, count in enumerate(request):
            config[(s + 1) * 20:(s + 2) * 20] = count
        env.rejected_microsteps = 0
        env._advance(min(60, 28800 - env.t), config)
        obs = env.observe()
        _, density, capacity, safety = env.kernel.reward_components(env._history, env.t, obs)
        occupancy = max(obs[4], obs[9], obs[14], obs[19], obs[24], obs[29])
        safety = float(np.float32(safety)) if .5 < occupancy <= 1 else 0.
        rejected = env.rejected_microsteps > 0
        score += density + capacity + safety + (
            -35. if rejected else -.05 * (1 + 13 * env.t / 5000)
        )
        rejects += rejected
        actual = obs[30:34]
        switches += np.abs(actual - previous).sum()
        previous = actual.copy()
        max_n = max(max_n, abs(env.latest_n_error))
        max_c = max(max_c, abs(env.latest_c_error))
        if env.t == cutoff:
            fixed, departures = score, env.cumulative_main_output
    success = env.t >= cutoff and env.residual() < 1e-7
    if not success:
        raise RuntimeError(f"Demand did not clear: step={env.t}, residual={env.residual()}")
    t = env.t
    z = env._history[:t]
    qm = env.input_supply[:t] - env.actual_input[:t]
    qr = env.ramp_inflow[:t] - env.ramp_input_actual_ts[:t]
    fresh = np.zeros(t)
    fresh[:cutoff] = data["input_supply"] + data["ramp_inflow"]
    ledger = (data["init_car"].sum() + np.cumsum(fresh)
              - np.r_[0., np.cumsum(z[:-1, -1, k.O])]) * 15 / 3600
    road = (z[:, :, k.N].sum(1) + env.actual_input[:t]
            + env.ramp_input_actual_ts[:t]) * 15 / 3600
    area_error = np.max(np.abs(ledger - road - (qm + qr) * 15 / 3600))
    if max_n >= 1e-5 or max_c >= 1e-5 or area_error >= 1e-6:
        raise ValueError(f"Conservation failure: {max_n}, {max_c}, {area_error}")
    total = data["init_car"].sum() + fresh.sum()
    cav = np.zeros(t)
    cav[:cutoff] = (data["input_supply"] * data["input_cav_supply"]
                   + data["ramp_inflow"] * data["ramp_cav_ratio"])
    initial_cav = data["init_car"] @ data["init_cav_ratio"]
    cav_time = (initial_cav + np.cumsum(cav)
                - np.r_[0., np.cumsum(z[:-1, -1, k.OC])]).sum() * 15 / 3600
    cav_count = initial_cav + cav.sum()
    cav_minutes = cav_time / cav_count * 60 if cav_count > 1e-10 else np.nan
    hv_minutes = ((ledger.sum() - cav_time) / (total - cav_count) * 60
                  if total - cav_count > 1e-10 else np.nan)
    result = dict(
        J_fixed_control=float(fixed), complete_tts=float(ledger.sum()),
        mean_entry_wait_h=float((qm + qr).sum() * 15 / 3600 / fresh.sum()),
        fixed_departures=float(departures), total_departures=float(z[:, -1, k.O].sum()),
        total_vehicles=float(total), end_step=t, success=success,
        executed_lane_switches=float(switches), rejected_controls=int(rejects),
        max_mass_error=float(max_n), max_CAV_error=float(max_c),
        area_error_max=float(area_error), CAV_mean_system_minutes=float(cav_minutes),
        HV_mean_system_minutes=float(hv_minutes),
    )
    from cav_dl.metrics import physical_metrics
    result.update(physical_metrics(z, lanes, qm, qr,
                                   env.actual_input[:t] + env.ramp_input_actual_ts[:t]))
    if trajectory is not None:
        
        period = z[120:cutoff:4]
        time, cell = np.indices(period.shape[:2])
        stock = period[:, :, k.N]
        speed = np.divide(period[:, :, k.O] / 3, stock * 15 / 3600,
                          out=np.full_like(stock, np.nan), where=stock > 1e-10)
        export_table(pd.DataFrame({
            "time_h": time.ravel() / 60 + .5, "cell": cell.ravel() + 1,
            "density": (stock / (lanes / 3)).ravel(),
            "flow": (period[:, :, k.O] * 240).ravel(), "speed": speed.ravel(),
        }), trajectory)
    if recorder is not None:
        recorder(env, data)
        result.update(recorder.metrics)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Directory of case_XXXXX folders")
    parser.add_argument("--algorithm", default="FS1", help="ppo/dqn/sac/rule/FS1..FS9")
    parser.add_argument("--model", help="Directory containing model_final and normalizer_final")
    parser.add_argument("--lanes", type=int, choices=[3, 4, 5], default=3)
    parser.add_argument("--max-dl", type=int)
    parser.add_argument("--transfer", action="store_true", help="Frozen three-lane projection")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, help="Maximum number of cases")
    parser.add_argument("--trajectory", action="store_true")
    parser.add_argument("--label", help="Optional arm name for exported comparisons")
    parser.add_argument("--compliance", type=float, default=1., help="HV stress: 3 lanes only")
    args = parser.parse_args()
    policy = load_policy(args.algorithm, args.model, args.lanes, args.max_dl, args.transfer)
    folders = sorted(Path(args.cases).glob("case_*"))
    if args.limit is not None:
        folders = folders[:args.limit]
    if not folders:
        parser.error("No case_XXXXX demand directories found")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    seed = -1
    if args.model:
        match = re.search(r"seed(\d+)", str(args.model))
        if match:
            seed = int(match.group(1))
        metadata = Path(args.model) / "manifest.json"
        if metadata.exists():
            seed = int(json.loads(metadata.read_text()).get("seed", seed))
    for folder in folders:
        data = load_case(folder)
        native = native_episode(data, policy, args.lanes, args.max_dl) if args.compliance == 1 else {
            "J_episode": float("nan"), "reason": "not_evaluated_noncompliance",
        }
        trajectory = output / f"{folder.name}_trajectory.csv" if args.trajectory else None
        metrics = clearance_episode(data, policy, args.lanes, args.max_dl,
                                    trajectory, args.compliance)
        row = dict(case_id=int(folder.name.split("_")[-1]), lanes=args.lanes,
                   arm=args.label or args.algorithm, training_seed=seed,
                   p=data["p"], compliance=args.compliance, traffic_seed=data["seed"],
                   date=data.get("date"), rep=data.get("rep"),
                   boundary_scale=data.get("boundary_scale"),
                   scenario_sha256=data["sha256"], native_episode_reward=native["J_episode"],
                   native_reason=native["reason"], **metrics)
        rows.append(row)
        export_table(pd.DataFrame(rows), output / "raw.csv", omit=("scenario_sha256",))
        print(f"{folder.name}: TTS={metrics['complete_tts']:.6f}", flush=True)
    (output / "complete.json").write_text(json.dumps(dict(cases=len(rows), passed=True)))


if __name__ == "__main__":
    main()
