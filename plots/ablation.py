import numpy as np
import matplotlib.pyplot as plt
from cav_dl.utils.io import read_table
from cav_dl.utils.style import save

ARMS = ["Full", "NoDensity", "NoCapacity", "NoSafety", "NoTime", "NoEarlyPenalty"]
COLORS = ["#bd403a", "#dc9136", "#3d8d74", "#8669a8", "#477fa7", "#88705c"]

def render(data, output):
    frame = read_table(data, ["metric", "arm", "mean", "low", "high"])
    metrics = [("J_fixed_control", "Reward"), ("complete_tts", "$10^5$ veh h"), ("mean_entry_wait_h", "min/vehicle"), ("fixed_departures", "$10^3$ vehicles"), ("full_high_occupancy_vehicle_hours", "veh h"), ("executed_lane_switches", "Count")]
    for letter, (metric, unit) in zip("bcdefg", metrics):
        q = frame[frame.metric.eq(metric)].set_index("arm").loc[ARMS]
        m, lo, hi = (q[c].to_numpy() for c in ["mean", "low", "high"])
        fig, ax = plt.subplots(figsize=(5.6, 3.8), layout="constrained")
        ax.bar(range(6), m, color=COLORS, yerr=np.array([m-lo, hi-m]), capsize=3)
        pad = max((hi.max()-lo.min())*.16, abs(hi.max())*.001)
        bottom = max(0., lo.min()-pad) if lo.min() >= 0 else lo.min()-pad
        ax.set_ylim(bottom, hi.max()+pad)
        ax.set_xticks(range(6), ["Full", "-D", "-C", "-S", "-T", "-E"])
        ax.set_ylabel(unit)
        if bottom > 0:
            ax.plot([-.015, .015], [-.02, .02], transform=ax.transAxes, color="black", clip_on=False)
        save(fig, output, f"reward_ablation_{letter}")

def learning(data, output):
    q = read_table(data, ["arm", "steps", "mean", "seed_sd"])
    fig, ax = plt.subplots(figsize=(5.6, 9.8), layout="constrained")
    for arm, color in zip(ARMS, COLORS):
        t = q[q.arm.eq(arm)].sort_values("steps")
        mean, sd = t["mean"], t.seed_sd
        ax.plot(t.steps/1e6, mean, color=color, label=arm)
        ax.fill_between(t.steps/1e6, mean-sd, mean+sd, color=color, alpha=.12)
    ax.set(ylim=(-40, 20), yticks=[-40, -20, 0, 20],
           xlabel=r"Training steps ($\times10^6$)",
           ylabel="Common full-five episode reward")
    ax.legend(fontsize=13)
    save(fig, output, "reward_ablation_a")
