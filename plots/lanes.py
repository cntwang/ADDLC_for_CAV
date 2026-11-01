import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from cav_dl.utils.io import read_table
from cav_dl.utils.style import save

def render(data, output):
    frame = read_table(data, ["lanes", "arm", "native_episode_reward", "complete_tts", "mean_entry_wait_h", "fixed_departures"]).set_index(["lanes", "arm"])
    arms = ["Transfer", "Retrained", "FS1"]
    colors = ["#327DA0", "#B34B50", "#BA963B"]
    labels = ["Transferred ADDLC", "Retrained ADDLC", "FS1"]
    panels = [("reward", "native_episode_reward", "Episode reward", 1), ("travel_time", "complete_tts", "Total system time\n($10^3$ veh h)", .001), ("waiting_time", "mean_entry_wait_h", "Entry waiting time\n(min/vehicle)", 60), ("throughput", "fixed_departures", "Downstream throughput\n($10^3$ vehicles)", .001)]
    for name, metric, label, scale in panels:
        fig, ax = plt.subplots(figsize=(5.5, 4.2), layout="constrained")
        for j, (arm, color) in enumerate(zip(arms, colors)):
            values = [frame.loc[(n, arm), metric] * scale for n in (4, 5)]
            ax.bar(np.arange(2) + (j - 1) * .18, values, width=.14, color=color)
        ax.set_xticks([0, 1], ["4 lanes", "5 lanes"])
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
        ax.margins(y=.22)
        if name == "waiting_time":
            ax.set_ylim(0, frame[metric].max() * scale * 1.4)
            ax.legend(handles=[Patch(color=c, label=l) for c, l in zip(colors, labels)], frameon=False, fontsize=14)
        if name == "reward":
            ax.axhline(0, color="grey", lw=.6)
        save(fig, output, f"lanes_{name}")
