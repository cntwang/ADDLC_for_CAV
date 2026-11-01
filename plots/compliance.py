

import numpy as np
import matplotlib.pyplot as plt

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


def render(data, output):
    table = read_table(data, ["p", "compliance", "PPO_vs_FS1_TTS_pct"])
    for letter, c in zip("abcdef", [1., .9, .8, .7, .6, .5]):
        q = table[np.isclose(table.compliance, c)].sort_values("p")
        
        q = q[q.p.gt(0)]
        fig, ax = plt.subplots(figsize=(5.5, 4), layout="constrained")
        ax.plot(q.p, q.PPO_vs_FS1_TTS_pct, "o-", color="#bd443d", label="PPO vs FS1")
        ax.axhline(0, color="grey", ls="--", lw=.8)
        ax.set(xlabel="CAV penetration rate", ylabel="Complete TTS reduction (%)",
               xlim=(0, 1), ylim=(-6, 22), xticks=[0, .2, .4, .6, .8, 1])
        ax.legend(fontsize=13)
        ax.grid(alpha=.15)
        save(fig, output, f"fig10_{letter}")
