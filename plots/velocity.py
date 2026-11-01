import numpy as np
import matplotlib.pyplot as plt
from cav_dl.utils.io import read_table
from cav_dl.utils.style import save

def render(data, output):
    q = read_table(data, ["p", "arm", "kind", "cell", "speed_kmh"])
    arms = ["Full", "Rule", "FS1", "FS2", "FS3", "FS4", "FS5"]
    order = [(a, k) for a in arms for k in ["DL", "GL", "CAV"]]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    fig.subplots_adjust(left=.215, right=.92, bottom=.08, top=.95, wspace=.055, hspace=.18)
    cmap = plt.get_cmap("Spectral").copy()
    cmap.set_bad("white")
    for i, (ax, p) in enumerate(zip(axes.flat, [.4,.5,.6,.7,.8,.9])):
        t = q[np.isclose(q.p, p)].pivot(index=["arm", "kind"], columns="cell", values="speed_kmh").reindex(order)
        im = ax.imshow(t.to_numpy(), aspect="auto", cmap=cmap, vmin=0, vmax=80, extent=(.5,120.5,21,0), interpolation="nearest")
        ax.set_title(f"$p={p:.1f}$")
        ax.set_yticks(np.arange(21)+.5, [f'{"ADDLC" if a=="Full" else "Rule-based" if a=="Rule" else a} | {k} velocity' for a,k in order], fontsize=10)
        ax.set_xticks([30,60,90,120])
        ax.vlines(np.arange(121)+.5, 0, 21, color="black", lw=.25)
        ax.hlines(np.arange(22), .5, 120.5, color="black", lw=.3)
        if i >= 3: ax.set_xlabel("Cell index")
    fig.colorbar(im, cax=fig.add_axes([.94,.17,.013,.67]), label="Velocity (km/h)")
    save(fig, output, "cell_velocity")
