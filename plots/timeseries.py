

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


def lane_image(ax, frame, extent, origin="upper", colors=None):
    matrix = frame.pivot(index="unit", columns="time_h", values="lanes")
    return ax.imshow(
        matrix.to_numpy(), aspect="auto", interpolation="nearest", extent=extent,
        cmap=ListedColormap(colors or ["#f0f0f0", "#efb3b3", "#9d292f"]),
        norm=BoundaryNorm([-.5, .5, 1.5, 2.5], 3), origin=origin,
    )


def synthetic(data, inputs, lanes, output):
    q = read_table(data, ["time_h", "Full_minus_FS1_outflow"])
    raw = read_table(inputs, ["time_h", "main_CAV_share", "ramp_CAV_share"])
    actions = read_table(lanes, ["time_h", "unit", "lanes"])
    orange, green = "#ff7028", "#2e6f38"
    for letter in "abcdef":
        fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
        x = q.time_h - .125
        if letter == "a":
            for inlet, color in [("main", green), ("ramp", orange)]:
                ax.step(raw.time_h, raw[f"{inlet}_CAV_share"], color=color,
                        alpha=.15, lw=.4, where="post")
                ax.plot(x, q[f"{inlet}_CAV_share"], color=color,
                        label=f"{inlet.capitalize()}, 15-min mean")
            ax.set(ylabel="Input CAV fraction", ylim=(0, 1))
        elif letter == "b":
            ax.plot(q.time_h, q.Full_minus_FS1_outflow, color=orange,
                    label="ADDLC minus FS1")
            ax.axhline(0, color="grey", lw=.7)
            ax.set_ylabel("Cumulative outflow\ndifference (veh)")
        elif letter == "e":
            image = lane_image(ax, actions, [0, 10.5, 4.5, .5],
                               colors=["#fed78b", "#ffad4c", "#ff7028"])
            ax.set(ylabel="Control unit", yticks=[1, 2, 3, 4])
            fig.colorbar(image, ax=ax, ticks=[0, 1, 2], label="Dedicated lanes")
        else:
            metric, unit = {
                "c": ("CAV_share", "Controlled-area CAV fraction"),
                "d": ("density", "Density (veh/km/lane)"),
                "f": ("speed_kmh", "Flow/stock speed (km/h)"),
            }[letter]
            for arm, color in [("Full", orange), ("FS1", green)]:
                ax.plot(x, q[f"{arm}_{metric}"], color=color,
                        ls="-" if arm == "Full" else "--",
                        label="ADDLC" if arm == "Full" else arm)
                if letter == "d":
                    ax.plot(x, q[f"{arm}_critical_density"], color=color, ls=":",
                            label=f"{arm} critical density")
            ax.set_ylabel(unit)
            if letter == "c":
                ax.set_ylim(0, 1)
            if letter == "f":
                ax.set_ylim(0, 85)
        ax.set(xlabel="Simulation time (h)", xlim=(0, 10.5))
        if letter != "e":
            ax.legend(fontsize=10)
            ax.grid(alpha=.15)
        save(fig, output, f"fig05_{letter}")


def pems_control(data, lanes, difference, output):
    q = read_table(data, ["arm", "time_h", "density", "cumulative_outflow"])
    actions = read_table(lanes, ["time_h", "unit", "lanes"])
    gain = read_table(difference, ["time_h", "outflow_difference"])
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), layout="constrained")
    for arm, color in [("FS1", "#c7a449"), ("PPO", "#b4494e")]:
        group = q[q.arm.eq(arm)].sort_values("time_h")
        axes[0].plot(group.time_h, group.density, color=color,
                     label="ADDLC" if arm == "PPO" else arm)
    axes[0].set_ylabel("Density\n(veh/km/lane)")
    axes[0].legend(fontsize=12)
    image = lane_image(axes[1], actions, [3.5, 13.5, .5, 4.5], origin="lower")
    axes[1].set(xlabel="Time (h)", ylabel="Controlled unit", yticks=[1, 2, 3, 4])
    fig.colorbar(image, ax=axes[1], ticks=[0, 1, 2], label="DL count")
    save(fig, output, "fig13_a")
    fig, ax = plt.subplots(figsize=(6.5, 4), layout="constrained")
    ax.plot(gain.time_h, gain.outflow_difference, color="#b4494e",
            label="ADDLC minus FS1")
    ax.axhline(0, color="grey", lw=.8)
    ax.set(xlabel="Time (h)", ylabel="Cumulative outflow difference (veh)")
    ax.grid(alpha=.15)
    ax.legend(fontsize=12)
    save(fig, output, "fig13_b")


def pems_heatmaps(data, output):
    q = read_table(data, ["arm", "time_h", "cell", "density", "flow", "speed"])
    units = {
        "density": "Density (veh/km/lane)", "flow": "Flow (veh/h)",
        "speed": "Speed (km/h)",
    }
    for arm, letters in [("FS1", "abc"), ("PPO", "def")]:
        group = q[q.arm.eq(arm)]
        for letter, (metric, unit) in zip(letters, units.items()):
            matrix = group.pivot(index="cell", columns="time_h", values=metric)
            fig, ax = plt.subplots(figsize=(6.5, 4.5), layout="constrained")
            
            low, high = q[metric].min(), q[metric].max()
            image = ax.imshow(
                matrix.to_numpy(), origin="upper", aspect="auto",
                extent=[matrix.columns.min(), matrix.columns.max(), 120, 0],
                vmin=low, vmax=high, cmap="viridis", interpolation="nearest",
            )
            ax.set(xlabel="Time (h)", ylabel="Cell index")
            fig.colorbar(image, ax=ax, label=unit)
            save(fig, output, f"fig14_{letter}")
