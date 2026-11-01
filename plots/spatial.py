

import numpy as np
import matplotlib.pyplot as plt

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save

ARMS = ["Full", "Rule", "FS1", "FS2", "FS3", "FS4", "FS5"]
COLORS = ["#49666d", "#398f86", "#a5c99a", "#cdd09a", "#f4d48a",
          "#ffbf85", "#ff8b74"]


def label(arm):
    return {"Full": "ADDLC", "Rule": "Rule-based"}.get(arm, arm)


def profiles(data, output):
    table = read_table(data, ["p", "arm", "unit", "density", "speed"])
    for letter, metric, unit in [
        ("a", "density", "Density (veh/km/lane)"),
        ("b", "speed", "Speed (km/h)"),
    ]:
        fig, axes = plt.subplots(1, 6, figsize=(19, 4), sharey=True)
        for ax, p in zip(axes, [.4, .5, .6, .7, .8, .9]):
            for arm, color in zip(ARMS, COLORS):
                q = table[np.isclose(table.p, p) & table.arm.eq(arm)]
                q = q.sort_values("unit")
                ax.plot(q.unit, q[metric], "o-", color=color, label=label(arm), ms=4)
            ax.set_title(f"$p={p:.1f}$")
            ax.set_xticks([0, 2, 5])
            ax.set_ylim(bottom=0)
            ax.grid(alpha=.15)
        axes[0].set_ylabel(unit)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=12)
        fig.supxlabel("Unit index (upstream to downstream)", y=.1)
        fig.subplots_adjust(left=.065, right=.99, bottom=.3, top=.85, wspace=.12)
        save(fig, output, f"fig07_{letter}")


def density_metrics(data, output):
    table = read_table(data, ["p", "arm", "density_std_units", "critical_density_RMSE"])
    panels = [
        ("a", "density_std_units", "Density standard deviation\n(veh/km/lane)"),
        ("b", "critical_density_RMSE", "Critical-density RMSE\n(veh/km/lane)"),
    ]
    ps = sorted(table.p.unique())
    for letter, metric, unit in panels:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        width = .79 / len(ARMS)
        for j, (arm, color) in enumerate(zip(ARMS, COLORS)):
            q = table[table.arm.eq(arm)].set_index("p").loc[ps]
            ax.bar(np.arange(len(ps)) + (j - 3) * width, q[metric], width,
                   color=color, label=label(arm), edgecolor="#333333", lw=.4)
        ax.set_xticks(range(len(ps)), [f"{p:.1f}" for p in ps])
        ax.set(xlabel="CAV penetration rate", ylabel=unit)
        ax.legend(fontsize=11, ncol=2)
        ax.set_ylim(0, table[metric].max() * 1.3)
        ax.grid(axis="y", alpha=.15)
        ax.set_axisbelow(True)
        save(fig, output, f"fig08_{letter}")


def fuel(data, output):
    table = read_table(data, ["p", "arm", "source_total_fuel_kg"])
    arms = ["Full", "Rule"] + [f"FS{i}" for i in range(1, 10)]
    colors = ["#ff8a72", "#428f89", "#ffba78", "#f6b45c", "#f7d281",
              "#bcd3cc", "#9bc9d0", "#71b3c5", "#4991aa", "#346988", "#244c72"]
    ps = sorted(table.p.unique())
    fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
    width = .79 / len(arms)
    for j, (arm, color) in enumerate(zip(arms, colors)):
        q = table[table.arm.eq(arm)].set_index("p").loc[ps]
        x = np.arange(len(ps)) + (j - (len(arms) - 1) / 2) * width
        ax.bar(x, q.source_total_fuel_kg / 1000, width * .94, color=color,
               label=label(arm), edgecolor="#333333", lw=.4)
    ax.set_xticks(range(len(ps)), [f"{p:.1f}" for p in ps])
    ax.set(xlabel="CAV penetration rate", ylabel="Total fuel consumption ($10^3$ kg)")
    ax.legend(fontsize=12, ncol=6)
    ax.set_ylim(0, table.source_total_fuel_kg.max() / 1000 * 1.25)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, output, "fig09")
