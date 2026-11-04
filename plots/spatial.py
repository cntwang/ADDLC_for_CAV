

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
    ps = [.1, .3, .5, .7, .9]
    styles = ['o-', 's--', '^-', 'v--', 'D:', 'P-.', 'x-']
    with plt.rc_context({'font.size': 16, 'axes.titlesize': 18, 'axes.labelsize': 17,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15,
                         'mathtext.fontset': 'stix'}):
        for letter, metric in [('a', 'density'), ('b', 'speed')]:
            speed = metric == 'speed'
            fig, axes = plt.subplots(1, 5, figsize=(12.5, 3.0 if speed else 2.5), sharey=True)
            for ax, p in zip(axes, ps):
                for arm, color, style in zip(ARMS, COLORS, styles):
                    q = table[np.isclose(table.p, p) & table.arm.eq(arm)].sort_values('unit')
                    if q.unit.tolist() != list(range(6)):
                        raise ValueError(f'Missing unit profile: p={p}, arm={arm}')
                    ax.plot(q.unit, q[metric], style, color=color, label=label(arm), lw=1.5, ms=4)
                if not speed:
                    ax.set_title(f'$p={p:.1f}$', pad=8)
                ax.set_xticks(range(6))
                ax.grid(axis='y', alpha=.17)
                ax.tick_params(length=3)
            if speed:
                axes[0].set_ylim(0, 85)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc='lower center', ncol=7, fontsize=12,
                           frameon=False, bbox_to_anchor=(.52, -.005))
                fig.supxlabel('Unit index (upstream to downstream)', y=.15, fontsize=17)
            else:
                visible = np.any(np.isclose(table.p.to_numpy()[:, None], ps), axis=1)
                axes[0].set_ylim(0, table.loc[visible, 'density'].max() * 1.08)
            fig.subplots_adjust(left=.04, right=.995, bottom=.34 if speed else .14,
                                top=.97 if speed else .84, wspace=.17)
            save(fig, output, f'fig09_{letter}')


def density_metrics(data, output):
    table = read_table(data, ["p", "arm", "density_std_units", "critical_density_RMSE"])
    panels = [
        ("a", "density_std_units", "Density standard deviation\n(veh/km/lane)"),
        ("b", "critical_density_RMSE", "Critical-density RMSE\n(veh/km/lane)"),
    ]
    ps = sorted(table.p.unique())
    with plt.rc_context({'font.size': 20, 'axes.labelsize': 22,
                         'xtick.labelsize': 20, 'ytick.labelsize': 20}):
        for letter, metric, unit in panels:
            fig, ax = plt.subplots(figsize=(8.6, 4.9), layout='constrained')
            width = .79 / len(ARMS)
            for j, (arm, color) in enumerate(zip(ARMS, COLORS)):
                q = table[table.arm.eq(arm)].set_index('p').loc[ps]
                ax.bar(np.arange(len(ps)) + (j - 3) * width, q[metric], width,
                       color=color, label=label(arm), edgecolor='#333333', lw=.4)
            ax.set_xticks(range(len(ps)), [f'{p:.1f}' for p in ps])
            ax.set(xlabel='CAV penetration rate', ylabel=unit)
            ax.legend(fontsize=16, ncol=2, loc='upper right')
            ax.set_ylim(0, table[metric].max() * 1.28)
            ax.grid(axis='y', alpha=.15)
            ax.set_axisbelow(True)
            save(fig, output, f'fig10_{letter}')



def fuel(data, output):
    table = read_table(data, ["p", "arm", "mean_fuel_kg_per_vehicle"])
    arms = ["Full", "Rule"] + [f"FS{i}" for i in range(1, 10)]
    colors = ["#ff8a72", "#428f89", "#ffba78", "#f6b45c", "#f7d281",
              "#bcd3cc", "#9bc9d0", "#71b3c5", "#4991aa", "#346988", "#244c72"]
    ps = sorted(table.p.unique())
    fig, ax = plt.subplots(figsize=(15, 5.5), layout="constrained")
    width = .79 / len(arms)
    for j, (arm, color) in enumerate(zip(arms, colors)):
        q = table[table.arm.eq(arm)].set_index("p").loc[ps]
        x = np.arange(len(ps)) + (j - (len(arms) - 1) / 2) * width
        ax.bar(x, q.mean_fuel_kg_per_vehicle, width * .94, color=color,
               label=label(arm), edgecolor="#333333", lw=.4)
    ax.set_xticks(range(len(ps)), [f"{p:.1f}" for p in ps])
    ax.set(xlabel="CAV penetration rate", ylabel="Average fuel consumption (kg/veh)")
    ax.legend(fontsize=12, ncol=6)
    ax.set_ylim(0, table.mean_fuel_kg_per_vehicle.max() * 1.25)
    ax.grid(axis="y", alpha=.15)
    ax.set_axisbelow(True)
    save(fig, output, "fig07")
