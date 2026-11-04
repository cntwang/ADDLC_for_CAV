

import matplotlib.pyplot as plt

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


TITLES = [
    "Fixed peak", "Random peak", "Multiple peaks", "Random steps",
    "Correlated noise", "Independent noise",
]


def _synthetic_axes(ax, group, legend=False):
    twin = ax.twinx()
    main, = ax.plot(group.hours, group.main_veh_h, color="#2675ae", lw=.85)
    ramp, = twin.plot(group.hours, group.ramp_veh_h, color="#e17b24", lw=.85)
    ax.set(xlim=(0, 10.5), ylim=(4000, 14000), xticks=[0, 2, 4, 6, 8, 10],
           yticks=[4000, 6000, 8000, 10000, 12000, 14000])
    twin.set(ylim=(150, 900), yticks=[200, 400, 600, 800])
    ax.set_xlabel("Simulation time (h)", fontsize=18, labelpad=6)
    ax.set_ylabel("Mainline demand (veh/h)", fontsize=18, labelpad=5)
    twin.set_ylabel("On-ramp demand (veh/h)", fontsize=18, labelpad=6)
    ax.tick_params(axis='both', labelsize=15, length=3, pad=3)
    twin.tick_params(axis='y', labelsize=15, length=3, pad=3)
    ax.grid(alpha=.2)
    for spine in ax.spines.values():
        spine.set_visible(True)
    twin.spines['right'].set_visible(True)
    twin.spines['top'].set_visible(True)
    if legend:
        ax.legend([main, ramp], ["Mainline", "On-ramp"], loc='upper right',
                  fontsize=13, handlelength=1.4, handletextpad=.4,
                  borderpad=.3, labelspacing=.2, framealpha=.95)
    return twin


def synthetic(data, output):
    table = read_table(data, ["mode", "hours", "main_veh_h", "ramp_veh_h"])
    for index, mode in enumerate(table["mode"].unique()):
        group = table[table["mode"].eq(mode)].sort_values("hours")
        with plt.rc_context({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif']}):
            fig, ax = plt.subplots(figsize=(5.4, 4.7), layout="constrained")
            _synthetic_axes(ax, group, legend=index == 0)
            # Panel captions are placed by LaTeX, avoiding duplicate titles.
            save(fig, output, f"fig02_{chr(97 + index)}")


def synthetic_preview(data, output):
    table = read_table(data, ["mode", "hours", "main_veh_h", "ramp_veh_h"])
    fig, axes = plt.subplots(2, 3, figsize=(19.5, 12), layout="constrained")
    for index, (ax, mode, title) in enumerate(zip(axes.flat, table["mode"].unique(), TITLES)):
        group = table[table["mode"].eq(mode)].sort_values("hours")
        _synthetic_axes(ax, group, legend=index == 0)
        ax.set_title(f"({chr(97 + index)}) {title}", fontsize=21, pad=12)
    save(fig, output, "demand_profiles_preview")


def empirical(data, output):
    table = read_table(data, ["date", "time_h", "inlet", "flow_veh_h"])
    colors = plt.get_cmap("tab20")
    for letter, inlet in zip("ab", ["Mainline", "On-ramp"]):
        fig, ax = plt.subplots(figsize=(7.5, 5.2), layout="constrained")
        groups = table[table.inlet.eq(inlet)].groupby("date", sort=True)
        for i, (date, group) in enumerate(groups):
            group = group.sort_values("time_h")
            ax.plot(
                group.time_h, group.flow_veh_h, color=colors(i),
                lw=1, label=str(date)[5:],
            )
        ax.set(xlim=(0, 24), xticks=[0, 4, 8, 12, 16, 20, 24])
        ax.set_xlabel("Time (h)", fontsize=24)
        ax.set_ylabel("Flow (veh/h)", fontsize=24)
        ax.tick_params(axis="both", labelsize=20)
        if inlet == "Mainline":
            ax.legend(fontsize=18, ncol=3, title="Date (2025, MM-DD)",
                      title_fontsize=20, loc="lower right", bbox_to_anchor=(.88, .015),
                      handlelength=1.1, handletextpad=.3, columnspacing=.5,
                      borderpad=.2, labelspacing=.15, frameon=False)
        ax.grid(alpha=.15)
        save(fig, output, f"fig11_{letter}")
