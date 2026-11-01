

import matplotlib.pyplot as plt

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


def synthetic(data, output):
    table = read_table(data, ["mode", "hours", "main_veh_h", "ramp_veh_h"])
    titles = [
        "Fixed peak", "Random peak", "Multiple peaks", "Random steps",
        "Correlated noise", "Independent noise",
    ]
    for letter, (mode, title) in enumerate(zip(table["mode"].unique(), titles)):
        group = table[table["mode"].eq(mode)].sort_values("hours")
        fig, ax = plt.subplots(figsize=(6, 3.6), layout="constrained")
        twin = ax.twinx()
        ax.plot(group.hours, group.main_veh_h, color="#2675ae", lw=.7)
        twin.plot(group.hours, group.ramp_veh_h, color="#e17b24", lw=.7)
        ax.set(
            xlim=(0, 10.5), ylim=(4000, 14000), xlabel="Simulation time (h)",
            ylabel="Mainline demand (veh/h)",
        )
        twin.set(ylim=(150, 900), ylabel="On-ramp demand (veh/h)")
        ax.legend(
            [ax.lines[0], twin.lines[0]], ["Mainline", "On-ramp"], fontsize=11,
        )
        ax.grid(alpha=.15)
        save(fig, output, f"fig02_{chr(97 + letter)}")


def empirical(data, output):
    table = read_table(data, ["date", "time_h", "inlet", "flow_veh_h"])
    colors = plt.get_cmap("tab20")
    for letter, inlet in zip("ab", ["Mainline", "On-ramp"]):
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        groups = table[table.inlet.eq(inlet)].groupby("date", sort=True)
        for i, (date, group) in enumerate(groups):
            group = group.sort_values("time_h")
            ax.plot(
                group.time_h, group.flow_veh_h, color=colors(i),
                lw=1, label=str(date)[5:],
            )
        ax.set(xlim=(0, 24), xlabel="Time (h)", ylabel="Flow (veh/h)")
        ax.legend(fontsize=10, ncol=3, title="Date (2025, MM-DD)", title_fontsize=11)
        ax.grid(alpha=.15)
        save(fig, output, f"fig11_{letter}")
