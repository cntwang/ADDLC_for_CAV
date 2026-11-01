

import numpy as np
import matplotlib.pyplot as plt

from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


def algorithms(data, output):
    table = read_table(data, ["algorithm", "metric", "steps", "mean", "seed_sd"])
    colors = {"PPO": "#aa393e", "DQN": "#c6a23c", "SAC": "#367fa2"}
    for letter, metric, label in [
        ("a", "reward5", "Common full-five episodic reward"),
        ("b", "controls", "Episode length (control steps)"),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        for arm, color in colors.items():
            group = table[table.algorithm.eq(arm) & table.metric.eq(metric)]
            group = group.sort_values("steps")
            x = group.steps / 1e6
            ax.plot(x, group["mean"], color=color, label=arm)
            ax.fill_between(
                x, group["mean"] - group.seed_sd,
                group["mean"] + group.seed_sd, color=color, alpha=.2,
            )
        ax.set(xlim=(0, 1), xlabel="Training steps ($10^6$)", ylabel=label)
        if metric == "reward5":
            ax.set(ylim=(-40, 20), yticks=[-40, -20, 0, 20])
        ax.legend(fontsize=13)
        ax.grid(alpha=.15)
        save(fig, output, f"fig03_{letter}")


def pems(training, penetration, dates, output):
    train = read_table(training, ["steps", "mean", "seed_sd"])
    ratios = read_table(penetration, ["p", "arm", "mean", "lower95", "upper95"])
    days = read_table(dates, ["scope", "key", "arm", "mean", "lower95", "upper95"])
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    x = train.steps / 1e6
    ax.plot(x, train["mean"], color="#b4494e", label="PPO")
    ax.fill_between(
        x, train["mean"] - train.seed_sd, train["mean"] + train.seed_sd,
        color="#b4494e", alpha=.2,
    )
    ax.set(xlabel="Training steps ($10^6$)", ylabel="Episodic cumulative reward")
    ax.legend(fontsize=13)
    ax.grid(alpha=.15)
    save(fig, output, "fig12_a")
    settings = [
        ("b", ratios, "p", "CAV penetration rate"),
        ("c", days[days.scope.eq("date")], "key", "Date"),
        ("d", days[days.scope.eq("weekday")], "key", "Day of week"),
    ]
    for letter, table, key, xlabel in settings:
        fig, ax = plt.subplots(figsize=(7 if letter == "c" else 6, 4),
                               layout="constrained")
        keys = sorted(table[key].unique(), key=str)
        x = np.array(keys, dtype=float) if key == "p" else np.arange(len(keys))
        for arm, color in [("PPO", "#b4494e"), ("FS1", "#c7a449")]:
            group = table[table.arm.eq(arm)].set_index(key).loc[keys]
            ax.plot(x, group["mean"], "o-", color=color,
                    label="ADDLC" if arm == "PPO" else arm)
            ax.fill_between(x, group.lower95, group.upper95, color=color, alpha=.16)
        if key != "p":
            labels = [str(k)[5:] for k in keys] if letter == "c" else [
                "Mon", "Tue", "Wed", "Thu", "Fri"
            ]
            ax.set_xticks(x, labels, rotation=45 if letter == "c" else 0)
            if letter == "c":
                ax.tick_params(axis="x", labelsize=12)
        ax.set(xlabel=xlabel, ylabel="Episodic cumulative reward")
        ax.legend(fontsize=13)
        ax.grid(alpha=.15)
        save(fig, output, f"fig12_{letter}")
