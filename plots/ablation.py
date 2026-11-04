"""Five-term reward ablation, using common training and evaluation scores."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from cav_dl.utils.io import read_table
from cav_dl.utils.style import save

ARMS = ['Full', 'NoDensity', 'NoCapacity', 'NoSafety', 'NoTime', 'NoEarlyPenalty']
COLORS = ['#bd403a', '#dc9136', '#3d8d74', '#8669a8', '#477fa7', '#88705c']
STYLE = {'font.size': 19, 'axes.titlesize': 20, 'axes.labelsize': 19,
         'xtick.labelsize': 17, 'ytick.labelsize': 17, 'legend.fontsize': 15}


def render(data, output):
    frame = read_table(data, ['metric', 'arm', 'mean', 'low', 'high'])
    metrics = [('J_fixed_control', 'Reward'), ('complete_tts', r'$10^5$ veh h'),
               ('mean_entry_wait_h', 'min/vehicle'), ('fixed_departures', r'$10^3$ vehicles'),
               ('full_high_occupancy_vehicle_hours', 'veh h'), ('executed_lane_switches', 'Count')]
    with plt.rc_context(STYLE):
        for letter, (metric, unit) in zip('bcdefg', metrics):
            q = frame[frame.metric.eq(metric)].set_index('arm').loc[ARMS]
            m, lo, hi = (q[c].to_numpy() for c in ['mean', 'low', 'high'])
            fig, ax = plt.subplots(figsize=(5.6, 3.8))
            fig.subplots_adjust(left=.23, right=.97, bottom=.2, top=.96)
            ax.bar(range(6), m, width=.68, color=COLORS, edgecolor='white', lw=.5,
                   yerr=np.array([m-lo, hi-m]),
                   error_kw=dict(ecolor='#333333', capsize=3, elinewidth=1.1))
            span = max(hi.max()-lo.min(), abs(hi.max())*.001)
            bottom = lo.min() - .16*span
            if lo.min() >= 0:
                bottom = max(0., bottom)
            ax.set_ylim(bottom, hi.max()+.16*span)
            ax.set_xticks(range(6), ['Full', '-D', '-C', '-S', '-T', '-E'])
            ax.set_ylabel(unit)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.ticklabel_format(axis='y', style='plain', useOffset=False)
            ax.grid(axis='y', alpha=.2)
            if bottom > 0:
                kw = dict(transform=ax.transAxes, color='#333333', clip_on=False, lw=1.1)
                ax.plot((-.013, .013), (-.018, .018), **kw)
                ax.plot((-.013, .013), (.025, .061), **kw)
            save(fig, output, f'reward_ablation_{letter}', pad_inches=3/72)


def learning(data, output):
    q = read_table(data, ['arm', 'steps', 'mean', 'seed_sd'])
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(5.6, 9.8))
        fig.subplots_adjust(left=.22, right=.97, bottom=.12, top=.94)
        for arm, color in zip(ARMS, COLORS):
            t = q[q.arm.eq(arm)].sort_values('steps')
            mean, sd = t['mean'], t.seed_sd
            ax.plot(t.steps/1e6, mean, color=color, lw=2, label=arm)
            ax.fill_between(t.steps/1e6, mean-sd, mean+sd, color=color, alpha=.12, lw=0)
        ax.set(xlim=(0, 1), ylim=(-40, 20), yticks=[-40, -20, 0, 20],
               xlabel=r'Training steps ($\times10^6$)', ylabel='Common full-five episode reward')
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.grid(alpha=.2)
        ax.legend(loc='lower right', framealpha=.95)
        save(fig, output, 'reward_ablation_a', pad_inches=3/72)
