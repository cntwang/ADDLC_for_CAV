import numpy as np
import matplotlib.pyplot as plt
from cav_dl.utils.io import read_table
from cav_dl.utils.style import save


def render(data, output):
    q = read_table(data, ['p', 'arm', 'kind', 'cell', 'speed_kmh'])
    arms = ['Full', 'Rule', 'FS1', 'FS2', 'FS3', 'FS4', 'FS5']
    order = [(a, k) for a in arms for k in ['DL', 'GL', 'CAV']]
    ps = np.arange(1, 10) / 10
    if not all(np.isclose(q.p.unique(), p).any() for p in ps):
        raise ValueError('Cell velocities require p=0.1 through 0.9')
    labels = [f'{"ADDLC" if a == "Full" else "Rule-based" if a == "Rule" else a} | {k} velocity'
              for a, k in order]
    with plt.rc_context({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 14,
                         'xtick.labelsize': 13, 'ytick.labelsize': 10,
                         'mathtext.fontset': 'stix'}):
        fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
        fig.subplots_adjust(left=.19, right=.92, bottom=.065, top=.97, wspace=.055, hspace=.12)
        cmap = plt.get_cmap('Spectral').copy()
        cmap.set_bad('white')
        for i, (ax, p) in enumerate(zip(axes.flat, ps)):
            t = q[np.isclose(q.p, p)].pivot(index=['arm', 'kind'], columns='cell', values='speed_kmh')
            t = t.reindex(index=order, columns=np.arange(1, 121))
            im = ax.imshow(t.to_numpy(), aspect='auto', cmap=cmap, vmin=0, vmax=80,
                           extent=(.5, 120.5, 21, 0), interpolation='nearest')
            ax.set_title(f'$p={p:.1f}$', pad=9)
            ax.set_yticks(np.arange(21) + .5, labels)
            ax.tick_params(axis='y', length=0)
            ax.set_xticks([30, 60, 90, 120])
            ax.vlines(np.arange(121) + .5, 0, 21, color='black', lw=.25)
            ax.hlines(np.arange(22), .5, 120.5, color='black', lw=.3)
            for spine in ax.spines.values():
                spine.set_visible(True)
            if i >= 6:
                ax.set_xlabel('Cell index')
        fig.colorbar(im, cax=fig.add_axes([.94, .16, .013, .69]),
                     label='Velocity (km/h)', ticks=[0, 20, 40, 60, 80])
        save(fig, output, 'cell_velocity', pad_inches=.01)
