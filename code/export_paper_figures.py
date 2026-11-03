from pathlib import Path
import argparse
import json
import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cav_dl.cli import render
from cav_dl.utils.style import configure, save


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Export CSV-based figures with manuscript filenames.")
    parser.add_argument('--data', type=Path, default=project / 'data')
    parser.add_argument('--output', type=Path, default=project / 'outputs/paper')
    args = parser.parse_args()
    panels = args.output / '_panels'
    configure('pdf')
    for figure in range(2, 16):
        render(figure, args.data, panels)
    table = pd.read_csv(args.data / 'fig10.csv')
    with plt.rc_context({'font.size': 13, 'axes.labelsize': 14,
                         'xtick.labelsize': 13, 'ytick.labelsize': 13}):
        fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True, sharey=True,
                                 layout='constrained')
        for ax, compliance in zip(axes.flat, [1., .9, .8, .7, .6, .5]):
            data = table[np.isclose(table.compliance, compliance)].sort_values('p')
            ax.plot(data.p, data.PPO_vs_FS1_TTS_pct, 'o-', color='#bd443d', ms=4, lw=1.4)
            ax.axhline(0, color='grey', ls='--', lw=.7)
            ax.set_title(f'HV compliance = {compliance:.1f}', fontsize=14)
            ax.set(xlim=(0, 1), xticks=[0, .2, .4, .6, .8, 1], ylim=(-6, 22))
            ax.grid(alpha=.15)
        fig.supxlabel('CAV penetration rate', fontsize=16)
        fig.supylabel('Complete TTS reduction relative to FS1 (%)', fontsize=16)
        save(fig, panels / 'fig10', 'TTS_compliance')
    mapping = json.loads((project / 'docs/paper_figures.json').read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    for name, relative in mapping.items():
        shutil.copyfile(panels / relative, args.output / name)
    print(f'Exported {len(mapping)} manuscript figure files to {args.output}')


if __name__ == '__main__':
    main()
