from pathlib import Path
import argparse
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
    mapping = {'Demand_fixed_peak.pdf': 'fig02/fig02_a.pdf', 'Demand_random_peak.pdf': 'fig02/fig02_b.pdf', 'Demand_multiple_peaks.pdf': 'fig02/fig02_c.pdf', 'Demand_random_steps.pdf': 'fig02/fig02_d.pdf', 'Demand_correlated_noise.pdf': 'fig02/fig02_e.pdf', 'Demand_independent_noise.pdf': 'fig02/fig02_f.pdf', 'a_reward.pdf': 'fig03/fig03_a.pdf', 'b_episode_length.pdf': 'fig03/fig03_b.pdf', 'reward_ablation_a.pdf': 'fig04/reward_ablation_a.pdf', 'reward_ablation_b.pdf': 'fig04/reward_ablation_b.pdf', 'reward_ablation_c.pdf': 'fig04/reward_ablation_c.pdf', 'reward_ablation_d.pdf': 'fig04/reward_ablation_d.pdf', 'reward_ablation_e.pdf': 'fig04/reward_ablation_e.pdf', 'reward_ablation_f.pdf': 'fig04/reward_ablation_f.pdf', 'reward_ablation_g.pdf': 'fig04/reward_ablation_g.pdf', 'ADDLC_input_CAV_fraction.pdf': 'fig05/fig05_a.pdf', 'ADDLC_cumulative_outflow_difference.pdf': 'fig05/fig05_b.pdf', 'ADDLC_controlled_area_CAV_fraction.pdf': 'fig05/fig05_c.pdf', 'ADDLC_density.pdf': 'fig05/fig05_d.pdf', 'ADDLC_DL_allocation.pdf': 'fig05/fig05_e.pdf', 'ADDLC_speed.pdf': 'fig05/fig05_f.pdf', 'cell_velocity.pdf': 'fig06/cell_velocity.pdf', 'd_va.pdf': 'fig07/fig07_a.pdf', 'd_vb.pdf': 'fig07/fig07_b.pdf', 'Density_dispersion_a.pdf': 'fig08/fig08_a.pdf', 'Density_dispersion_b.pdf': 'fig08/fig08_b.pdf', 'fuel.pdf': 'fig09/fig09.pdf', 'TTS_compliance.pdf': 'fig10/TTS_compliance.pdf', 'PeMS_mainline.pdf': 'fig11/fig11_a.pdf', 'PeMS_on_ramp.pdf': 'fig11/fig11_b.pdf', 'reward_a_training.pdf': 'fig12/fig12_a.pdf', 'reward_CAV_95CI.pdf': 'fig12/fig12_b.pdf', 'reward_p05_date.pdf': 'fig12/fig12_c.pdf', 'reward_p05_week.pdf': 'fig12/fig12_d.pdf', 'bottleneck_density_DL_control.pdf': 'fig13/fig13_a.pdf', 'cumulative_outflow_difference.pdf': 'fig13/fig13_b.pdf', 'density_FS1.pdf': 'fig14/fig14_a.pdf', 'flow_FS1.pdf': 'fig14/fig14_b.pdf', 'speed_FS1.pdf': 'fig14/fig14_c.pdf', 'density_ADDLC.pdf': 'fig14/fig14_d.pdf', 'flow_ADDLC.pdf': 'fig14/fig14_e.pdf', 'speed_ADDLC.pdf': 'fig14/fig14_f.pdf', 'lanes_reward.pdf': 'fig15/lanes_reward.pdf', 'lanes_travel_time.pdf': 'fig15/lanes_travel_time.pdf', 'lanes_waiting_time.pdf': 'fig15/lanes_waiting_time.pdf', 'lanes_throughput.pdf': 'fig15/lanes_throughput.pdf'}
    args.output.mkdir(parents=True, exist_ok=True)
    for name, relative in mapping.items():
        shutil.copyfile(panels / relative, args.output / name)
    print(f'Exported {len(mapping)} manuscript figure files to {args.output}')


if __name__ == '__main__':
    main()
