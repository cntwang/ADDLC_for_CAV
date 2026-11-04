"""Export manuscript figures from generated CSVs."""
from pathlib import Path
import argparse
import shutil
from cav_dl.cli import render
from cav_dl.utils.style import configure

FILES = {'Demand_fixed_peak.pdf': 'fig02/fig02_a.pdf',
 'Demand_random_peak.pdf': 'fig02/fig02_b.pdf',
 'Demand_multiple_peaks.pdf': 'fig02/fig02_c.pdf',
 'Demand_random_steps.pdf': 'fig02/fig02_d.pdf',
 'Demand_correlated_noise.pdf': 'fig02/fig02_e.pdf',
 'Demand_independent_noise.pdf': 'fig02/fig02_f.pdf',
 'a_reward.pdf': 'fig03/fig03_a.pdf',
 'b_episode_length.pdf': 'fig03/fig03_b.pdf',
 'reward_ablation_a.pdf': 'fig04/reward_ablation_a.pdf',
 'reward_ablation_b.pdf': 'fig04/reward_ablation_b.pdf',
 'reward_ablation_c.pdf': 'fig04/reward_ablation_c.pdf',
 'reward_ablation_d.pdf': 'fig04/reward_ablation_d.pdf',
 'reward_ablation_e.pdf': 'fig04/reward_ablation_e.pdf',
 'reward_ablation_f.pdf': 'fig04/reward_ablation_f.pdf',
 'reward_ablation_g.pdf': 'fig04/reward_ablation_g.pdf',
 'ADDLC_input_CAV_fraction.pdf': 'fig05/fig05_a.pdf',
 'ADDLC_cumulative_outflow_difference.pdf': 'fig05/fig05_b.pdf',
 'ADDLC_controlled_area_CAV_fraction.pdf': 'fig05/fig05_c.pdf',
 'ADDLC_density.pdf': 'fig05/fig05_d.pdf',
 'ADDLC_DL_allocation.pdf': 'fig05/fig05_e.pdf',
 'ADDLC_speed.pdf': 'fig05/fig05_f.pdf',
 'cell_velocity.pdf': 'fig08/cell_velocity.pdf',
 'd_va.pdf': 'fig09/fig09_a.pdf',
 'd_vb.pdf': 'fig09/fig09_b.pdf',
 'Density_dispersion_a.pdf': 'fig10/fig10_a.pdf',
 'Density_dispersion_b.pdf': 'fig10/fig10_b.pdf',
 'TTS_compliance.pdf': 'fig06/TTS_compliance.pdf',
 'PeMS_mainline.pdf': 'fig11/fig11_a.pdf',
 'PeMS_on_ramp.pdf': 'fig11/fig11_b.pdf',
 'reward_a_training.pdf': 'fig12/fig12_a.pdf',
 'reward_CAV_95CI.pdf': 'fig12/fig12_b.pdf',
 'reward_p05_date.pdf': 'fig12/fig12_c.pdf',
 'reward_p05_week.pdf': 'fig12/fig12_d.pdf',
 'bottleneck_density_DL_control.pdf': 'fig13/fig13_a.pdf',
 'cumulative_outflow_difference.pdf': 'fig13/fig13_b.pdf',
 'density_FS1.pdf': 'fig14/fig14_a.pdf',
 'flow_FS1.pdf': 'fig14/fig14_b.pdf',
 'speed_FS1.pdf': 'fig14/fig14_c.pdf',
 'density_ADDLC.pdf': 'fig14/fig14_d.pdf',
 'flow_ADDLC.pdf': 'fig14/fig14_e.pdf',
 'speed_ADDLC.pdf': 'fig14/fig14_f.pdf',
 'lanes_reward.pdf': 'fig15/lanes_reward.pdf',
 'lanes_travel_time.pdf': 'fig15/lanes_travel_time.pdf',
 'lanes_waiting_time.pdf': 'fig15/lanes_waiting_time.pdf',
 'lanes_throughput.pdf': 'fig15/lanes_throughput.pdf',
 'fuel.pdf': 'fig07/fig07.pdf'}


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=project / 'data')
    parser.add_argument('--output', type=Path, default=project / 'outputs/paper')
    parser.add_argument('--figures', type=int, nargs='+', default=list(range(2, 16)))
    args = parser.parse_args()
    if not set(args.figures).issubset(set(range(2, 16))):
        parser.error('Figures must be from 2 to 15.')
    panels = args.output / '_panels'
    configure('pdf')
    for figure in sorted(set(args.figures)):
        render(figure, args.data, panels)
    args.output.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, relative in FILES.items():
        number = int(relative[3:5])
        if number not in args.figures:
            continue
        source = panels / relative
        shutil.copyfile(source, args.output / name)
        count += 1
    print(f'Exported {count} manuscript figure files to {args.output}')


if __name__ == '__main__':
    main()
