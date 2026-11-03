from pathlib import Path
import argparse
import shutil

from cav_dl.data.illustration import generate
from cav_dl.plots.timeseries import synthetic
from cav_dl.utils.style import configure


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Replay Figure 5 and plot its generated CSVs.")
    parser.add_argument('--output', type=Path, default=project / 'outputs/figure05')
    parser.add_argument('--format', choices=['pdf', 'png', 'svg', 'all'], default='pdf')
    args = parser.parse_args()
    data = args.output / 'data'
    figures = args.output / 'figures'
    generate(project, data)
    configure(args.format)
    synthetic(data / 'fig05.csv', data / 'fig05_input.csv', data / 'fig05_lanes.csv', figures)
    names = ['input_CAV_fraction', 'cumulative_outflow_difference',
             'controlled_area_CAV_fraction', 'density', 'DL_allocation', 'speed']
    formats = ['pdf', 'png', 'svg'] if args.format == 'all' else [args.format]
    for name, letter in zip(names, 'abcdef'):
        for extension in formats:
            shutil.copyfile(figures / f'fig05_{letter}.{extension}',
                            figures / f'ADDLC_{name}.{extension}')
    print(f'Figure 5 generated from the saved scenario: {args.output}')


if __name__ == '__main__':
    main()
