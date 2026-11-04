from pathlib import Path
import argparse
from cav_dl.experiments import run
from cav_dl.data.figure_data import Builder
from cav_dl.cli import render
from cav_dl.utils.style import configure


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduce Figures 8--10 with the main PPO models.")
    parser.add_argument('--from-records', type=Path)
    parser.add_argument('--output', type=Path, default=project / 'outputs/spatial')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    records = args.from_records
    if records is None:
        records = args.output / 'records'
        run(project, records, {8, 9, 10}, args.workers, resume=args.resume)
    data = args.output / 'data'
    Builder(project, records, data).build({8, 9, 10})
    configure('pdf')
    for number in [8, 9, 10]:
        render(number, data, args.output / 'figures')


if __name__ == '__main__':
    main()
