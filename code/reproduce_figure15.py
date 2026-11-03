from pathlib import Path
import argparse

from cav_dl.data.figure_data import Builder
from cav_dl.experiments import run
from cav_dl.plots.lanes import render
from cav_dl.utils.style import configure


def main():
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Reproduce Figure 15 from saved lane-width scenarios.")
    parser.add_argument('--from-records', type=Path, help="Aggregate existing simulation records")
    parser.add_argument('--output', type=Path, default=project / 'results/figure15')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    records = args.from_records
    if records is None:
        run(project, args.output, {15}, workers=args.workers, resume=args.resume)
        records = args.output
    data = args.output / 'data'
    Builder(project, records, data).build({15})
    configure('pdf')
    render(data / 'fig15.csv', args.output / 'figures')


if __name__ == '__main__':
    main()
