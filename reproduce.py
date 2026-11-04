from pathlib import Path
import argparse
import importlib.util
import runpy
import sys


PROJECT = Path(__file__).resolve().parent
if 'cav_dl' not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        'cav_dl', PROJECT / '__init__.py', submodule_search_locations=[str(PROJECT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['cav_dl'] = package
    spec.loader.exec_module(package)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--figures', type=int, nargs='+', default=list(range(1, 16)))
    parser.add_argument('--output', type=Path, default=PROJECT / 'outputs/reproduction')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--replay', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    figures = set(args.figures)
    if not figures.issubset(set(range(1, 16))):
        parser.error('Figures must be from 1 to 15.')
    from cav_dl.data.figure_data import Builder
    schematic = 1 in figures
    figures.discard(1)
    records = PROJECT / 'records'
    if args.replay:
        from cav_dl.experiments import run
        records = args.output / 'records'
        run(PROJECT, records, figures, workers=args.workers, resume=args.resume)
        if 5 in figures:
            from cav_dl.data.illustration import generate
            generate(PROJECT, records / 'figure05')
    data = args.output / 'data'
    Builder(PROJECT, records, data, PROJECT / 'records/training').build(figures)
    if schematic:
        import shutil
        (args.output / 'figures').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT / 'assets/Fig1.pdf', args.output / 'figures/Fig1.pdf')
    previous = sys.argv
    try:
        sys.argv = ['export_paper_figures.py', '--data', str(data), '--output',
                    str(args.output / 'figures'), '--figures', *map(str, sorted(figures))]
        runpy.run_path(str(PROJECT / 'code/export_paper_figures.py'), run_name='__main__')
    finally:
        sys.argv = previous


if __name__ == '__main__':
    main()
