

import argparse
from pathlib import Path

from cav_dl.utils.style import configure


def render(figure, data, output):
    from cav_dl.plots import (
        ablation, compliance, curves, demand, lanes, spatial, timeseries, velocity,
    )

    data, output = Path(data), Path(output) / f"fig{figure:02d}"
    def source(name):
        return data / name
    if figure == 2:
        demand.synthetic(source("fig02.csv"), output)
    elif figure == 3:
        curves.algorithms(source("fig03.csv"), output)
    elif figure == 4:
        ablation.learning(source("fig04_training.csv"), output)
        ablation.render(source("fig04_test.csv"), output)
    elif figure == 5:
        timeseries.synthetic(source("fig05.csv"), source("fig05_input.csv"),
                             source("fig05_lanes.csv"), output)
    elif figure == 6:
        velocity.render(source("fig06.csv"), output)
    elif figure == 7:
        spatial.profiles(source("fig07.csv"), output)
    elif figure == 8:
        spatial.density_metrics(source("fig08.csv"), output)
    elif figure == 9:
        spatial.fuel(source("fig09.csv"), output)
    elif figure == 10:
        compliance.render(source("fig10.csv"), output)
    elif figure == 11:
        demand.empirical(source("fig11.csv"), output)
    elif figure == 12:
        curves.pems(source("fig12_training.csv"), source("fig12_penetration.csv"),
                    source("fig12_dates.csv"), output)
    elif figure == 13:
        timeseries.pems_control(source("fig13.csv"), source("fig13_lanes.csv"),
                               source("fig13_difference.csv"), output)
    elif figure == 14:
        timeseries.pems_heatmaps(source("fig14.csv"), output)
    elif figure == 15:
        lanes.render(source("fig15.csv"), output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", help="Figure number 2–15, or all")
    parser.add_argument("--data", default="data", help="Exported CSV directory")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--format", choices=["pdf", "png", "svg", "all"], default="pdf")
    args = parser.parse_args()
    if args.figure != "all" and args.figure not in {str(n) for n in range(2, 16)}:
        parser.error("figure must be all or an integer from 2 to 15")
    configure(args.format)
    figures = range(2, 16) if args.figure == "all" else [int(args.figure)]
    for figure in figures:
        render(figure, args.data, args.output)
        print(f"Figure {figure}: saved to {Path(args.output) / f'fig{figure:02d}'}")


if __name__ == "__main__":
    main()
