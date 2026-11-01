

import argparse

import pandas as pd

from cav_dl.utils.io import read_table, export_table


def lane_summary(paths, output):
    metrics = ["native_episode_reward", "complete_tts", "mean_entry_wait_h", "fixed_departures"]
    frame = pd.concat([
        read_table(path, ["case_id", "lanes", "arm", *metrics]) for path in paths
    ], ignore_index=True)
    if "training_seed" in frame:
        if frame.duplicated(["case_id", "lanes", "arm", "training_seed"]).any():
            raise ValueError("Duplicate paired replays supplied")
    cases = frame.groupby(["case_id", "lanes", "arm"])[metrics].mean()
    summary = cases.groupby(["lanes", "arm"])[metrics].mean().reset_index()
    export_table(summary, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", nargs="+", required=True)
    parser.add_argument("--output", required=True, help="Write fig15.csv")
    args = parser.parse_args()
    lane_summary(args.raw, args.output)


if __name__ == "__main__":
    main()
