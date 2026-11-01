

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cav_dl.utils.io import export_table


class Exporter:
    

    def __init__(self, workspace, output):
        self.workspace = Path(workspace).resolve()
        self.project = self.workspace / "review_revision_20261015"
        self.output = Path(output).resolve()
        self.sources = {}
        self.tables = []

    def source(self, relative, project=True):
        path = (self.project if project else self.workspace) / relative
        if not path.is_file():
            raise FileNotFoundError(f"Required saved result is missing: {path}")
        self.sources[str(path.relative_to(self.workspace))] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        return path

    def csv(self, relative, project=True):
        return pd.read_csv(self.source(relative, project))

    def write(self, frame, name):
        path = self.output / name
        export_table(frame, path)
        self.tables.append(
            {"file": name, "rows": len(frame), "columns": list(frame.columns),
             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )

    def export(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.basic_tables()
        self.case_timeseries()
        self.pems()
        self.lanes()
        manifest = {
            "reward": "Five terms: density, capacity, safety, time, early; no timeout",
            "sources_sha256": self.sources,
            "tables": self.tables,
            "selection": "Saved outcome-selected examples retained, no reselection",
            "figure05": (
                "Saved p=0.4 transfer-preview case, flow seed 770000000, "
                "training seed 202610152; outflow gain 2936.471974 vehicles. "
                "This is a separate illustrative case, not the 100-case aggregate."
            ),
            "figure15": (
                "Four-lane boundary demand scale 0.8, five-lane 1.0. "
                "Same saved cases and frozen policies; no reselection or retraining."
            ),
        }
        (self.output / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return manifest

    def basic_tables(self):
        items = [
            (
                "output/pdf/Requested_demand_and_convergence/demand_profiles.csv",
                "fig02.csv", False,
            ),
            (
                "reports/PPO_SAC_DQN_five_reward_20261020/"
                "reward_episode_length_plot_data.csv", "fig03.csv", True,
            ),
            (
                "evaluations/ablation_same_saved100_20261023/plot_statistics.csv",
                "fig04_test.csv", True,
            ),
            (
                "output/pdf/cell_velocity_recomputed_20261027/"
                "cell_velocity_values.csv", "fig06.csv", True,
            ),
            (
                "output/matched_clean_figures_20261023/"
                "spatial_density_speed_values.csv", "fig07.csv", True,
            ),
            (
                "output/matched_clean_figures_20261023/density_dispersion_RMSE.csv",
                "fig08.csv", True,
            ),
            (
                "output/fuel_source_all_vehicles_rule_p03_p09_20261025/"
                "sum_over_10_matched_cases_per_p.csv", "fig09.csv", True,
            ),
            (
                "evaluations/curated100_all_compliance_20261023/"
                "comparison_summary.csv", "fig10.csv", True,
            ),
        ]
        for source, name, project in items:
            self.write(self.csv(source, project), name)
        curves = self.csv("evaluations/five_reward_step15_20261020/training_curves.csv")
        rows = []
        for arm, group in curves.groupby("arm"):
            table = group.pivot(index="steps", columns="seed", values="J_full5")
            rows.append(pd.DataFrame({
                "arm": arm, "steps": table.index,
                "mean": table.mean(axis=1), "seed_sd": table.std(axis=1, ddof=0),
            }))
        self.write(pd.concat(rows, ignore_index=True), "fig04_training.csv")

    def case_timeseries(self):
        folder = (
            "cav40_conservative_mixing/deliverables/"
            "cav40_transfer_preview/selected_case/"
        )
        self.write(self.csv(folder + "plot_timeseries_15min.csv", False), "fig05.csv")
        self.write(self.csv(folder + "input_every_15s.csv", False), "fig05_input.csv")
        path = self.source(folder + "replayed_histories.npz", False)
        
        
        with np.load(path, allow_pickle=False) as archive:
            lanes = archive["NoTimeoutPenalty"][:2520, [20, 40, 60, 80], 2]
        t, u = np.indices(lanes.shape)
        self.write(
            pd.DataFrame({
                "time_h": t.ravel() * 15 / 3600,
                "unit": u.ravel() + 1,
                "lanes": lanes.ravel(),
            }), "fig05_lanes.csv",
        )

    def pems(self):
        profiles = []
        for station, inlet in [(717772, "Mainline"), (716699, "On-ramp")]:
            directory = self.workspace / "data/processed_pems14_20261018"
            for file in sorted(directory.glob(f"station_{station}_5min_*.csv")):
                relative = str(file.relative_to(self.workspace))
                frame = self.csv(relative, False)
                stamp = pd.to_datetime(frame.Timestamp)
                profiles.append(pd.DataFrame({
                    "date": stamp.dt.strftime("%Y-%m-%d"),
                    "time_h": stamp.dt.hour + stamp.dt.minute / 60,
                    "inlet": inlet,
                    "flow_veh_h": frame.Flow * 12,
                }))
        if len(profiles) != 28:
            raise ValueError("Expected 14 daily files for each of two PeMS stations")
        self.write(pd.concat(profiles, ignore_index=True), "fig11.csv")

        curves = []
        run = "runs/pems_both_scaled06_five_reward_20261022/main"
        for seed in (202610150, 202610151, 202610152):
            frame = self.csv(f"{run}/PPO_seed{seed}/episodes.csv")
            frame = frame.dropna(subset=["steps", "J_episode"])
            frame["bin"] = ((frame.steps - 1) // 2000).astype(int)
            curves.append(
                frame.groupby("bin").J_episode.mean().reindex(range(500)).to_numpy()
            )
        a = np.array(curves)
        self.write(pd.DataFrame({
            "steps": (np.arange(500) + .5) * 2000,
            "mean": np.nanmean(a, axis=0),
            "seed_sd": np.nanstd(a, axis=0, ddof=1),
        }), "fig12_training.csv")
        prefix = "deliverables/PeMS_both06_Reward_panels_moderate100_CI/"
        self.write(self.csv(prefix + "bootstrap_intervals.csv"), "fig12_penetration.csv")
        self.write(
            self.csv("deliverables/PeMS_ten_weekdays_p05/summary.csv"),
            "fig12_dates.csv",
        )

        folder = "deliverables/PeMS_current_policy_spatial_p05/display_details/"
        traces, actions, heat = [], [], []
        outflows = {}
        for arm, seed in [("FS1", -1), ("PPO", 202610150)]:
            with np.load(
                self.source(f"{folder}case0_{arm}_seed{seed}.npz"),
                allow_pickle=False,
            ) as archive:
                times = archive["time_hours"] + 3
                outflows[arm] = archive["out_cumulative"].copy()
                traces.append(pd.DataFrame({
                    "arm": arm, "time_h": times,
                    "density": archive["detector_density"],
                    "cumulative_outflow": archive["out_cumulative"],
                }))
                if arm == "PPO":
                    lane = archive["lanes"]
                    t, u = np.indices(lane.shape)
                    actions.append(pd.DataFrame({
                        "time_h": 3.5 + t.ravel() * .25,
                        "unit": u.ravel() + 1, "lanes": lane.ravel(),
                    }))
                t, cell = np.indices(archive["density_heat"].shape)
                heat.append(pd.DataFrame({
                    "arm": arm, "time_h": times[t.ravel()],
                    "cell": cell.ravel() + 1,
                    "density": archive["density_heat"].ravel(),
                    "flow": archive["flow_heat"].ravel(),
                    "speed": archive["speed_heat"].ravel(),
                }))
        self.write(pd.concat(traces, ignore_index=True), "fig13.csv")
        self.write(pd.concat(actions, ignore_index=True), "fig13_lanes.csv")
        self.write(pd.DataFrame({
            "time_h": times,
            "outflow_difference": outflows["PPO"] - outflows["FS1"],
        }), "fig13_difference.csv")
        self.write(pd.concat(heat, ignore_index=True), "fig14.csv")

    def lanes(self):
        prefix = "output/lane4_scale08_lane5_scale10_20261030/"
        frames = [
            self.csv(f"{prefix}lane{n}_summary.csv").assign(lanes=n)
            for n in (4, 5)
        ]
        columns = ["lanes", "arm", "native_episode_reward", "complete_tts",
                   "mean_entry_wait_h", "fixed_departures"]
        self.write(pd.concat(frames, ignore_index=True)[columns], "fig15.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Original research workspace")
    parser.add_argument("--output", default="data", help="CSV output directory")
    args = parser.parse_args()
    manifest = Exporter(args.workspace, args.output).export()
    print(f"Exported {len(manifest['tables'])} tables to {args.output}")


if __name__ == "__main__":
    main()
