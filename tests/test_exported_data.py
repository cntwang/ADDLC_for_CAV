"""Optional checks against the separately distributed CSV data bundle."""

import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


@unittest.skipUnless(os.environ.get("CAV_DL_CSV_DIR"), "CSV bundle not supplied")
class ExportedResults(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ["CAV_DL_CSV_DIR"])

    def test_lane_and_cell_grids_are_complete(self):
        lane = pd.read_csv(self.root / "fig15.csv")
        self.assertEqual(len(lane), 6)
        self.assertEqual(len(lane[["lanes", "arm"]].drop_duplicates()), 6)
        speed = pd.read_csv(self.root / "fig08.csv")
        self.assertEqual(set(np.round(speed.p.unique(), 1)), {i / 10 for i in range(1, 10)})
        self.assertEqual(len(speed), 9 * 7 * 3 * 120)
        self.assertEqual(
            len(speed[["p", "arm", "kind", "cell"]].drop_duplicates()), len(speed)
        )

    def test_velocity_matches_distance_over_exposure(self):
        q = pd.read_csv(self.root / "fig08.csv")
        mask = q.vehicle_hours.gt(1e-10)
        np.testing.assert_allclose(
            q.loc[mask, "speed_kmh"],
            q.loc[mask, "distance_km"] / q.loc[mask, "vehicle_hours"],
            rtol=1e-12, atol=1e-10,
        )
        self.assertTrue(q.loc[~mask, "speed_kmh"].isna().all())

    def test_downstream_difference_uses_paired_cumulative_exits(self):
        trace = pd.read_csv(self.root / "fig13.csv")
        wide = trace.pivot(index="time_h", columns="arm", values="cumulative_outflow")
        gain = pd.read_csv(self.root / "fig13_difference.csv").set_index("time_h")
        np.testing.assert_allclose(
            gain.outflow_difference, wide.PPO - wide.FS1, atol=1e-8, rtol=1e-12,
        )

    def test_ablation_keeps_unfavorable_values(self):
        q = pd.read_csv(self.root / "fig04_test.csv")
        reward = q[q.metric.eq("J_fixed_control")].set_index("arm")
        self.assertLess(reward.loc["NoEarlyPenalty", "mean"], 0)
        self.assertTrue((q.low <= q["mean"]).all())
        self.assertTrue((q.high >= q["mean"]).all())
