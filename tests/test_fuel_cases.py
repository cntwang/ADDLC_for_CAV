from pathlib import Path
import unittest
import numpy as np
from cav_dl.data.cases import read_manifest
from cav_dl.data.fuel import case_data

class FuelCases(unittest.TestCase):
    def test_saved_manifest_preserves_load_and_merge_share(self):
        cases = read_manifest(Path(__file__).resolve().parents[1] / 'configs/fuel_cases.csv')
        self.assertEqual(len(cases), 330)
        self.assertEqual(cases.traffic_seed.nunique(), 30)
        a, b = [case_data(cases.iloc[i].to_dict()) for i in [0, 8]]
        for name in ['init_car', 'input_supply', 'ramp_inflow']:
            np.testing.assert_array_equal(a[name], b[name])
        self.assertAlmostEqual(a['input_supply'].sum(), 51000)
        self.assertAlmostEqual(a['ramp_inflow'].sum(), 9000)
        self.assertAlmostEqual(b['input_supply'] @ b['input_cav_supply'] + b['ramp_inflow'] @ b['ramp_cav_ratio'], 48000)
