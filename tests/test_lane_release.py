"""Check that lane-width comparisons use the released models and full cohorts."""
import json
from pathlib import Path
import unittest

from cav_dl.experiments import plan
from cav_dl.utils.seeds import SEEDS, LANE_SEEDS


class LaneRelease(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]

    def test_four_lane_seed_selection_does_not_change_other_algorithms(self):
        self.assertEqual(SEEDS, [202610150, 202610151, 202610152])
        self.assertEqual(LANE_SEEDS[4], [202610151, 202610152, 202610153])
        self.assertEqual(LANE_SEEDS[5], SEEDS)

    def test_complete_lane_comparison_plan(self):
        tasks = plan(self.root, {15})
        self.assertEqual(len(tasks), (360 + 100) * 7)
        for width, count in [(4, 360), (5, 100)]:
            subset = [t for t in tasks if t['lanes'] == width]
            self.assertEqual(len({t['case_id'] for t in subset}), count)
            for arm in ['Transfer', 'Retrained']:
                self.assertEqual(sorted({t['training_seed'] for t in subset if t['arm'] == arm}), LANE_SEEDS[width])

    def test_every_released_policy_has_its_matching_normalizer(self):
        index = json.loads((self.root / 'models/index.json').read_text())
        paths = {item['path'] for item in index}
        for width, seeds in LANE_SEEDS.items():
            for seed in seeds:
                for model_width in [3, width]:
                    relative = f'models/pems{model_width}/PPO_seed{seed}'
                    self.assertIn(relative, paths)
                    for name in ['model_final.zip', 'normalizer_final.pkl']:
                        self.assertTrue((self.root / relative / name).is_file())
        self.assertNotIn('models/pems4/PPO_seed202610150', paths)


if __name__ == '__main__':
    unittest.main()
