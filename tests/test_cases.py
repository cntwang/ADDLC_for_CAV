from pathlib import Path
import tempfile
import unittest

import pandas as pd

from cav_dl.data.cases import read_manifest


class ScenarioManifest(unittest.TestCase):
    def test_legacy_and_current_names_describe_the_same_case(self):
        with tempfile.TemporaryDirectory() as directory:
            current = Path(directory) / 'current.csv'
            legacy = Path(directory) / 'legacy.csv'
            current.write_text('case_id,traffic_seed,p,main,ramp\n7,840000007,0.4,3,0\n')
            legacy.write_text('case,flow,p,main,ramp\n7,840000007,0.4,3,0\n')
            pd.testing.assert_frame_equal(read_manifest(current), read_manifest(legacy))

    def test_fractional_seed_cannot_silently_change_the_demand(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.csv'
            path.write_text('case_id,traffic_seed,p,main,ramp\n7,840000007.5,0.4,3,0\n')
            with self.assertRaisesRegex(ValueError, 'traffic_seed'):
                read_manifest(path)

    def test_duplicate_ids_cannot_overwrite_a_saved_case(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'case.csv'
            path.write_text('case_id,traffic_seed,p,main,ramp\n7,840000007,0.4,3,0\n7,840000008,0.4,3,0\n')
            with self.assertRaisesRegex(ValueError, 'case_id'):
                read_manifest(path)
