import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

AVAILABLE = importlib.util.find_spec('numba') is not None


@unittest.skipUnless(AVAILABLE, 'Install cav-dl[rl]')
class Recording(unittest.TestCase):
    def test_controls_keep_first_period_and_lane_values(self):
        from cav_dl.records import Recorder
        from cav_dl.envs._legacy import kernel as k
        history = np.zeros((240, 120, len(k.FIELDS)))
        history[120:180, 20:100, k.A] = 1
        history[180:240, 20:100, k.A] = 2
        with tempfile.TemporaryDirectory() as directory:
            recorder = Recorder(directory, 'case', {'lanes': 3}, 'pems', 3.)
            recorder(SimpleNamespace(_history=history, t=240), {'input_supply': np.zeros(240)})
            controls = pd.read_csv(Path(directory) / 'case_controls.csv')
            self.assertEqual(len(controls), 8)
            self.assertEqual(list(controls.time_h.unique()), [3.5, 3.75])
            self.assertEqual(list(controls.lanes.unique()), [1., 2.])
            self.assertTrue(controls.road_lanes.eq(3).all())

    def test_resume_does_not_append_completed_job(self):
        from cav_dl import experiments
        task = dict(case={'case_id': 1, 'p': .5}, series='synthetic', case_id=1,
                    arm='FS1', training_seed=-1, lanes=3, compliance=1.,
                    native=False, mode='', job_id='case1')
        result = {key: task[key] for key in ['series', 'job_id', *experiments.KEY]}
        result.update(p=.5, physical_failure=False, success=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'models').mkdir()
            (root / 'src').mkdir()
            (root / 'models/index.json').write_text('[]')
            with patch.object(experiments, 'plan', return_value=[task]), \
                 patch.object(experiments, 'initialize'), \
                 patch.object(experiments, 'execute', return_value=result):
                experiments.run(root, root / 'output', {8}, workers=1)
            with patch.object(experiments, 'plan', return_value=[task]), \
                 patch.object(experiments, 'initialize'), \
                 patch.object(experiments, 'execute', side_effect=AssertionError('Repeated job')):
                experiments.run(root, root / 'output', {8}, workers=1, resume=True)
            frame = pd.read_csv(root / 'output/synthetic/raw.csv')
            self.assertEqual(len(frame), 1)
            self.assertEqual(json.loads((root / 'output/complete.json').read_text())['completed'], 1)
