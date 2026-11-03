import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from cav_dl.utils.io import read_table, export_table

class TableValidation(unittest.TestCase):
    def test_missing_required_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"input.csv"
            path.write_text("arm,value\nFull,1\n")
            with self.assertRaises(ValueError):
                read_table(path, ["arm", "metric"])

    def test_csv_roundtrip_preserves_sign_and_missing_exposure(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.csv"
            original = pd.DataFrame({
                "reward": [-28.136443726, 10.639638],
                "speed": [np.nan, 79.99999999999999],
            })
            export_table(original, path, required=["reward", "speed"])
            got = read_table(path, ["reward", "speed"])
            np.testing.assert_allclose(
                got.to_numpy(), original.to_numpy(), atol=1e-12, rtol=1e-12,
                equal_nan=True,
            )

    def test_export_rejects_incomplete_contract_before_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.csv"
            with self.assertRaises(ValueError):
                export_table(pd.DataFrame({"arm": ["Full"]}), path, ["mean"])
            self.assertFalse(path.exists())
