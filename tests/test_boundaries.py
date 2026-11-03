"""The renderer must not grow a dependency on experimental execution."""

import ast
import unittest
from pathlib import Path

import cav_dl


class RendererBoundary(unittest.TestCase):
    def test_no_model_or_binary_data_loading_in_plots(self):
        root = Path(cav_dl.__file__).parent
        forbidden = {"torch", "stable_baselines3", "numba", "pickle", "subprocess"}
        for path in (root / "plots").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [item.name for item in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    names = []
                for name in names:
                    self.assertNotIn(name.split(".")[0], forbidden, str(path))
                    self.assertNotIn("export", name.split("."), str(path))
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    self.assertNotIn(node.func.attr, {"load", "read_pickle"}, str(path))
