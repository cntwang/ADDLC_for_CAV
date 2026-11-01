

import importlib
from types import SimpleNamespace

import numpy as np

from .clearance import ClearanceNative
from ._legacy import kernel as k
from .stay_helpers import dl_share


class StayNative(ClearanceNative):
    def __init__(self, compliance=1., lanes=3, max_dl=None):
        if lanes != 3:
            raise ValueError("The HV persistence stress model was validated only on 3 lanes")
        if not 0 <= compliance <= 1:
            raise ValueError("Compliance must be in [0, 1]")
        self.compliance = compliance
        super().__init__(lanes, max_dl)
        if compliance < 1:
            key = round((1 - compliance) * 10000)
            try:
                module = importlib.import_module(f"cav_dl.envs.kernels.stay_{key:04d}")
            except ModuleNotFoundError as exc:
                raise ValueError("No archived kernel for the requested compliance level") from exc
            attributes = {name: getattr(self.kernel, name) for name in dir(self.kernel)
                          if not name.startswith("__")}
            attributes["ulp_clear_advance"] = module.advance_stay
            self.kernel = SimpleNamespace(**attributes)

    def _allocate_road(self):
        if not hasattr(self, "_history"):
            self._history = np.zeros((28802, 120, 37))
            self._last_written = 0
            self._road_views = [
                SimpleNamespace(length=1 / 3, dt=15, lane_num=3, sim_time_step=28802,
                                **{name: self._history[:, i, j]
                                   for j, name in enumerate(k.FIELDS)})
                for i in range(120)
            ]
        else:
            self._history[:self._last_written].fill(0.)
            self._last_written = 0
        return self._road_views

    def observe(self):
        obs = super().observe()
        if self.compliance < 1:
            row = self._history[self.t - 3]
            for segment in range(6):
                cell = row[segment * 20 + 10]
                if cell[k.A] > 0:
                    obs[segment * 5 + 2] = dl_share(cell)
        return obs
