import numpy as np
from types import SimpleNamespace
from ._legacy import kernel as k
from .native import LaneNative

class ClearanceNative(LaneNative):
    def __init__(self, lanes=3, max_dl=None):
        super().__init__(lanes, max_dl)
        self.sim_time_step = 28801
        self.actual_input = np.zeros(28801)
        self.cav_input = np.zeros(28801)
        self.ramp_input_actual_ts = np.zeros(28801)
        self.decisions = 0

    def reset(self, **kwargs):
        self.actual_input.fill(0.)
        self.cav_input.fill(0.)
        self.ramp_input_actual_ts.fill(0.)
        return super().reset(**kwargs)

    def _allocate_road(self):
        if not hasattr(self, '_history'):
            self._history = np.zeros((28802, 120, len(k.FIELDS)))
            self._last_written = 0
            self._road_views = [SimpleNamespace(length=1/3, dt=15, lane_num=self.total_lanes, sim_time_step=28802,
                **{name: self._history[:, i, j] for j, name in enumerate(k.FIELDS)}) for i in range(120)]
        else:
            self._history[:self._last_written].fill(0.)
            self._last_written = 0
        return self._road_views

    def _advance(self, count, lanes=None):
        assert self.t+count < min(len(self.actual_input), len(self.cav_input), len(self.ramp_input_actual_ts), len(self.input_supply), len(self.ramp_inflow))
        cumulative = np.array([self.cumulative_main_input, self.cumulative_ramp_input, self.cumulative_main_output], dtype=float)
        lane_config = np.zeros(120) if lanes is None else np.asarray(lanes, dtype=float)
        self.t, performed, _, _, rejects = self.kernel.ulp_clear_advance(self._history, self.input_supply, self.input_cav_supply,
            self.ramp_inflow, self.ramp_cav_ratio, self.ramp, self.actual_input, self.cav_input,
            self.ramp_input_actual_ts, self.t, count, lane_config, lanes is not None,
            self.ramp_entry_index, cumulative, self.actual_initial_car_num,
            self.actual_main_input+self.actual_ramp_input, False)
        self.cumulative_main_input, self.cumulative_ramp_input, self.cumulative_main_output = cumulative
        self.rejected_microsteps += rejects
        self.terminated = self.early_stop = self.stop_penalty = False
        self._last_written = max(self._last_written, self.t+1)
        for _ in range(performed):
            self.sys_time += 15/3600
        self._inventory_audit()

    def observe(self):
        current = np.array([self._history[self.t-1, (s+1)*20, k.A] for s in range(4)])
        return self.kernel.observation(self._history, self.t, current, self.actual_input, 60)

    def residual(self):
        fresh_m = self._fresh_m[self.t] if self.t < len(self._fresh_m) else 0.
        fresh_r = self._fresh_r[self.t] if self.t < len(self._fresh_r) else 0.
        return float(self._history[self.t, :, k.N].sum() + self.input_supply[self.t]-fresh_m + self.ramp_inflow[self.t]-fresh_r)