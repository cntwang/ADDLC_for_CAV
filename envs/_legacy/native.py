import numpy as np
import gymnasium as gym
from cav_dl.envs._legacy import kernel as k
from .warmup import BatchedWarmupNative
from .native3 import advance_ulp as ADVANCE

class UlpBatchedWarmupNative(BatchedWarmupNative):
    def __init__(self):
        super().__init__()

    def reset(self, **kwargs):
        result=super().reset(**kwargs)
        self.cav_mass_error_max=0.
        return result

    def _advance(self,count,lanes=None):
        cumulative=np.array([self.cumulative_main_input,self.cumulative_ramp_input,
                             self.cumulative_main_output],dtype=np.float64)
        row_lanes=np.zeros(120) if lanes is None else np.asarray(lanes,dtype=np.float64)
        t,performed,early,terminated=ADVANCE(
            self._history,self.input_supply,self.input_cav_supply,self.ramp_inflow,self.ramp_cav_ratio,
            self.ramp,self.actual_input,self.cav_input,self.ramp_input_actual_ts,self.t,count,row_lanes,
            lanes is not None,self.ramp_entry_index,cumulative,self.actual_initial_car_num,
            self.actual_main_input+self.actual_ramp_input,self.terminated)
        self.cumulative_main_input,self.cumulative_ramp_input,self.cumulative_main_output=cumulative
        self.terminated=terminated
        self.early_stop_flag=[False]*120
        self.early_stop_flag[20:99]=list(self._history[t-1,20:99,k.EARLY]>0)
        self.early_stop=early if lanes is not None else self.early_stop
        if t-1>=3920:
            self.stop_penalty=True
        for _ in range(performed):
            self.sys_time+=15/3600
        self.t=t
        self._last_written=max(self._last_written,t+1)
        self._inventory_audit()
        self.cav_mass_error_max=max(getattr(self,'cav_mass_error_max',0.),abs(self.latest_c_error))
        if getattr(self,'strict_cav_conservation',False) and abs(self.latest_c_error)>1e-5:
            raise ValueError(f'CTM CAV ledger error {self.latest_c_error:.6g} vehicles exceeds 1e-5')