from dataclasses import dataclass,asdict
import hashlib
import inspect
from pathlib import Path
import numpy as np
from numba import njit
import gymnasium as gym
from cav_dl.envs._legacy import kernel as k
from cav_dl.envs._legacy.environment import FastCTMEnv

COMPONENTS=('density','capacity','safety','time','early','timeout')
VARIANTS=('Full','NoDensity','NoCapacity','NoSafety','NoTime','NoEarlyPenalty','NoTimeoutPenalty','RejectWithPenalty','RejectNoPenalty')
DROP={'NoDensity':0,'NoCapacity':1,'NoSafety':2,'NoTime':3,'NoEarlyPenalty':4,'NoTimeoutPenalty':5,'RejectNoPenalty':4}


from .guarded import guarded_advance


class NativeExperimentEnv(FastCTMEnv):
    def __init__(self,reject=False):
        self.reject=reject
        self.rejected_microsteps=0
        super().__init__()

    def reset(self,**kwargs):
        s=kwargs['options']
        self._fresh_m=np.asarray(s['input_supply']).copy()
        self._fresh_r=np.asarray(s['ramp_inflow']).copy()
        self._fresh_mc=self._fresh_m*np.asarray(s['input_cav_supply'])
        self._fresh_rc=self._fresh_r*np.asarray(s['ramp_cav_ratio'])
        self._initial_n=float(s['init_car'].sum())
        self._initial_c=float(s['init_car']@s['init_cav_ratio'])
        return super().reset(**kwargs)

    def _inventory_audit(self):
        t=self.t
        fresh_m=self._fresh_m[t] if t<len(self._fresh_m) else 0.
        fresh_r=self._fresh_r[t] if t<len(self._fresh_r) else 0.
        fresh_mc=self._fresh_mc[t] if t<len(self._fresh_mc) else 0.
        fresh_rc=self._fresh_rc[t] if t<len(self._fresh_rc) else 0.
        queue_n=self.input_supply[t]+self.ramp_inflow[t]-fresh_m-fresh_r
        queue_c=self.input_supply[t]*self.input_cav_supply[t]+self.ramp_inflow[t]*self.ramp_cav_ratio[t]-fresh_mc-fresh_rc
        row=self._history[t]
        n_expected=self._initial_n+self._fresh_m[:t].sum()+self._fresh_r[:t].sum()
        c_expected=self._initial_c+self._fresh_mc[:t].sum()+self._fresh_rc[:t].sum()
        self.latest_n_error=float(n_expected-self.cumulative_main_output-row[:,k.N].sum()-queue_n)
        self.latest_c_error=float(c_expected-self._history[:t,-1,k.OC].sum()-row[:,k.N]@row[:,k.P]-queue_c)

    def _advance(self,count,lanes=None):
        if not self.reject or lanes is None:
            result=super()._advance(count,lanes)
            self._inventory_audit()
            return result
        cumulative=np.array([self.cumulative_main_input,self.cumulative_ramp_input,self.cumulative_main_output],dtype=float)
        t,performed,early,terminated,rejects=guarded_advance(self._history,
            self.input_supply,self.input_cav_supply,self.ramp_inflow,self.ramp_cav_ratio,self.ramp,
            self.actual_input,self.cav_input,self.ramp_input_actual_ts,self.t,count,
            np.asarray(lanes,dtype=float),True,self.ramp_entry_index,cumulative,
            self.actual_initial_car_num,self.actual_main_input+self.actual_ramp_input,self.terminated)
        self.rejected_microsteps+=rejects
        self.cumulative_main_input,self.cumulative_ramp_input,self.cumulative_main_output=cumulative
        self.terminated=terminated
        self.early_stop=False
        self.early_stop_flag=[False]*120
        if t-1>=3920:
            self.stop_penalty=True
        for _ in range(performed):
            self.sys_time+=15/3600
        self.t=t
        self._last_written=max(self._last_written,t+1)
        self._inventory_audit()


class RewardExperiment(gym.Wrapper):
    def __init__(self,variant):
        assert variant in VARIANTS
        self.variant=variant
        super().__init__(NativeExperimentEnv(variant.startswith('Reject')))

    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.episode_p=float(self.env.input_cav_supply_1[0])
        self.fresh_main=np.asarray(kwargs['options']['input_supply']).copy()
        self.fresh_ramp=np.asarray(kwargs['options']['ramp_inflow']).copy()
        self.initial=float(kwargs['options']['init_car'].sum())
        self.component_totals=np.zeros(6)
        cri=self.env
        
        
        self.warmup_reference=float(cri.episode_total_reward)
        self.actual_return=0.
        self.controls=0
        self.rejected_controls=0
        self.executed_switches=0.
        self.last_executed=np.zeros(4)
        self.action_counts=np.zeros(self.env.lane_num,dtype=int)
        return obs,info

    def step(self,action):
        self.env.rejected_microsteps=0
        obs,native,done,trunc,info=self.env.step(action)
        z=info['custom_reward_info']['step_info']
        rejected=self.env.rejected_microsteps>0
        early=bool(self.env.early_stop)
        values=np.array([z['step_density'],z['step_flow'],z['step_safety'],
            z['step_time_early'] if not early else 0.,-35. if early else 0.,z['step_time_protracted']],dtype=float)
        reference=float(native)
        if rejected:
            
            
            reference=reference-values[3]-35.
            values[3]=0.
            values[4]=-35.
        assert abs(values.sum()-reference)<1e-6,(values,reference)
        reward=reference-(values[DROP[self.variant]] if self.variant in DROP else 0.)
        self.component_totals+=values
        self.actual_return+=reward
        self.controls+=1
        self.rejected_controls+=int(rejected)
        request=self.env.decode_discrete_action(int(action))
        self.action_counts+=np.bincount(request,minlength=self.env.lane_num)
        actual=np.array([self.env._history[self.env.t-1,(j+1)*20,k.A] for j in range(4)])
        self.executed_switches+=np.abs(actual-self.last_executed).sum()
        self.last_executed=actual
        info['six_reward_components']=values.copy()
        info['full_reference_reward']=reference
        info['rejected_request']=rejected
        if done or trunc:
            reason='infeasible' if early else ('timeout' if self.env.stop_penalty else (
                'output_65000' if self.env.cumulative_main_output>=65000 else 'near_complete'))
            n=self.env._history[self.env.t,:,k.N].sum()
            arrivals=self.initial+self.fresh_main[:self.env.t].sum()+self.fresh_ramp[:self.env.t].sum()
            outputs=self.env._history[:self.env.t,-1,k.O]
            fresh=np.zeros(self.env.t)
            limit=min(self.env.t,len(self.fresh_main))
            fresh[:limit]=self.fresh_main[:limit]+self.fresh_ramp[:limit]
            ledger=self.initial+np.cumsum(fresh)-np.r_[0.,np.cumsum(outputs[:-1])]
            info['ablation_episode']={'p':self.episode_p,'actual_return':self.actual_return,
                'full_reference_return':self.component_totals.sum(),
                'reference_with_warmup':self.component_totals.sum()+self.warmup_reference,
                **dict(zip(COMPONENTS,self.component_totals.tolist())),
                'controls':self.controls,'end_step':self.env.t,'reason':reason,
                'output':float(self.env.cumulative_main_output),
                'road_residual':float(n),'queue_residual':float(arrivals-self.env.cumulative_main_output-n),
                'future_unreleased':float(self.fresh_main[self.env.t:].sum()+self.fresh_ramp[self.env.t:].sum()),
                'partial_system_vehicle_hours':float(ledger.sum()*15/3600),
                'mass_error':self.env.latest_n_error,'cav_mass_error':self.env.latest_c_error,
                'rejected_controls':self.rejected_controls,'executed_switches':self.executed_switches,
                'request_0':int(self.action_counts[0]),'request_1':int(self.action_counts[1]),'request_2':int(self.action_counts[2])}
        return obs,float(reward),done,trunc,info
