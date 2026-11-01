
import ast
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np
from . import kernel

SOURCE=Path(__file__).parent/'original'
from .original.environment import traffic_simulator as original_module
from .original.environment.flow_models.input_flow import EpisodeInputGenerator

OriginalCTMEnv=original_module.CTMEnv


class StructuralEdits(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id=="CTMEnv":
            node.id="CompiledLegacyBase"
        return node

    def visit_ClassDef(self,node):
        if node.name=="CTMEnv":
            node.name="CompiledLegacyBase"
        return self.generic_visit(node)

    def visit_Assign(self,node):
        if len(node.targets)==1 and ast.unparse(node.targets[0])=="self.road" and isinstance(node.value,ast.ListComp):
            node.value=ast.parse("self._allocate_road()",mode="eval").body
        return self.generic_visit(node)

    def visit_For(self,node):
        if ast.unparse(node.iter)=="range(self.step_size)":
            return ast.copy_location(ast.parse("self._advance_control(cav_lane_config)").body[0],node)
        return self.generic_visit(node)


tree=StructuralEdits().visit(ast.parse((SOURCE/"environment/traffic_simulator.py").read_text(encoding="utf-8-sig")))
ast.fix_missing_locations(tree)
namespace=original_module.__dict__.copy()
namespace["print"]=lambda *args,**kwargs:None
exec(compile(tree,str(SOURCE/"environment/traffic_simulator.py"),"exec"),namespace)


class FastCTMEnv(namespace["CompiledLegacyBase"]):
    def __init__(self):
        super().__init__(cell_num=120,segment_num=6,sim_time_step=5000,lane_num=3,dt=15)

    def _allocate_road(self):
        if not hasattr(self,"_history"):
            self._history=np.zeros((5001,120,len(kernel.FIELDS)),dtype=np.float64)
            self._last_written=0
            self._road_views=[SimpleNamespace(length=1/3,dt=15,lane_num=3,sim_time_step=5001,
                **{name:self._history[:,i,j] for j,name in enumerate(kernel.FIELDS)}) for i in range(120)]
        else:
            self._history[:self._last_written].fill(0.)
            self._last_written=0
        return self._road_views

    def _advance(self,count,lanes=None):
        cumulative=np.array([self.cumulative_main_input,self.cumulative_ramp_input,self.cumulative_main_output],dtype=np.float64)
        row_lanes=np.zeros(120) if lanes is None else np.asarray(lanes,dtype=np.float64)
        t,performed,early,terminated=kernel.advance(self._history,self.input_supply,self.input_cav_supply,
            self.ramp_inflow,self.ramp_cav_ratio,self.ramp,self.actual_input,self.cav_input,
            self.ramp_input_actual_ts,self.t,count,row_lanes,lanes is not None,self.ramp_entry_index,
            cumulative,self.actual_initial_car_num,self.actual_main_input+self.actual_ramp_input,self.terminated)
        self.cumulative_main_input,self.cumulative_ramp_input,self.cumulative_main_output=cumulative
        self.terminated=terminated
        
        self.early_stop_flag=[False]*120
        self.early_stop_flag[20:99]=list(self._history[t-1,20:99,kernel.EARLY]>0)
        self.early_stop=early if lanes is not None else self.early_stop
        if t-1>=3920:
            self.stop_penalty=True
        for _ in range(performed):
            self.sys_time+=15/3600
        self.t=t
        self._last_written=max(self._last_written,t+1)

    def _run_one_step(self,t,cav_lane=None):
        self._advance(1,cav_lane)

    def _advance_control(self,cav_lane):
        self._advance(self.step_size,cav_lane)

    def get_obs(self):
        return kernel.observation(self._history,self.t,self.action_recording[self.current_step],
            self.actual_input,self.ramp_entry_index)

    def cal_reward(self,state):
        reward,rd,rf,penalty=kernel.reward_components(self._history,self.t,state)
        
        penalty=np.float32(penalty) if .5<max(state[4],state[9],state[14],state[19],state[24],state[29])<=1 else 0.
        return float(reward),{"reward_density":rd,"reward_flow":rf,"safety_penalty":penalty,"base_reward":reward}


def original_options(episode_id):
    
    
    return EpisodeInputGenerator().generate_test_reset_config(episode_id,10600,6500,[.4,.5,.6,.7,.8,.9])
