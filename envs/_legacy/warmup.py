
import gymnasium as gym
from .core import NativeExperimentEnv,RewardExperiment,VARIANTS

class BatchedWarmupNative(NativeExperimentEnv):
    def reset(self,**kwargs):
        self._in_batched_reset=True
        try:return super().reset(**kwargs)
        finally:self._in_batched_reset=False

    def _run_one_step(self,t,cav_lane=None):
        if getattr(self,'_in_batched_reset',False) and cav_lane is None:
            
            
            
            assert 0<=t<120
            if t%60==0:
                assert self.t==t
                self._advance(60)
            return
        return super()._run_one_step(t,cav_lane)

class BatchedRewardExperiment(RewardExperiment):
    def __init__(self,variant):
        assert variant in VARIANTS
        self.variant=variant
        gym.Wrapper.__init__(self,BatchedWarmupNative(variant.startswith('Reject')))
