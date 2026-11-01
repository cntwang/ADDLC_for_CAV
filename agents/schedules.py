from stable_baselines3.common.callbacks import BaseCallback

class EntCoefSchedulerBySteps(BaseCallback):
    
    def __init__(self, start: float, end: float,
                 schedule_steps: int,
                 warmup_steps: int = 0,
                 log_key: str = "train/ent_coef",
                 force_dump: bool = False,
                 verbose: int = 0):
        super().__init__(verbose)
        assert schedule_steps >= 1
        self.start = float(start)
        self.end = float(end)
        self.schedule_steps = int(schedule_steps)
        self.warmup_steps = int(warmup_steps)
        self.log_key = log_key
        self.force_dump = force_dump
        self._val = start

    def _compute(self, t: int) -> float:
        
        if t <= self.warmup_steps:
            return self.start
        t2 = t - self.warmup_steps
        if t2 >= self.schedule_steps:
            return self.end
        ratio = t2 / self.schedule_steps  
        return self.start + (self.end - self.start) * ratio

    def _on_step(self) -> bool:
        t = int(self.model.num_timesteps)  
        self._val = self._compute(t)
        self.model.ent_coef = float(self._val)
        if self.logger is not None:
            self.logger.record(self.log_key, float(self._val))
            if self.force_dump:
                self.logger.dump(t)
        return True