import itertools
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from cav_dl.envs._legacy.original.reward_plot import poisson_reward_with_cav
from .flow_models.cell_models import Cell
from .flow_models.fd_parameters import update_fundamental_parameters
from .flow_models.input_flow import compute_actual_input_step


class EpisodeEarlyStopper:
    def __init__(self, patience, delta):
        
        self.patience = patience
        self.delta = delta
        self.reset()

    def reset(self):
        self.best_reward = -float('inf')
        self.counter = 0
        self.triggered = False

    def step(self, reward):
        if reward > self.best_reward + self.delta:
            self.best_reward = reward
            self.counter = 0
        else:
            self.counter += 1

        self.triggered = self.counter >= self.patience
        return self.triggered


class CTMEnv(gym.Env):
    

    def __init__(
            self,
            cell_num=120,
            segment_num=6,
            sim_time_step=3000,
            lane_num=3,
            dt=15,
    ):
        super(CTMEnv, self).__init__()

        
        self.input_manager = None
        self.cell_num = cell_num
        self.segment_num = segment_num
        self.segment_cell_num = self.cell_num // self.segment_num
        self.sim_time_step = sim_time_step
        self.lane_num = lane_num
        self.dt = dt  
        self.t = 0
        self.step_size = int(0.25 * 3600 / self.dt)  

        self.sys_time = 0
        self.current_episode = 0
        self.input_supply = np.zeros(sim_time_step)
        self.input_cav_supply = np.zeros(sim_time_step)
        self.ramp_inflow = np.zeros(sim_time_step)
        self.ramp_cav_ratio = np.zeros(sim_time_step)
        self.actual_input = np.zeros(sim_time_step)
        self.cav_input = np.zeros(sim_time_step)
        self.ramp = np.zeros((self.sim_time_step, 2))  
        self.init_cav_ratio = np.zeros(sim_time_step)
        self.init_car = np.zeros(sim_time_step)

        
        self.road = [Cell(self.sim_time_step + 1) for _ in range(self.cell_num)]

        
        self.action_space = spaces.Discrete(81)

        
        self.segment_feature_num = 5
        self.obs_dim = self.segment_num * self.segment_feature_num + 8
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.early_stopper = EpisodeEarlyStopper(patience=10, delta=0.001)

        self.early_stop_flag = [False] * self.cell_num
        self.control_start = 20
        self.control_end = 100 - 1
        self.control_segment_num = 4
        self.early_stop = False

        self.max_steps = 500
        self.state = np.zeros((self.max_steps, self.obs_dim))  
        self.current_step = 0
        self.episode_total_reward = 0.0
        self.episode_total_density_reward = 0.0  
        self.episode_total_flow_reward = 0.0  
        self.episode_total_safety_penalty = 0.0  
        self.episode_total_time_early = 0.0  
        self.episode_total_time_protracted = 0.0  
        self.cumulative_main_input = 0
        self.cumulative_ramp_input = 0
        self.cumulative_main_output = 0
        self.action_recording = np.zeros((self.max_steps, 4), dtype=np.float32)
        self.terminated = False
        self.input_sim_step = int(10.5 * 3600 / self.dt)
        self.input_supply_1 = np.zeros(self.input_sim_step)
        self.input_cav_supply_1 = np.zeros(self.input_sim_step)
        self.ramp_inflow_1 = np.zeros(self.input_sim_step)
        self.ramp_cav_ratio_1 = np.zeros(self.input_sim_step)
        self.ramp_entry_index = 0
        self.actual_initial_car_num = 0.0
        self.actual_main_input = 0.0
        self.actual_ramp_input = 0.0
        self.stop_penalty = False
        self.reward_accord = np.zeros(int(50 / 0.25) + 1)
        self.action_table = list(itertools.product([0, 1, 2], repeat=4))  
        
        
        self.target_car_num = 65000
        
        
        self.ramp_input_actual_ts = np.zeros(sim_time_step)

    def reset(self, *, options=None, test=False):  
        self.actual_initial_car_num = 0.0
        self.actual_main_input = 0.0
        self.actual_ramp_input = 0.0
        self.episode_total_reward = 0.0
        self.episode_total_density_reward = 0.0  
        self.episode_total_flow_reward = 0.0  
        self.episode_total_safety_penalty = 0.0  
        self.episode_total_time_early = 0.0  
        self.episode_total_time_protracted = 0.0  
        self.cumulative_main_input = 0
        self.cumulative_ramp_input = 0
        self.cumulative_main_output = 0
        self.current_step = 0
        self.action_recording = np.zeros((self.max_steps, 4), dtype=np.float32)
        
        self.terminated = False
        self.sys_time = 0
        self.t = 0
        self.early_stop = False
        self.stop_penalty = False
        self.early_stop_flag = [False] * self.cell_num
        self.early_stopper.reset()
        
        self.ramp_input_actual_ts[:] = 0.0
        
        
        

        if options is not None:
            self.init_car = np.array(options["init_car"])
            self.init_cav_ratio = np.array(options["init_cav_ratio"])
            self.input_supply_1 = np.array(options["input_supply"])
            self.input_cav_supply_1 = np.array(options["input_cav_supply"])
            self.ramp_inflow_1 = np.array(options["ramp_inflow"])
            self.ramp_cav_ratio_1 = np.array(options["ramp_cav_ratio"])
            self.ramp_entry_index = int(options["ramp_entry_index"])
            if test:
                if "real_main_flow" in options and options["real_main_flow"] is not None:
                    self.input_supply_1 = np.asarray(options["real_main_flow"], dtype=float)
                if "real_ramp_flow" in options and options["real_ramp_flow"] is not None:
                    self.ramp_inflow_1 = np.asarray(options["real_ramp_flow"], dtype=float)
                self.input_sim_step = len(self.ramp_inflow_1)

        self.actual_initial_car_num = np.sum(self.init_car)
        

        self.road = [Cell(self.sim_time_step + 1) for _ in range(self.cell_num)]
        for i in range(self.cell_num):
            self.road[i].car_num[0] = self.init_car[i]
            self.road[i].cav_ratio[0] = self.init_cav_ratio[i]
            self.road[i].cav_lane[0] = 0
        
        

        
        if self.sim_time_step > self.input_sim_step:
            pad_zeros_1 = np.zeros(self.sim_time_step - len(self.input_cav_supply_1))
            self.input_cav_supply = np.concatenate([self.input_cav_supply_1, pad_zeros_1])
            
            pad_zeros_2 = np.zeros(self.sim_time_step - len(self.input_supply_1))
            self.input_supply = np.concatenate([self.input_supply_1, pad_zeros_2])
            self.actual_main_input = np.sum(self.input_supply)  
            
            pad_zeros_3 = np.zeros(self.sim_time_step - len(self.ramp_cav_ratio_1))
            self.ramp_cav_ratio = np.concatenate([self.ramp_cav_ratio_1, pad_zeros_3])
            
            pad_zeros_4 = np.zeros(self.sim_time_step - len(self.ramp_inflow_1))
            self.ramp_inflow = np.concatenate([self.ramp_inflow_1, pad_zeros_4])
            self.actual_ramp_input = np.sum(self.ramp_inflow)  
            self.ramp = np.column_stack((self.ramp_inflow, self.ramp_cav_ratio))
        else:
            self.input_cav_supply = self.input_cav_supply_1[:self.sim_time_step]
            self.input_supply = self.input_supply_1[:self.sim_time_step]
            self.actual_main_input = np.sum(self.input_supply)  
            self.ramp_cav_ratio = self.ramp_cav_ratio_1[:self.sim_time_step]
            self.ramp_inflow = self.ramp_inflow_1[:self.sim_time_step]
            self.actual_ramp_input = np.sum(self.ramp_inflow)  
            self.ramp = np.column_stack((self.ramp_inflow, self.ramp_cav_ratio))

        for i in range(int(0.5 * 3600 / self.dt)):
            self._run_one_step(i)
            if i == int(0.25 * 3600 / self.dt) - 1:
                obs_1 = self.get_obs()
                print(f"时步0 动作序号0 动作(0, 0, 0, 0)")
                
                self.state[0] = obs_1
                reward_1, com_1 = self.cal_reward(obs_1)
                self.reward_accord[0] = reward_1
                print(f"奖励 {reward_1:.2f}")
                self.episode_total_reward += reward_1
                self.episode_total_density_reward += com_1["reward_density"]  
                self.episode_total_flow_reward += com_1["reward_flow"]  
                self.episode_total_safety_penalty += com_1["safety_penalty"]  
        obs = self.get_obs()
        print(f"时步60 动作序号0 动作(0, 0, 0, 0)")
        
        self.state[1] = obs
        reward_2, com_2 = self.cal_reward(obs)
        print(f"奖励 {reward_2:.2f}")
        self.reward_accord[1] = reward_2
        
        self.episode_total_reward += reward_2
        self.episode_total_density_reward += com_2["reward_density"]  
        self.episode_total_flow_reward += com_2["reward_flow"]  
        self.episode_total_safety_penalty += com_2["safety_penalty"]  
        self.current_episode += 1
        self.current_step = 1
        self.action_recording[0] = [0, 0, 0, 0]
        self.action_recording[1] = [0, 0, 0, 0]
        print(
            f"第{self.current_episode}轮，总计期望输入{self.actual_main_input + self.actual_ramp_input}，主路期望输入{self.actual_main_input}，匝道期望输入{self.actual_ramp_input}")

        return obs, {}

    def step(self, action):
        
        cav_lane_config = self.trans_action(action)
        self.early_stop = False
        for i in range(self.step_size):
            self.early_stop_flag = [False] * self.cell_num
            t = self.t
            if i == 0:  
                print(f"时步{t} 动作{action} {self.decode_discrete_action(action)}")
            self._run_one_step(t, cav_lane_config)
            if any(self.early_stop_flag):
                self.early_stop = True
                break

        obs = self.get_obs()
        
        self.current_step += 1
        self.state[self.current_step] = obs
        self.action_recording[self.current_step] = self.decode_discrete_action(action)
        
        
        reward, comp = self.cal_reward(obs)
        
        self.episode_total_density_reward += comp["reward_density"]  
        self.episode_total_flow_reward += comp["reward_flow"]  
        self.episode_total_safety_penalty += comp["safety_penalty"]  

        
        
        
        
        

        time_penalty_early_stop = 0.0  
        time_penalty_protracted = 0.0  
        if self.early_stop:
            
            
            time_penalty_early_stop = -35.0
            reward += time_penalty_early_stop  
            terminated = True
            self.episode_total_reward += reward
        else:
            
            progress = self.t / self.sim_time_step
            
            kappa0, beta = 0.05, 13
            
            
            time_penalty_early_stop = -kappa0 * (1.0 + beta * progress)
            reward += time_penalty_early_stop  
            self.episode_total_reward += reward
            terminated = self.terminated
            if self.terminated:
                
                
                
                
                
                
                
                
                
                if self.stop_penalty:
                    time_penalty_protracted = -35.0
                    reward += time_penalty_protracted
                    self.episode_total_reward += time_penalty_protracted
                else:
                    
                    self.episode_total_reward += 0
                    print(f"正常结束")
        print(f"折减后奖励差{reward}")
        if abs(reward) < 1e-6:
            step_ratio_density = step_ratio_flow = step_ratio_safety = step_ratio_time_early = step_ratio_time_protracted = 0.0
        else:
            step_ratio_density = comp["reward_density"] / reward
            step_ratio_flow = comp["reward_flow"] / reward
            step_ratio_safety = comp["safety_penalty"] / reward
            step_ratio_time_early = time_penalty_early_stop / reward
            step_ratio_time_protracted = time_penalty_protracted / reward
        self.episode_total_time_early += time_penalty_early_stop  
        self.episode_total_time_protracted += time_penalty_protracted  

        ep_info = {}
        truncated = False
        if terminated:
            self.init_car = np.zeros(self.input_sim_step)
            self.init_cav_ratio = np.zeros(self.input_sim_step)
            self.input_supply_1 = np.zeros(self.input_sim_step)
            self.input_cav_supply_1 = np.zeros(self.input_sim_step)
            self.ramp_inflow_1 = np.zeros(self.input_sim_step)
            self.ramp_cav_ratio_1 = np.zeros(self.input_sim_step)
            self.ramp_entry_index = 0
            self.input_cav_supply = np.zeros(self.sim_time_step)
            self.input_supply = np.zeros(self.sim_time_step)
            self.ramp_cav_ratio = np.zeros(self.sim_time_step)
            self.ramp_inflow = np.zeros(self.sim_time_step)
            self.ramp = np.zeros((self.sim_time_step, 2))
            
            if abs(self.episode_total_reward) < 1e-6:
                ep_ratio_density = ep_ratio_flow = ep_ratio_safety = ep_ratio_time_early = ep_ratio_time_protracted = 0.0
            else:
                ep_ratio_density = 0.5 * self.episode_total_density_reward / self.episode_total_reward
                ep_ratio_flow = 0.5 * self.episode_total_flow_reward / self.episode_total_reward
                ep_ratio_safety = self.episode_total_safety_penalty / self.episode_total_reward
                ep_ratio_time_early = self.episode_total_time_early / self.episode_total_reward
                ep_ratio_time_protracted = self.episode_total_time_protracted / self.episode_total_reward
            
            ep_info = {
                "ep_density": self.episode_total_density_reward,
                "ep_flow": self.episode_total_flow_reward,
                "ep_safety": self.episode_total_safety_penalty,
                "ep_time_early": self.episode_total_time_early,
                "ep_time_protracted": self.episode_total_time_protracted,
                "ep_total_reward": self.episode_total_reward,
                "ep_ratio_density": ep_ratio_density,
                "ep_ratio_flow": ep_ratio_flow,
                "ep_ratio_safety": ep_ratio_safety,
                "ep_ratio_time_early": ep_ratio_time_early,
                "ep_ratio_time_protracted": ep_ratio_time_protracted,
            }
            print(
                f"第{self.current_episode}轮，实际累计输出{self.cumulative_main_output}辆，总计输入{self.cumulative_main_input + self.cumulative_ramp_input}辆")
            print(f"匝道总计输入{self.cumulative_ramp_input}辆，主路总计输入{self.cumulative_main_input}辆")
            print(f"累计奖励值:{self.episode_total_reward}")
        
        step_info = {
            "step_reward": reward,
            "step_density": comp["reward_density"],
            "step_flow": comp["reward_flow"],
            "step_safety": comp["safety_penalty"],
            "step_time_early": time_penalty_early_stop,
            "step_time_protracted": time_penalty_protracted,
            "step_ratio_density": step_ratio_density,
            "step_ratio_flow": step_ratio_flow,
            "step_ratio_safety": step_ratio_safety,
            "step_ratio_time_early": step_ratio_time_early,
            "step_ratio_time_protracted": step_ratio_time_protracted,
        }
        return obs, reward, terminated, truncated, {
            "custom_reward_info": {
                "step_reward": reward,
                "episode_reward": self.episode_total_reward,
                "step_info": step_info,
                "ep_info": ep_info,
            }
        }

    def _run_one_step(self, t, cav_lane=None):

        if cav_lane is not None:
            for i in range(self.cell_num):
                self.road[i].cav_lane[t] = cav_lane[i]

        for i in range(self.cell_num):
            self.road[i].update_timestep(t)

        for i in range(self.cell_num - 1):
            if i == self.ramp_entry_index - 1:  
                self.road[i].cal_outflow(t, down_cell=self.road[i + 1], ramp=self.ramp[t, :])
                if self.ramp_inflow[t] > self.road[i].ramp_output[t] and t < self.sim_time_step:  
                    car_remain = self.ramp_inflow[t] - self.road[i].ramp_output[t]
                    cav_new = self.ramp_inflow[t] * self.ramp_cav_ratio[t] - self.road[i].ramp_cav_output[t] + \
                              self.ramp_inflow[t + 1] * self.ramp_cav_ratio[t + 1]
                    self.ramp_inflow[t + 1] += car_remain
                    self.ramp_cav_ratio[t + 1] = cav_new / self.ramp_inflow[t + 1] if self.ramp_inflow[t + 1] > 0 else 0
                    self.ramp[t + 1, :] = np.column_stack((self.ramp_inflow[t + 1], self.ramp_cav_ratio[t + 1]))
                self.cumulative_ramp_input += self.road[i].ramp_output[t]
            else:
                self.road[i].cal_outflow(t, down_cell=self.road[i + 1])
        self.road[-1].cal_outflow(t)
        
        if self.ramp_entry_index > 0:
            self.ramp_input_actual_ts[t] = self.road[self.ramp_entry_index - 1].ramp_output[t]
        else:
            self.ramp_input_actual_ts[t] = 0.0
        self.cumulative_main_output += self.road[-1].outflow[t]

        self.actual_input[t], self.cav_input[t] = compute_actual_input_step(t, self.input_supply[t],
                                                                            self.input_cav_supply[t], self.road[0])
        self.cumulative_main_input += self.actual_input[t]
        if self.input_supply[t] > self.actual_input[t] and t < self.sim_time_step:
            car_remain = self.input_supply[t] - self.actual_input[t]
            cav_new = (self.input_supply[t] - self.actual_input[t]) * self.input_cav_supply[t] + self.input_supply[
                t + 1] * self.input_cav_supply[t + 1]
            if np.isnan(cav_new):
                print("❌ cav_new is NaN!")
                print(f"→ t = {t}, i = {i}")
                print(f"→ self.ramp_inflow[t] = {self.ramp_inflow[t]}")
                print(f"→ self.ramp_cav_ratio[t] = {self.ramp_cav_ratio[t]}")
                print(f"→ self.road[i].ramp_cav_output[t] = {self.road[i].ramp_cav_output[t]}")
                print(f"→ self.ramp_inflow[t+1] = {self.ramp_inflow[t + 1]}")
                print(f"→ self.ramp_cav_ratio[t+1] = {self.ramp_cav_ratio[t + 1]}")
                raise ValueError("cav_new evaluated to NaN, terminating for debugging.")
            self.input_supply[t + 1] += car_remain
            self.input_cav_supply[t + 1] = cav_new / self.input_supply[t + 1] if self.input_supply[t + 1] > 0 else 0

        self.road[0].update_state(t, up_input=self.actual_input[t], up_input_cav=self.cav_input[t])
        for i in range(1, self.cell_num):
            if i == self.ramp_entry_index:
                self.road[i].update_state(t,
                                          up_input=self.road[i - 1].outflow[t] + self.road[i - 1].ramp_output[t],
                                          up_input_cav=self.road[i - 1].outflow_cav[t] +
                                                       self.road[i - 1].ramp_cav_output[t])
            else:
                self.road[i].update_state(t,
                                          up_input=self.road[i - 1].outflow[t],
                                          up_input_cav=self.road[i - 1].outflow_cav[t]
                                          )

        for i in range(self.control_start, self.control_end):
            if self.road[i].capacity_early_stop[t]:
                print(
                    f"⚠️ 第 {self.current_episode} 轮：第 {i} 个元胞在 t={t} 时 capacity_early_stop=True, 车辆数为 {self.road[i].car_num[t]}, 渗透率为 {self.road[i].cav_ratio[t]}, 动作为 {self.road[i].cav_lane[t]}")
                
                self.early_stop_flag[i] = True

        
        
        
        
        
        
        
        
        

        
        if self.cumulative_main_output >= self.target_car_num or np.abs((
                                                                                self.actual_initial_car_num + self.actual_main_input + self.actual_ramp_input) - self.cumulative_main_output) < 10 or self.t >= 3920:  
            self.terminated = True
            if self.t >= 3920:
                self.stop_penalty = True
                print(f"没满足车辆数停止条件")

        
        
        
        
        
        

        self.sys_time += self.dt / 3600
        self.t += 1

    def get_obs(self):
        t = self.t - 3  
        state = np.zeros(self.segment_num * self.segment_feature_num)  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        dl_jam_density, _, _, _, _ = update_fundamental_parameters(1.0)

        for i in range(self.segment_num):
            segment_security = []
            seg_detector = i * self.segment_cell_num + 10  
            if self.road[seg_detector].cav_lane[t] > 0:
                seg_dl_density = self.road[seg_detector].dl_density[t]
                seg_gl_density = self.road[seg_detector].gl_density[t]
                seg_dl_cav_ratio = 1.0
                seg_gl_cav_ratio = self.road[seg_detector].gl_cav_ratio[t]
            else:
                seg_dl_density = 0
                seg_gl_density = self.road[seg_detector].density[t]
                seg_dl_cav_ratio = 0
                seg_gl_cav_ratio = self.road[seg_detector].cav_ratio[t]
            for j in range(i * self.segment_cell_num, (i + 1) * self.segment_cell_num):
                
                if self.road[j].cav_lane[t] > 0:
                    gl_jam_density, _, _, _, _ = update_fundamental_parameters(self.road[j].gl_cav_ratio[t])
                    
                    
                    gl_car_num_max = gl_jam_density * self.road[j].length * (self.lane_num - self.road[j].cav_lane[t])
                    
                    security = self.road[j].gl_car_num[t] / gl_car_num_max if gl_car_num_max > 0 else 0
                else:
                    jam_density, _, _, _, _ = update_fundamental_parameters(self.road[j].cav_ratio[t])
                    car_num_max = jam_density * self.lane_num * self.road[j].length
                    security = self.road[j].car_num[t] / car_num_max if car_num_max > 0 else 0
                segment_security.append(security)
            seg_max_security_index = max(segment_security)

            

            
            offset = i * self.segment_feature_num
            state[offset] = seg_dl_density  
            state[offset + 1] = seg_gl_density  
            state[offset + 2] = seg_dl_cav_ratio  
            state[offset + 3] = seg_gl_cav_ratio  
            
            state[offset + 4] = seg_max_security_index  

        ave_main_input = np.mean(
            [self.actual_input[k] for k in range(t - int(0.25 * 3600 / self.dt) + 1, t + 1)])  
        
        
        ave_ramp_input = np.mean(
            [self.road[self.ramp_entry_index - 1].ramp_output[k] for k in
             range(t - int(0.25 * 3600 / self.dt) + 1, t + 1)])  
        
        

        
        
        ave_main_output = np.mean(
            [self.road[-1].outflow[k] for k in
             range(t - int(0.25 * 3600 / self.dt) + 1, t + 1)])  
        
        time_progress = self.t / self.sim_time_step

        state = np.concatenate([
            state,  
            self.action_recording[self.current_step],  
            np.array([
                ave_main_input,
                
                ave_ramp_input,
                
                ave_main_output,
                
                
                time_progress,
                
            ])
        ])

        
        
        
        return np.round(state, 2).astype(np.float32)

    def cal_reward(self, state):

        t = self.t
        cell_reward_step = []
        
        cell_reward_flow_step = []
        _, dl_capacity, _, _, _ = update_fundamental_parameters(1.0)
        cell_capacity_max = dl_capacity * self.lane_num
        
        for j in range(t - int(0.25 * 3600 / self.dt), t):
            cell_reward_t = []
            cell_reward_flow_t = []
            for i in range(self.cell_num):  
                if self.road[i].cav_lane[j] > 0:
                    dl_reward = poisson_reward_with_cav(self.road[i].dl_density[j], 1.0)
                    gl_reward = poisson_reward_with_cav(self.road[i].gl_density[j], self.road[i].gl_cav_ratio[j])
                    cell_reward = (dl_reward * self.road[i].cav_lane[j] + gl_reward * (
                            self.lane_num - self.road[i].cav_lane[j])) / self.lane_num
                    _, gl_capacity, _, _, _ = update_fundamental_parameters(self.road[i].gl_cav_ratio[j])
                    cell_capacity = dl_capacity * self.road[i].cav_lane[j] + gl_capacity * (
                                self.lane_num - self.road[i].cav_lane[j])
                else:
                    cell_reward = poisson_reward_with_cav(self.road[i].density[j], self.road[i].cav_ratio[j])
                    _, gl_capacity, _, _, _ = update_fundamental_parameters(self.road[i].cav_ratio[j])
                    cell_capacity = gl_capacity * self.lane_num
                cell_reward_t.append(cell_reward)
                cell_reward_flow_t.append(cell_capacity / cell_capacity_max)
            cell_reward_step.append(np.mean(cell_reward_t))
            cell_reward_flow_step.append(np.mean(cell_reward_flow_t))
            
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        secure = [state[4 + i * self.segment_feature_num] for i in range(self.segment_num)]
        reward_density = np.mean(cell_reward_step)
        reward_flow = np.mean(cell_reward_flow_step)
        reward_secure = max(secure)
        lambda_safety = 1.0
        
        if 0.5 < reward_secure <= 1.0:
            
            safety_penalty = - lambda_safety * (reward_secure - 0.5)
            
            
            
        else:
            safety_penalty = 0.0
        
        
        
        
        
        
        
        
        
        
        reward = 0.5 * reward_density + 0.5 * reward_flow + safety_penalty
        
        print(f'渗透率 {state[3]}, 加权后总奖励 {reward}, 密度奖励 {reward_density}, 通行能力奖励 {reward_flow}, 安全惩罚 {safety_penalty}')
        
        components = {
            "reward_density": 0.5 * reward_density,
            "reward_flow": 0.5 * reward_flow,
            "safety_penalty": safety_penalty,
            "base_reward": reward,  
        }

        return float(reward), components
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    

    def trans_action(self, rl_action_idx, control_segments=5):
        cav_lane = np.zeros(self.cell_num)
        control_start = self.segment_num - control_segments
        control_end = self.segment_num - 1

        decoded_action = self.decode_discrete_action(rl_action_idx)
        for i in range(self.segment_num):
            seg_start = i * self.segment_cell_num
            seg_end = seg_start + self.segment_cell_num
            if control_start <= i < control_end:
                cav_lane[seg_start:seg_end] = decoded_action[i - control_start]
            else:
                cav_lane[seg_start:seg_end] = 0

        return cav_lane

    def decode_discrete_action(self, idx):
        
        if isinstance(idx, (tuple, list, np.ndarray)):
            idx = idx[0]  
        idx = int(idx)  
        return self.action_table[idx]

    def encode_discrete_action(self, action_vec):
        
        return self.action_table.index(tuple(action_vec))

    def calculate_density(self, t_start):
        
        density = []
        for i in range(self.segment_num):
            seg_start = i * self.segment_cell_num  
            seg_end = seg_start + self.segment_cell_num  
            cell_density_period = []
            for t in range(t_start, t_start + self.step_size):
                cell_density_t = []
                for j in range(seg_start, seg_end):
                    if self.road[j].cav_lane[t] > 0:
                        cell_dl_density = self.road[j].dl_density[t]
                        cell_gl_density = self.road[j].gl_density[t]
                        cell_density = (cell_dl_density * self.road[j].cav_lane[t] + cell_gl_density * (
                                1 - self.road[j].cav_lane[t])) / self.lane_num
                    else:
                        cell_density = self.road[j].density[t]
                    cell_density_t.append(cell_density)
                cell_density_period.append(np.mean(cell_density_t))
            cell_density = np.mean(cell_density_period)
            density.append(cell_density)

        return density

    def calculate_cell_density_difference(self, t_start):
        

        all_character = []
        for i in range(self.segment_num):
            
            
            
            
            seg_start = i * self.segment_cell_num  
            seg_end = seg_start + self.segment_cell_num  
            lane_density_dl_period = []
            lane_density_gl_period = []
            density_difference_dl_period = []
            density_difference_gl_period = []
            for t in range(t_start, t_start + self.step_size):
                lane_density_dl_t = []
                lane_density_gl_t = []
                density_difference_dl_t = []
                density_difference_gl_t = []
                for j in range(seg_start, seg_end):
                    if self.road[j].cav_lane[t] > 0:
                        lane_density_dl_t.append(self.road[j].dl_density[t])
                        lane_density_gl_t.append(self.road[j].gl_density[t])
                        _, _, _, dl_cri_density, _ = update_fundamental_parameters(1.0)
                        _, _, _, gl_cri_density, _ = update_fundamental_parameters(self.road[j].cav_ratio[t])
                        density_difference_dl_t.append(self.road[j].dl_density[t] - dl_cri_density)
                        density_difference_gl_t.append(self.road[j].gl_density[t] - gl_cri_density)
                    else:
                        lane_density_gl_t.append(self.road[j].density[t])
                        _, _, _, cri_density, _ = update_fundamental_parameters(self.road[j].cav_ratio[t])
                        density_difference_gl_t.append(self.road[j].density[t] - cri_density)
                if lane_density_dl_t:
                    lane_density_dl_period.append(np.mean(lane_density_dl_t))
                lane_density_gl_period.append(np.mean(lane_density_gl_t))
                if density_difference_dl_t:
                    density_difference_dl_period.append(np.mean(density_difference_dl_t))
                density_difference_gl_period.append(np.mean(density_difference_gl_t))
            if not lane_density_dl_period:
                lane_density_dl = 0
            else:
                lane_density_dl = np.mean(lane_density_dl_period)
            lane_density_gl = np.mean(lane_density_gl_period)
            if not density_difference_dl_period:
                density_difference_dl = 0
            else:
                density_difference_dl = np.mean(density_difference_dl_period)
            density_difference_gl = np.mean(density_difference_gl_period)
            all_character.append((lane_density_dl, lane_density_gl, density_difference_dl, density_difference_gl))

        return all_character

    def get_queue_metrics(self):
        
        if self.t <= 0:
            
            return {
                "main_queue": 0.0,
                "ramp_queue": 0.0,
                "ramp_inflow": 0.0,
            }

        
        t_end = self.t - 1

        
        window = self.step_size  
        t_start = max(0, t_end - window + 1)

        
        main_queue_list = []
        ramp_queue_list = []
        ramp_inflow_list = []

        for k in range(t_start, t_end + 1):
            main_q = self.input_supply[k] - self.actual_input[k]
            
            ramp_q = self.ramp_inflow[k] - self.road[self.ramp_entry_index - 1].ramp_output[k]

            main_queue_list.append(main_q)
            ramp_queue_list.append(ramp_q)
            ramp_inflow_list.append(self.ramp_inflow[k])

        return {
            "main_queue": main_queue_list,
            "ramp_queue": ramp_queue_list,
            "ramp_inflow": ramp_inflow_list,
        }

    def get_lane_cell_metrics(self):
        
        metrics = []

        if self.t <= 0:
            
            for _ in range(self.cell_num):
                metrics.append({
                    "ded_density": 0.0,
                    "gen_density": 0.0,
                    "n_ded_lanes": 0.0,
                    "gen_cav_penetration": 0.0,
                    "ded_flow": 0.0,
                    "gen_flow": 0.0,
                })
            return metrics

        t_end = self.t - 1
        window = self.step_size
        t_start = max(0, t_end - window + 1)

        for j in range(self.cell_num):
            cell = self.road[j]

            
            dl_density_list = []
            gl_density_list = []
            dl_flow_list = []
            gl_flow_list = []
            gl_cav_ratio_list = []

            for k in range(t_start, t_end + 1):
                if cell.cav_lane[k] > 0:
                    
                    dl_density_list.append(cell.dl_density[k])
                    gl_density_list.append(cell.gl_density[k])
                    dl_flow_list.append(cell.dl_output_flow[k])
                    gl_flow_list.append(cell.gl_output_flow[k])
                    gl_cav_ratio_list.append(cell.gl_cav_ratio[k])
                else:
                    
                    dl_density_list.append(0.0)
                    dl_flow_list.append(0.0)
                    gl_density_list.append(cell.density[k])
                    gl_flow_list.append(cell.gl_output_flow[k])  
                    gl_cav_ratio_list.append(cell.cav_ratio[k])

            
            n_ded_lanes = float(cell.cav_lane[t_end])

            metrics.append({
                "ded_density": dl_density_list,
                "gen_density": gl_density_list,
                "n_ded_lanes": n_ded_lanes,
                "gen_cav_penetration": gl_cav_ratio_list,
                "ded_flow": dl_flow_list,
                "gen_flow": gl_flow_list,
            })

        return metrics
