import os
import random
import numpy as np
import pandas as pd
from ..flow_models.fd_parameters import update_fundamental_parameters

import glob
import os


def _align_to_steps(values_per_block, block_seconds, dt_seconds):
    
    values_per_block = np.asarray(values_per_block, dtype=float)
    steps_per_block = int(round(block_seconds / dt_seconds))
    per_step = values_per_block / steps_per_block
    return np.repeat(per_step, steps_per_block)


def _load_flow_series_from_csv(path, preferred_cols=("Flow", "flow"), unit="veh_per_5min", dt_seconds=15):
    
    df = pd.read_csv(path, dtype=str)
    col = None
    for c in preferred_cols:
        if c in df.columns:
            col = c
            break
    if col is None:
        
        best, best_cnt = None, -1
        for c in df.columns:
            vals = pd.to_numeric(
                df[c].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce"
            )
            cnt = vals.notna().sum()
            if cnt > best_cnt:
                best, best_cnt = c, cnt
        col = best

    series = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    ).fillna(0.0)

    if unit.lower() == "veh_per_5min":
        per_step = _align_to_steps(series.to_numpy(float), 5*60, dt_seconds)
    elif unit.lower() == "veh_per_step":
        per_step = series.to_numpy(float)
    else:
        raise ValueError("only veh_per_5min/veh_per_step are supported here")

    
    per_step = per_step.astype(float)
    per_step[~np.isfinite(per_step)] = 0.0
    per_step = np.clip(per_step, 0.0, None)
    return per_step


class RealFlowDataset:
    
    def __init__(self, data_dir, main_glob="station_318017_5min_*.csv",
                 ramp_glob="station_3021081_5min_*.csv", dt_seconds=15):
        self.dt = dt_seconds
        self.main_files = sorted(glob.glob(os.path.join(data_dir, main_glob)))
        self.ramp_files = sorted(glob.glob(os.path.join(data_dir, ramp_glob)))
        
        def key(p): return os.path.splitext(os.path.basename(p))[0].split("_")[-1]
        m_map = {key(p): p for p in self.main_files}
        r_map = {key(p): p for p in self.ramp_files}
        self.days = sorted(set(m_map) & set(r_map))
        self.pairs = [(m_map[d], r_map[d]) for d in self.days]
        if not self.pairs:
            raise RuntimeError("没有找到可配对的主路/匝道 CSV")

    def __len__(self): return len(self.pairs)

    def get_pair_by_episode(self, episode_id, start_step=0, sim_steps=2520):
        
        idx = int(episode_id) % len(self.pairs)
        main_path, ramp_path = self.pairs[idx]
        main = _load_flow_series_from_csv(main_path, unit="veh_per_5min", dt_seconds=self.dt)
        ramp = _load_flow_series_from_csv(ramp_path, unit="veh_per_5min", dt_seconds=self.dt)

        
        def slice_pad(x):
            x = x[start_step:start_step+sim_steps]
            if len(x) < sim_steps:
                x = np.concatenate([x, np.zeros(sim_steps-len(x), dtype=float)])
            return x

        return slice_pad(main), slice_pad(ramp)


class InputRandomManager:
    def __init__(self, seed, dt=15, length=1/3):  

        
        base_ss = np.random.SeedSequence(seed)
        
        child_seeds = base_ss.spawn(6)

        self.rng_main_flow = np.random.default_rng(child_seeds[0])
        self.rng_main_cav = np.random.default_rng(child_seeds[1])
        self.rng_ramp_flow = np.random.default_rng(child_seeds[2])
        self.rng_ramp_cav = np.random.default_rng(child_seeds[3])
        self.rng_init_vehicle = np.random.default_rng(child_seeds[4])
        self.rng_init_cav = np.random.default_rng(child_seeds[5])

        
        self.log_main_flow = []
        self.log_main_cav = []
        self.log_ramp_flow = []
        self.log_ramp_cav = []
        self.log_init_vehicle = []
        self.log_init_cav = []

        self.main_flow_delta = 50  
        self.ramp_delta = 5  
        self.dt = dt
        self.hour_step = 3600 / self.dt  
        self.all_input_time = int(10.5 * 3600 / self.dt)  
        self.length = length

    def generate_main_input(self, duration, delta, input_level, test=False, fixed_input=1000):

        if not test:
            min_q = max(0, input_level - delta)
            max_q = input_level + delta
        else:  
            min_q = max(0, fixed_input - self.main_flow_delta)
            max_q = fixed_input + self.main_flow_delta
        dt_min_num = min_q / self.hour_step
        dt_max_num = max_q / self.hour_step
        main_supply = self.rng_main_flow.uniform(dt_min_num, dt_max_num, size=duration)
        self.log_main_flow.append(main_supply)

        return main_supply

    def generate_main_input_cav_ratio(self, sim_time_step, cav_ratio=None):
        
        if cav_ratio is None:
            main_cav_ratio = self.rng_main_cav.uniform(0.0, 1.0, size=sim_time_step)
        else:
            delta = 0.02
            lower_bound = max(0.0, cav_ratio - delta)
            upper_bound = min(1.0, cav_ratio + delta)
            main_cav_ratio = self.rng_main_cav.uniform(lower_bound, upper_bound, size=sim_time_step)
        self.log_main_cav.append(main_cav_ratio)

        return main_cav_ratio

    def generate_main_initial_cav_ratio(self, cell_num, cav_ratio=None):
        
        if cav_ratio is None:
            initial_cav = self.rng_init_cav.uniform(0.0, 1.0, size=cell_num)
        elif cav_ratio == 0:
            lower_bound = 0
            upper_bound = 0
            initial_cav = self.rng_init_cav.uniform(lower_bound, upper_bound, size=cell_num)
        else:
            delta = 0.05
            lower_bound = max(0.0, cav_ratio - delta)
            upper_bound = min(1.0, cav_ratio + delta)
            initial_cav = self.rng_init_cav.uniform(lower_bound, upper_bound, size=cell_num)
        self.log_init_cav.append(initial_cav)

        return initial_cav

    
    def generate_initial_car_num(self, initial_cav_array, cell_num, density_level, lane_num=3):
        

        max_car_nums = np.zeros(cell_num)
        car_num = np.zeros(cell_num)
        for i in range(cell_num):
            
            jam_density, _, _, _, _ = update_fundamental_parameters(cav_ratio=initial_cav_array[i])
            
            max_car_nums[i] = jam_density * self.length * lane_num
            if density_level is None:
                lower_bound = 0
                upper_bound = max_car_nums[i]
            elif density_level == 1:
                lower_bound = 0
                upper_bound = max_car_nums[i] / 5
            elif density_level == 2:
                lower_bound = max_car_nums[i] / 5
                upper_bound = (max_car_nums[i] * 2) / 5
            elif density_level == 3:
                lower_bound = (max_car_nums[i] * 2) / 5
                upper_bound = (max_car_nums[i] * 3) / 5
            elif density_level == 4:
                lower_bound = (max_car_nums[i] * 3) / 5
                upper_bound = (max_car_nums[i] * 4) / 5
            elif density_level == 5:
                lower_bound = (max_car_nums[i] * 4) / 5
                upper_bound = max_car_nums[i]
            else:
                lower_bound = 0.3
                upper_bound = 0.8
            car_num[i] = self.rng_init_vehicle.uniform(low=lower_bound, high=upper_bound)

        self.log_init_vehicle.append(car_num)
        return car_num

    def generate_ramp_cav_ratio(self, sim_time_step, cav_ratio=None):

        if cav_ratio is None:
            ramp_cav_ratio = self.rng_ramp_cav.uniform(0.0, 1.0, size=sim_time_step)
        else:
            delta = 0.02
            lower_bound = max(0.0, cav_ratio - delta)
            upper_bound = min(1.0, cav_ratio + delta)
            ramp_cav_ratio = self.rng_ramp_cav.uniform(lower_bound, upper_bound, size=sim_time_step)
        self.log_ramp_cav.append(ramp_cav_ratio)

        return ramp_cav_ratio

    def generate_ramp_input(self, duration, delta, input_q, test=False, fixed_input=1000):

        if not test:
            min_q = max(0, input_q - delta)
            max_q = input_q + delta
        else:  
            min_q = max(0, fixed_input - self.ramp_delta)
            max_q = fixed_input + self.ramp_delta
        dt_min_num = min_q / self.hour_step
        dt_max_num = max_q / self.hour_step
        ramp_supply = self.rng_ramp_flow.uniform(dt_min_num, dt_max_num, size=duration)
        self.log_main_flow.append(ramp_supply)

        return ramp_supply

    def export_log(self):
        
        return {
            "main_flow": np.array(self.log_main_flow),
            "main_cav_ratio": np.array(self.log_main_cav),
            "ramp_flow": np.array(self.log_ramp_flow),
            "ramp_cav_ratio": np.array(self.log_ramp_cav),
            "initial_vehicle": np.array(self.log_init_vehicle),
            "initial_cav_ratio": np.array(self.log_init_cav),
        }

    
    def concatenate_main_flow_segments(self, flow_config, input_type):

        result = []
        if input_type == "fixed":
            q_base_level = flow_config["base_level"]  
            result = self.generate_main_input(self.all_input_time, self.main_flow_delta, q_base_level)
        elif input_type == "single_peak":
            start = int(flow_config["start_step"] * 3600 / self.dt)  
            dur = int(flow_config["duration"] * 3600 / self.dt)  
            trans = int(flow_config["transition"] * 3600 / self.dt)  
            q_base_level = flow_config["base_level"]  
            q_peak_level = flow_config["peak_level"]  
            pre_base_dur = start - trans  
            post_base_dur = self.all_input_time - (start + dur + trans)  
            if pre_base_dur < 0 or post_base_dur < 0:
                raise ValueError("时间配置非法，仿真总时长不足以容纳单峰结构。")
            pre_base = self.generate_main_input(pre_base_dur, self.main_flow_delta, q_base_level)
            peak = self.generate_main_input(dur, self.main_flow_delta, q_peak_level)
            post_base = self.generate_main_input(post_base_dur, self.main_flow_delta, q_base_level)
            pre_transition = self.generate_transition_segment(pre_base[-1], peak[0], trans, "cosine", 0.2)
            post_transition = self.generate_transition_segment(peak[-1], post_base[0], trans, "cosine", 0.2)
            result = np.concatenate([pre_base, pre_transition, peak, post_transition, post_base])
        elif input_type == "double_peak":
            q_base_level = flow_config["base_level"]
            q1_peak_level = flow_config["peak1_level"]
            q2_peak_level = flow_config["peak2_level"]
            start1 = int(flow_config["start1_step"] * 3600 / self.dt)  
            start2 = int(flow_config["start2_step"] * 3600 / self.dt)  
            dur1 = int(flow_config["duration1"] * 3600 / self.dt)  
            dur2 = int(flow_config["duration2"] * 3600 / self.dt)  
            trans1 = int(flow_config["transition1"] * 3600 / self.dt)  
            trans2 = int(flow_config["transition2"] * 3600 / self.dt)  
            pre_base_dur1 = start1 - trans1  
            post_base_dur1 = (start2 - trans2) - (start1 + dur1 + trans1)  
            post_base_dur2 = self.all_input_time - (start2 + dur2 + trans2)  
            if pre_base_dur1 < 0 or post_base_dur1 < 0 or post_base_dur2 < 0:
                raise ValueError("时间配置非法，仿真总时长不足以容纳单峰结构。")
            pre1_base = self.generate_main_input(pre_base_dur1, self.main_flow_delta, q_base_level)
            peak1 = self.generate_main_input(dur1, self.main_flow_delta, q1_peak_level)
            pre_transition1 = self.generate_transition_segment(pre1_base[-1], peak1[0], trans1, "cosine", 0.2)
            post1_base = self.generate_main_input(post_base_dur1, self.main_flow_delta, q_base_level)
            post_transition1 = self.generate_transition_segment(peak1[-1], post1_base[0], trans1, "cosine", 0.2)
            peak2 = self.generate_main_input(dur2, self.main_flow_delta, q2_peak_level)
            pre_transition2 = self.generate_transition_segment(post1_base[-1], peak2[0], trans2, "cosine", 0.2)
            post2_base = self.generate_main_input(post_base_dur2, self.main_flow_delta, q_base_level)
            post_transition2 = self.generate_transition_segment(peak2[-1], post2_base[0], trans2, "cosine", 0.2)
            result = np.concatenate([pre1_base, pre_transition1, peak1, post_transition1, post1_base, pre_transition2,
                                     peak2, post_transition2, post2_base])
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

        return np.array(result)

    def concatenate_ramp_flow_segments(self, flow_config):

        start = int(flow_config["start_step"] * 3600 / self.dt)  
        dur = int(flow_config["duration"] * 3600 / self.dt)  
        trans = int(flow_config["transition"] * 3600 / self.dt)  
        q_base_flow = flow_config["base_level"]  
        q_peak_flow = flow_config["peak_level"]  
        pre_base_dur = start - trans  
        post_base_dur = self.all_input_time - (start + dur + trans)  
        if pre_base_dur < 0 or post_base_dur < 0:
            raise ValueError("时间配置非法，仿真总时长不足以容纳单峰结构。")
        pre_base = self.generate_ramp_input(pre_base_dur, self.ramp_delta, q_base_flow)
        peak = self.generate_ramp_input(dur, self.ramp_delta, q_peak_flow)
        post_base = self.generate_ramp_input(post_base_dur, self.ramp_delta, q_base_flow)
        pre_transition = self.generate_transition_segment(pre_base[-1], peak[0], trans, "cosine", 0.02)
        post_transition = self.generate_transition_segment(peak[-1], post_base[0], trans, "cosine", 0.02)
        result = np.concatenate([pre_base, pre_transition, peak, post_transition, post_base])

        return np.array(result)

    
    def concatenate_cav_ratio(self, cav_ratio_config, input_type):

        if input_type == "fixed":
            r_base_level = cav_ratio_config["base_level"]  
            main_result = self.generate_main_input_cav_ratio(self.all_input_time, r_base_level)
            ramp_result = self.generate_ramp_cav_ratio(self.all_input_time, r_base_level)
        elif input_type == "synchronous":
            r_base_level1 = cav_ratio_config["base_level1"]  
            r_base_level2 = cav_ratio_config["base_level2"]  
            dur = int(cav_ratio_config["duration"] * 3600 / self.dt)  
            trans = int(cav_ratio_config["transition"] * 3600 / self.dt)  
            main_pre_result = self.generate_main_input_cav_ratio(dur, r_base_level1)
            main_post_result = self.generate_main_input_cav_ratio(self.all_input_time - dur - trans, r_base_level2)
            main_transition = self.generate_transition_segment(main_pre_result[-1], main_post_result[0], trans, "linear", 0.02)
            ramp_pre_result = self.generate_ramp_cav_ratio(dur, r_base_level1)
            ramp_post_result = self.generate_ramp_cav_ratio(self.all_input_time - dur - trans, r_base_level2)
            ramp_transition = self.generate_transition_segment(ramp_pre_result[-1], ramp_post_result[0], trans, "linear", 0.02)
            main_result = np.concatenate([main_pre_result, main_transition, main_post_result])
            ramp_result = np.concatenate([ramp_pre_result, ramp_transition, ramp_post_result])
        elif input_type == "opposite":
            main_base_level = cav_ratio_config["main_level"]
            ramp_base_level = cav_ratio_config["ramp_level"]
            main_result = self.generate_main_input_cav_ratio(self.all_input_time, main_base_level)
            ramp_result = self.generate_ramp_cav_ratio(self.all_input_time, ramp_base_level)
        else:
            raise ValueError("Unknown input_type")

        return np.array(main_result), np.array(ramp_result)

    
    def generate_transition_segment(self, start, end, time_steps, method="cosine", noise_scale=5.0):
        
        if method == "linear":
            base = np.linspace(start, end, time_steps)
        elif method == "cosine":
            x = np.linspace(0, np.pi, time_steps)
            base = (start + end) / 2 + (start - end) / 2 * np.cos(x)
        else:
            raise ValueError("Unsupported transition method.")

        noise = np.random.normal(loc=0.0, scale=noise_scale, size=time_steps)
        return np.clip(base + noise, 0, None)  


def compute_actual_input_step(t, input_supply_t, input_cav_ratio_t, first_cell):
    
    cav_supply = input_supply_t * input_cav_ratio_t
    hv_supply = input_supply_t * (1 - input_cav_ratio_t)
    
    
    
    
    
    cav_receive = first_cell.receive[t] * input_cav_ratio_t
    hv_receive = first_cell.receive[t] * (1 - input_cav_ratio_t)
    cav_input = min(cav_supply, cav_receive)
    hv_input = min(hv_supply, hv_receive)
    actual_input = cav_input + hv_input
    return actual_input, cav_input


































def generate_main_flow_config(input_type: str, lane_num=3):  

    _, q_max_0, _, _, _ = update_fundamental_parameters(0.0)
    _, q_max_1, _, _, _ = update_fundamental_parameters(1.0)
    q_0 = q_max_0 * lane_num  
    q_1 = q_max_1 * lane_num  
    q_min = q_0 - 1000
    q_max = q_1 + 1000
    q_pool = list(range(int(q_min), int(q_max), 1000))  
    stage = int(q_max - q_min) / 3
    q_low_pool = list(range(int(q_min), int(q_min + stage), 1000))  
    q_middle_pool = list(range(int(q_min + stage), int(q_min + stage * 2), 1000))  
    q_high_pool = list(range(int(q_min + stage * 2), int(q_max), 1000))  
    if input_type == "fixed":  
        config = {
            "type": "fixed",
            "base_level": np.random.choice(q_pool)  
        }
    elif input_type == "single_peak":  
        duration_options = [1.0, 1.5, 2.0, 2.5, 3.0]  
        start_step_options = [1.5, 2.0, 2.5, 3.0]  
        transition_options = [0.25, 0.5, 0.75, 1.0]  
        config = {
            "type": "single_peak",
            "base_level": np.random.choice(q_low_pool),
            "peak_level": np.random.choice(q_high_pool),
            "start_step": np.random.choice(start_step_options),
            "duration": np.random.choice(duration_options),
            "transition": np.random.choice(transition_options)
        }
    elif input_type == "double_peak":  
        peak_duration_options = [0.5, 1, 1.5, 2.0]  
        peak1_start_step_options = [1.5, 2.0]  
        peak2_start_step_options = [6.5, 7.0]  
        transition_options = [0.25, 0.5, 0.75, 1.0]  
        config = {
            "type": "double_peak",
            "base_level": np.random.choice(q_low_pool),
            "peak1_level": np.random.choice(q_high_pool),
            "peak2_level": np.random.choice(q_high_pool),
            "start1_step": np.random.choice(peak1_start_step_options),
            "start2_step": np.random.choice(peak2_start_step_options),
            "duration1": np.random.choice(peak_duration_options),
            "duration2": np.random.choice(peak_duration_options),
            "transition1": np.random.choice(transition_options),
            "transition2": np.random.choice(transition_options)
        }
    elif input_type == "fluctuation":
        fluctuation_duration_options = [0.25, 0.5, 0.75]  
        start_step_options = [1.5, 2, 2.5, 3]  
        fluctuation_num = [2, 3]  
        fluct_num = np.random.choice(fluctuation_num)  
        segments = []
        for _ in range(fluct_num):
            seg = {
                "high": float(np.random.choice(q_high_pool)),
                "low": float(np.random.choice(q_middle_pool)),
                "duration": float(np.random.choice(fluctuation_duration_options))
            }
            segments.append(seg)
        config = {
            "type": "fluctuation",
            "base_level": np.random.choice(q_low_pool),
            "fluctuation_start_step": np.random.choice(start_step_options),
            "segments": segments
        }
    else:
        raise ValueError("Unknown input_type")
    return config


def generate_ramp_flow_config():  

    duration_options = [1.0, 1.5, 2.0, 2.5, 3.0]  
    start_step_options = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]  
    transition_options = [0.25, 0.5, 0.75, 1.0]  
    base_flow = [1000, 1500, 2000, 2500]  
    peak_flow = [2500, 3000, 3500, 4000]  
    config = {
        "base_level": np.random.choice(base_flow),
        "peak_level": np.random.choice(peak_flow),
        "start_step": np.random.choice(start_step_options),
        "duration": np.random.choice(duration_options),
        "transition": np.random.choice(transition_options)
    }
    return config


def generate_cav_ratio_config(input_type: str):  

    if input_type == "fixed":  
        base_level_option = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        base_level = np.random.choice(base_level_option)
        config = {
            "type": "fixed",
            "base_level": base_level,  
        }
    elif input_type == "synchronous":  
        base_level_option1 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
        base_level_option2 = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
        duration_options1 = [0.5, 1, 1.5, 2.0, 2.5, 3.0]  
        base_level1 = np.random.choice(base_level_option1)  
        base_level2 = np.random.choice(base_level_option2)  
        transition_options = [0.25, 0.5, 0.75, 1.0]  
        config = {
            "type": "synchronous",
            "base_level1": base_level1,
            "base_level2": base_level2,
            "duration": np.random.choice(duration_options1),
            "transition": np.random.choice(transition_options)
        }
    elif input_type == "opposite":  
        main_level_option = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  
        main_level = np.random.choice(main_level_option)  
        ramp_level = 1 - main_level
        config = {
            "type": "opposite",
            "main_level": main_level,
            "ramp_level": ramp_level
        }
    else:
        raise ValueError("Unknown input_type")

    return config


class EpisodeInputGenerator:
    def __init__(self, seed=1):
        
        self.main_flow_types = ["fixed", "single_peak", "double_peak"]
        self.cav_ratio_types = ["fixed", "synchronous", "opposite"]
        self.ramp_positions = list(range(30, 101, 10))  
        self.seed = seed
        self.episode_cav_ratio = 0

    def choose_from_pool(self, episode_id: int, target: str = "train"):

        if target == "train":
            if episode_id <= 5000:  
                flow_type = "fixed"
                cav_type = "fixed"
            elif 5000 < episode_id <= 10000:
                flow_type = "fixed"
                cav_type = random.choice(["synchronous", "opposite"])
            elif 10000 < episode_id <= 15000:
                flow_type = random.choice(["single_peak", "double_peak"])
                cav_type = random.choice(self.cav_ratio_types)
            elif 15000 < episode_id <= 20000:
                flow_type = random.choice(self.main_flow_types)
                cav_type = random.choice(self.cav_ratio_types)
            else:
                raise ValueError("训练episode数达到设置值")
        elif target == "test":
            flow_type = "fixed"
            cav_type = "fixed"
        else:
            raise ValueError("target must be 'train' or 'test'")
        return flow_type, cav_type

    def generate_episode_reset_config(self, episode_id: int, target: str = "train"):

        input_manager = InputRandomManager(self.seed + episode_id)
        flow_type, cav_type = self.choose_from_pool(episode_id, target)
        main_config = generate_main_flow_config(flow_type)
        ramp_config = generate_ramp_flow_config()
        cav_ratio_config = generate_cav_ratio_config(cav_type)
        main_cav_ratio, ramp_cav_ratio = input_manager.concatenate_cav_ratio(cav_ratio_config, cav_type)
        if cav_type == "fixed":
            main_cav_ratio_1 = cav_ratio_config["base_level"]
        elif cav_type == "synchronous":
            main_cav_ratio_1 = cav_ratio_config["base_level1"]
        elif cav_type == "opposite":
            main_cav_ratio_1 = cav_ratio_config["main_level"]
        else:
            raise ValueError("Unknown cav_type")
        main_flow = input_manager.concatenate_main_flow_segments(main_config, flow_type)
        ramp_flow = input_manager.concatenate_ramp_flow_segments(ramp_config)
        init_cav_ratio = input_manager.generate_main_initial_cav_ratio(120, main_cav_ratio_1)
        init_car = input_manager.generate_initial_car_num(init_cav_ratio, 120, 0)
        ramp_entry = random.choice(self.ramp_positions)
        reset_config = {
            "init_car": init_car,
            "init_cav_ratio": init_cav_ratio,
            "input_supply": main_flow,
            "input_cav_supply": main_cav_ratio,
            "ramp_inflow": ramp_flow,
            "ramp_cav_ratio": ramp_cav_ratio,
            "ramp_entry_index": ramp_entry,
            "input_level": flow_type,
            "cav_type": cav_type,
        }
        return reset_config

    def generate_test_reset_config(self, seed, peak, flat, cav_level):

        input_manager = InputRandomManager(self.seed + seed)
        
        
        flow_type = "single_peak"
        cav_type = "fixed"
        duration_options = 2.0  
        start_step_options = 1.5  
        transition_options = 1.0  
        main_config = {
            "type": "single_peak",
            "base_level": flat,
            "peak_level": peak,
            "start_step": start_step_options,  
            "duration": duration_options,  
            "transition": transition_options,  
        }
        
        ramp_duration_options = 2.0  
        ramp_start_step_options = 2.5  
        ramp_transition_options = 1.0  
        ramp_base_flow = 300  
        ramp_peak_flow = 700  
        ramp_config = {
            "base_level": ramp_base_flow,  
            "peak_level": ramp_peak_flow,  
            "start_step": ramp_start_step_options,  
            "duration": ramp_duration_options,  
            "transition": ramp_transition_options,  
        }
        
        
        
        
        
        
        
        
        cav_ratio = np.random.choice(cav_level)
        
        
        
        
        
        self.episode_cav_ratio = cav_ratio
        main_and_ramp_cav_ratio = np.full(input_manager.all_input_time, cav_ratio)
        main_flow = input_manager.concatenate_main_flow_segments(main_config, "single_peak")
        ramp_flow = input_manager.concatenate_ramp_flow_segments(ramp_config)
        init_cav_ratio = input_manager.generate_main_initial_cav_ratio(120, main_and_ramp_cav_ratio[0])
        init_car = input_manager.generate_initial_car_num(init_cav_ratio, 120, 0)
        ramp_entry = 60  
        reset_config = {
            "init_car": init_car,
            "init_cav_ratio": init_cav_ratio,
            "input_supply": main_flow,
            "input_cav_supply": main_and_ramp_cav_ratio,
            "ramp_inflow": ramp_flow,
            "ramp_cav_ratio": main_and_ramp_cav_ratio,
            "ramp_entry_index": ramp_entry,
            "input_level": flow_type,
            "cav_type": cav_type,
        }
        return reset_config


def save_reset_config_to_log(reset_config, episode_id, path="log_episode_config.csv"):
    def make_row(var_name, arr):
        arr = np.array(arr).flatten()
        return [episode_id, var_name] + arr.tolist()

    rows = []
    rows.append(make_row("init_car", reset_config["init_car"]))
    rows.append(make_row("init_cav_ratio", reset_config["init_cav_ratio"]))
    rows.append(make_row("main_supply", reset_config["input_supply"]))
    rows.append(make_row("main_cav_ratio", reset_config["input_cav_supply"]))
    rows.append(make_row("ramp_supply", reset_config["ramp_inflow"]))
    rows.append(make_row("ramp_cav_ratio", reset_config["ramp_cav_ratio"]))
    rows.append(make_row("ramp_entry_index", [reset_config["ramp_entry_index"]]))  

    df = pd.DataFrame(rows)
    df.to_csv(path, mode='a', index=False, header=not os.path.exists(path))
