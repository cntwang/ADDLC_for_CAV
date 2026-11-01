import numpy as np
import pandas as pd
from ..flow_models.fd_parameters import (
    update_fundamental_parameters,
    update_mix_supply,
    update_dedicated_supply
)


class Cell:
    def __init__(self, sim_time_step, length=1 / 3, lane_num=3, dt=15):
        self.length = length
        self.dt = dt
        self.sim_time_step = sim_time_step
        self.lane_num = lane_num

        self.car_num = np.zeros(sim_time_step)
        self.cav_ratio = np.zeros(sim_time_step)
        self.cav_lane = np.zeros(sim_time_step)  

        self.density = np.zeros(sim_time_step)
        self.supply = np.zeros(sim_time_step)
        self.receive = np.zeros(sim_time_step)

        self.outflow = np.zeros(sim_time_step)
        self.outflow_cav = np.zeros(sim_time_step)

        
        self.all_dl_supply = np.zeros(sim_time_step)
        self.all_gl_supply = np.zeros(sim_time_step)
        self.all_dl_receive = np.zeros(sim_time_step)
        self.all_gl_receive = np.zeros(sim_time_step)
        self.gl_cav_ratio = np.zeros(sim_time_step)
        self.dl_cav_num = np.zeros(sim_time_step)
        self.gl_jam_car_num = np.zeros(sim_time_step)
        self.dl_jam_car_num = np.zeros(sim_time_step)
        self.feasibility_flag = np.zeros(sim_time_step, dtype=bool)

        self.ramp_cav_output = np.zeros(sim_time_step + 1)
        self.ramp_hv_output = np.zeros(sim_time_step + 1)
        self.ramp_output = np.zeros(sim_time_step + 1)

        
        self.capacity_limit = np.zeros(sim_time_step + 1)
        self.all_capacity = np.zeros(sim_time_step + 1)
        self.capacity_early_stop = [False] * sim_time_step

        
        self.delta = 1
        self.dl_cav_num = np.zeros(sim_time_step)
        self.gl_car_num = np.zeros(sim_time_step)
        self.gl_cav_ratio = np.zeros(sim_time_step)

        self.dl_output = np.zeros(sim_time_step)
        self.gl_cav_output = np.zeros(sim_time_step)
        self.gl_hv_output = np.zeros(sim_time_step)
        self.gl_output = np.zeros(sim_time_step)
        self.dl_density = np.zeros(sim_time_step)
        self.gl_density = np.zeros(sim_time_step)
        self.dl_output_flow = np.zeros(sim_time_step)
        self.gl_output_flow = np.zeros(sim_time_step)

    def update_timestep(self, t):

        if self.cav_lane[t] == 0.0:
            self._update_mixed_parameters(t)
        else:
            self._update_dedicated_parameters(t)

    def _update_mixed_parameters(self, t):
        jam_density, q_max, wave_speed, cri_density, q_max_drop = update_fundamental_parameters(self.cav_ratio[t])
        self.all_capacity[t] = jam_density * self.length * self.lane_num
        density = self.car_num[t] / (self.length * self.lane_num)
        self.density[t] = density
        supply, receive = update_mix_supply(
            q_max, jam_density, cri_density, density,
            self.car_num[t], wave_speed, q_max_drop, self.lane_num
        )
        self.supply[t] = supply
        self.receive[t] = receive

    def _update_dedicated_parameters(self, t):
        self.capacity_early_stop[t] = True
        cav_lane_num = self.cav_lane[t]
        gl_lane_num = self.lane_num - cav_lane_num
        dl_jam_density, dl_q_max, dl_wave_speed, dl_cri_density, _ = update_fundamental_parameters(1.0)

        total_cav_num = self.car_num[t] * self.cav_ratio[t]
        total_hv_num = self.car_num[t] * (1 - self.cav_ratio[t])

        
        dl_opt_num = dl_cri_density * cav_lane_num * self.length
        dl_max_num = dl_jam_density * cav_lane_num * self.length
        dl_cav_num = min(total_cav_num, dl_opt_num)  
        gl_cav_num = total_cav_num - dl_cav_num
        gl_car_num = gl_cav_num + total_hv_num

        gl_cav_ratio = gl_cav_num / gl_car_num if gl_car_num > 0 else 0
        gl_jam_density, gl_q_max, gl_wave_speed, gl_cri_density, gl_q_max_drop = update_fundamental_parameters(gl_cav_ratio)
        max_gl_capacity = gl_jam_density * self.length * gl_lane_num

        if gl_car_num > max_gl_capacity:  
            if gl_cav_num > 0:  
                dl_max_cav = min(total_cav_num, dl_max_num)
                for dl_cav in np.arange(dl_cav_num, dl_max_cav, self.delta):  
                    gl_cav_num = total_cav_num - dl_cav
                    gl_car_num = gl_cav_num + total_hv_num
                    gl_cav_ratio = gl_cav_num / gl_car_num if gl_car_num > 0 else 0
                    gl_jam_density, *_ = update_fundamental_parameters(gl_cav_ratio)
                    max_gl_capacity = gl_jam_density * self.length * gl_lane_num
                    if gl_car_num <= max_gl_capacity:
                        self.dl_cav_num[t] = dl_cav
                        self.gl_car_num[t] = gl_car_num
                        self.gl_cav_ratio[t] = gl_cav_ratio
                        self.capacity_early_stop[t] = False
                        break
        else:
            self.dl_cav_num[t] = dl_cav_num
            self.gl_car_num[t] = gl_car_num
            self.gl_cav_ratio[t] = gl_cav_ratio
            self.capacity_early_stop[t] = False

        gl_jam_density, gl_q_max, gl_wave_speed, gl_cri_density, gl_q_max_drop = update_fundamental_parameters(self.gl_cav_ratio[t])
        self.dl_density[t] = self.dl_cav_num[t] / (self.length * cav_lane_num) if cav_lane_num > 0 else 0
        self.gl_density[t] = self.gl_car_num[t] / (self.length * gl_lane_num) if gl_lane_num > 0 else 0  

        self.all_capacity[t] = dl_jam_density * self.length * cav_lane_num + gl_jam_density * self.length * gl_lane_num
        self.density[t] = self.car_num[t] / (self.length * self.lane_num)

        supply, receive, all_dl_supply, all_gl_supply, all_dl_receive, all_gl_receive = update_dedicated_supply(
            dl_q_max, dl_jam_density, self.dl_density[t], self.dl_cav_num[t], cav_lane_num, dl_wave_speed,
            gl_q_max, gl_jam_density, gl_cri_density, self.gl_density[t], self.gl_car_num[t], gl_wave_speed, gl_q_max_drop
        )
        
        self.supply[t] = supply
        self.receive[t] = receive
        self.all_dl_supply[t] = all_dl_supply
        self.all_gl_supply[t] = all_gl_supply
        self.all_dl_receive[t] = all_dl_receive
        self.all_gl_receive[t] = all_gl_receive

    def cal_outflow(self, t, down_cell=None, ramp=None):
        if self.cav_lane[t] == 0.0:
            cav_supply = self.supply[t] * self.cav_ratio[t]
            hv_supply = self.supply[t] * (1 - self.cav_ratio[t])
        else:
            cav_supply = self.all_dl_supply[t] + self.all_gl_supply[t] * self.gl_cav_ratio[t]
            hv_supply = self.all_gl_supply[t] * (1 - self.gl_cav_ratio[t])

        if down_cell:  
            if self.cav_lane[t] > down_cell.cav_lane[t]:
                if down_cell.cav_lane[t] == 0:
                    cav_receive = down_cell.receive[t] * (cav_supply / (cav_supply + hv_supply)) if cav_supply + hv_supply > 0 else down_cell.receive[t] * down_cell.cav_ratio[t]
                    hv_receive = down_cell.receive[t] * (1 - (cav_supply / (cav_supply + hv_supply))) if cav_supply + hv_supply > 0 else down_cell.receive[t] * (1 - down_cell.cav_ratio[t])
                else:
                    if cav_supply - down_cell.all_dl_receive[t] > 0:
                        cav_receive = down_cell.all_dl_receive[t] + down_cell.all_gl_receive[t] * ((cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t]))
                        hv_receive = down_cell.all_gl_receive[t] * (1 - ((cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t])))
                    else:
                        cav_receive = down_cell.all_dl_receive[t]
                        hv_receive = down_cell.all_gl_receive[t]
            elif self.cav_lane[t] == down_cell.cav_lane[t]:
                if down_cell.cav_lane[t] == 0:
                    cav_receive = down_cell.receive[t] * self.cav_ratio[t]
                    hv_receive = down_cell.receive[t] * (1 - self.cav_ratio[t])
                else:
                    if cav_supply - down_cell.all_dl_receive[t] > 0:
                        cav_receive = down_cell.all_dl_receive[t] + down_cell.all_gl_receive[t] * ((cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t]))
                        hv_receive = down_cell.all_gl_receive[t] * (1 - ((cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t])))
                    else:
                        cav_receive = down_cell.all_dl_receive[t]
                        hv_receive = down_cell.all_gl_receive[t]
            else:
                if cav_supply > down_cell.all_dl_receive[t]:
                    cav_receive = down_cell.all_dl_receive[t] + down_cell.all_gl_receive[t] * (cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t])
                    hv_receive = down_cell.all_gl_receive[t] * (1 - (cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply - down_cell.all_dl_receive[t]))
                else:
                    cav_receive = down_cell.all_dl_receive[t]
                    hv_receive = down_cell.all_gl_receive[t]
        else:
            cav_receive = float("inf")
            hv_receive = float("inf")

        if ramp is not None:
            ramp_supply = ramp[0]
            ramp_cav_supply = ramp_supply * ramp[1]
            ramp_hv_supply = ramp_supply - ramp_cav_supply
            if down_cell.cav_lane[t] == 0:  
                cav_receive = down_cell.receive[t] * ((ramp_cav_supply + cav_supply) / (cav_supply + hv_supply + ramp_supply)) if cav_supply + hv_supply + ramp_supply > 0 else down_cell.receive[t] * down_cell.cav_ratio[t]  
                hv_receive = down_cell.receive[t] * (1 - (ramp_cav_supply + cav_supply) / (cav_supply + hv_supply + ramp_supply)) if cav_supply + hv_supply + ramp_supply > 0 else down_cell.receive[t] * (1 - down_cell.cav_ratio[t])
            else:  
                if cav_supply + ramp_cav_supply > down_cell.all_dl_receive[t]:
                    cav_receive = down_cell.all_dl_receive[t] + down_cell.all_gl_receive[t] * (cav_supply + ramp_cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply + ramp_supply - down_cell.all_dl_receive[t])
                    hv_receive = down_cell.all_gl_receive[t] * (1 - (cav_supply + ramp_cav_supply - down_cell.all_dl_receive[t]) / (cav_supply + hv_supply + ramp_supply - down_cell.all_dl_receive[t]))
            r_cav = 0 if cav_supply + ramp_cav_supply == 0 else cav_receive / (cav_supply + ramp_cav_supply)  
            r_hv = 0 if hv_supply + ramp_hv_supply == 0 else hv_receive / (hv_supply + ramp_hv_supply)  
            cav_output = cav_supply if r_cav >= 1 else cav_supply * r_cav
            hv_output = hv_supply if r_hv >= 1 else hv_supply * r_hv
            self.ramp_cav_output[t] = ramp_cav_supply if r_cav >= 1 else ramp_cav_supply * r_cav
            self.ramp_hv_output[t] = ramp_hv_supply if r_hv >= 1 else ramp_hv_supply * r_hv
            self.ramp_output[t] = self.ramp_cav_output[t] + self.ramp_hv_output[t]
        else:
            cav_output = min(cav_supply, cav_receive)
            hv_output = min(hv_supply, hv_receive)

        
        if cav_output == cav_supply and self.cav_lane[t] > 0:
            self.dl_output[t] = self.all_dl_supply[t]  
            self.dl_output_flow[t] = self.dl_output[t] / ((self.dt / 3600) * self.cav_lane[t])  
            self.gl_cav_output[t] = self.all_gl_supply[t] * self.gl_cav_ratio[t]  
        elif cav_output < cav_supply and self.cav_lane[t] > 0:
            o_cav = cav_output / cav_supply if cav_supply > 0 else 0
            self.dl_output[t] = self.all_dl_supply[t] * o_cav  
            self.dl_output_flow[t] = self.dl_output[t] / ((self.dt / 3600) * self.cav_lane[t])  
            self.gl_cav_output[t] = self.all_gl_supply[t] * self.gl_cav_ratio[t] * o_cav  
        self.gl_hv_output[t] = hv_output  
        self.gl_output[t] = self.gl_cav_output[t] + self.gl_hv_output[t]  
        self.gl_output_flow[t] = self.gl_output[t] / ((self.dt / 3600) * (self.lane_num - self.cav_lane[t]))  
        
        

        self.outflow_cav[t] = cav_output
        self.outflow[t] = cav_output + hv_output

    def update_state(self, t, up_input, up_input_cav):

        new_car = self.car_num[t] + up_input - self.outflow[t]
        new_cav = self.car_num[t] * self.cav_ratio[t] + up_input_cav - self.outflow_cav[t]
        new_ratio = 0.0 if new_car < 1e-6 else new_cav / new_car
        self.car_num[t + 1] = new_car
        self.cav_ratio[t + 1] = max(0, new_ratio)
