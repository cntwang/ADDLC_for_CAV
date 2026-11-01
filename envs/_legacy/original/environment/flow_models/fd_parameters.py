import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps


def update_fundamental_parameters(cav_ratio, f_speed=80, dis_cav=7.0 / 1000, time_cav=0.6 / 3600,
                                  dis_hv=8.53 / 1000, time_hv=1.61 / 3600, capacity_drop_max=0.3):
    

    avg_dis = cav_ratio * dis_cav + (1 - cav_ratio) * dis_hv
    avg_time = cav_ratio * time_cav + (1 - cav_ratio) * time_hv
    jam_density = 1 / avg_dis
    cri_density = 1 / (avg_time * f_speed + avg_dis)
    q_max = f_speed / (avg_time * f_speed + avg_dis)
    capacity_drop_index = capacity_drop_max * (1 - cav_ratio)
    q_max_drop = q_max * (1 - capacity_drop_index)
    
    wave_speed = avg_dis / avg_time

    return jam_density, q_max, wave_speed, cri_density, q_max_drop


















def update_mix_supply(q_max, jam_density, cri_density, density, car_num, wave_speed, q_max_drop, lane_num=3, dt=15,
                      jam_max_ratio=0.4):  
    
    q_jam = q_max_drop * jam_max_ratio
    scale_down = (q_max_drop - q_jam) / (jam_density - cri_density)  

    if density <= cri_density:
        supply = max(0.0, min(q_max * dt / 3600 * lane_num, car_num))
    else:
        supply = max(0.0, min(car_num, (q_max_drop - scale_down * (density - cri_density)) * dt / 3600 * lane_num))
    receive = max(0.0, min(q_max, wave_speed * (jam_density - density)) * dt / 3600 * lane_num)
    return supply, receive


def update_dedicated_supply(dl_q_max, dl_jam_density, dl_density, dl_car_num, dl_num, dl_wave_speed,
                            gl_q_max, gl_jam_density, gl_cri_density, gl_density, gl_car_num, gl_wave_speed,
                            gl_q_max_drop, lane_num=3, dt=15, jam_max_ratio=0.4):
    
    dl_supply = max(0.0, min(dl_q_max * dt / 3600 * dl_num, dl_car_num))
    dl_receive = max(0.0, min(dl_q_max, dl_wave_speed * (dl_jam_density - dl_density))) * dt / 3600 * dl_num

    gl_q_jam = gl_q_max_drop * jam_max_ratio
    gl_scale_down = (gl_q_max_drop - gl_q_jam) / (gl_jam_density - gl_cri_density)

    gl_num = lane_num - dl_num
    if gl_density <= gl_cri_density:
        gl_supply = max(0.0, min(gl_q_max * dt / 3600 * gl_num, gl_car_num))
    else:
        gl_supply = max(0.0, min(gl_car_num,
                                 (gl_q_max_drop - gl_scale_down * (gl_density - gl_cri_density)) * dt / 3600 * gl_num))
    gl_receive = max(0.0, min(gl_q_max, gl_wave_speed * (gl_jam_density - gl_density))) * dt / 3600 * gl_num

    supply = dl_supply + gl_supply
    receive = dl_receive + gl_receive

    return supply, receive, dl_supply, gl_supply, dl_receive, gl_receive



def plot_supply_receive_vs_density(lane_num=3, length=1/3):
    
    cav_ratios = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]  
    cmap = colormaps.get_cmap('tab10')  
    plt.figure(figsize=(10, 6))
    for idx, cav_ratio in enumerate(cav_ratios):
        jam_density, q_max, wave_speed, cri_density, q_max_drop = update_fundamental_parameters(cav_ratio)
        densities = np.linspace(0, jam_density, 500)  
        supplies = []
        receives = []
        for rho in densities:
            car_num = rho * length * lane_num
            supply_cell, receive_cell = update_mix_supply(q_max, jam_density, cri_density, rho, car_num, wave_speed, q_max_drop)
            supply_lane = supply_cell / lane_num * 240
            receive_lane = receive_cell / lane_num * 240
            supplies.append(supply_lane)
            receives.append(receive_lane)

        color = cmap(idx % 10)
        plt.plot(densities, supplies, linestyle='-', color=color, label=f'CAV ratio = {cav_ratio:.2f}')
        plt.plot(densities, receives, linestyle='--', color=color, label=f'CAV ratio = {cav_ratio:.2f}')

    plt.title("Sending & Receiving Capacity vs. Density (All CAV Ratios)")
    plt.xlabel("Density (veh/km/lane)")
    plt.ylabel("Capacity (veh/h/lane)")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig("supply_receive_together.png", dpi=300)



if __name__ == "__main__":
    plot_supply_receive_vs_density()
    print(f"画完了")
