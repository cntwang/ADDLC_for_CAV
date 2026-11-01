from .environment.flow_models.fd_parameters import update_fundamental_parameters


def poisson_reward_with_cav(density, cav_ratio, reward_peak=0.5, reward_low=-0.5, incentive=0.5, slope_low=None, slope_high=None, penalty_slope=-0.5, interval=2):
    
    jam_density, _, _, rho_crit, _ = update_fundamental_parameters(cav_ratio)
    rho_cong = jam_density / 2
    if slope_low is None:
        slope_low = reward_peak / rho_crit
    if slope_high is None:
        slope_high = -(0.4 - reward_low) / (jam_density - rho_crit)
    reward = 0.0
    if density < rho_crit - interval:
        reward = slope_low * density
    elif rho_crit - interval <= density <= rho_crit:
        reward = slope_low * density + incentive
    elif rho_crit < density <= rho_cong:
        reward = 0.4 + slope_high * (density - rho_crit)
    else:
        reward = reward_peak + slope_high * (density - rho_crit) - 0.6
    return reward

def flow_reward(flow, cav_ratio, reward_peak=0.7, incentive=0.2, incentive_double=0.3, slope_low=None, interval=500):
    _, q_max, _, _, _ = update_fundamental_parameters(cav_ratio)
    if slope_low is None:
        slope_low = reward_peak / q_max
    reward = 0.0
    if flow < q_max / 2:
        reward = slope_low * flow
    elif q_max / 2 <= flow <= q_max - interval:
        reward = slope_low * flow + incentive
    else:
        reward = slope_low * flow + incentive_double
    return reward
