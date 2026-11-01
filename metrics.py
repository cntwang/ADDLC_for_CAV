

import numpy as np

from cav_dl.envs._legacy import kernel as k


def physical_metrics(history, lanes, main_queue, ramp_queue, admitted):
    dt = 15 / 3600
    stock = history[:, :, k.N]
    active = history[:, :, k.A]
    p = np.where(active > 0, history[:, :, k.GP], history[:, :, k.P])
    jam = 1 / (p * (7 / 1000) + (1 - p) * (8.53 / 1000))
    gl = np.where(active > 0, history[:, :, k.GC], stock)
    occupancy = gl / (jam / 3 * (lanes - active))
    exposure = (stock * (occupancy > .5)).sum() * dt
    distance = history[:, :, k.O].sum() / 3
    
    if len(history) % 4:
        raise ValueError("Fuel aggregation requires complete 60-second blocks")
    block_stock = stock.reshape(-1, 4, 120).sum(axis=1)
    block_flow = history[:, :, k.O].reshape(-1, 4, 120).sum(axis=1)
    speed = np.divide(block_flow / 3, block_stock * dt,
                      out=np.zeros_like(block_stock), where=block_stock > 1e-10)
    rate = 156 + 3.54 * speed - .0388 * speed**2 + .000776 * speed**3
    road_g = (block_stock * dt * rate).sum()
    queue_g = 156 * (main_queue + ramp_queue).sum() * dt
    
    admission_g = 156 * admitted.sum() * dt
    return dict(
        full_high_occupancy_vehicle_hours=float(exposure),
        vehicle_km=float(distance), source_road_fuel_kg=float(road_g / 1000),
        source_queue_fuel_kg=float(queue_g / 1000),
        source_admission_fuel_kg=float(admission_g / 1000),
        source_total_fuel_kg=float((road_g + queue_g + admission_g) / 1000),
    )
