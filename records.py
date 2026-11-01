from pathlib import Path
import numpy as np
import pandas as pd
from cav_dl.envs._legacy import kernel as k
from cav_dl.utils.io import export_table


def critical(p):
    d = p * (7 / 1000) + (1 - p) * (8.53 / 1000)
    tau = p * (.6 / 3600) + (1 - p) * (1.61 / 3600)
    return 1 / (tau * 80 + d)


def spatial(history, cutoff, lanes):
    z = history[120:cutoff]
    active = z[:, :, k.A] > 0
    stock = np.stack([np.where(active, z[:, :, k.DC], 0),
                      np.where(active, z[:, :, k.GC], z[:, :, k.N]),
                      z[:, :, k.N] * z[:, :, k.P]])
    flow = np.stack([np.where(active, z[:, :, k.DO], 0),
                     np.where(active, z[:, :, k.GO], z[:, :, k.O]), z[:, :, k.OC]])
    hours = stock.sum(axis=1) * 15 / 3600
    distance = flow.sum(axis=1) / 3
    kind, cell = np.indices(hours.shape)
    cells = pd.DataFrame(dict(kind=np.array(['DL', 'GL', 'CAV'])[kind.ravel()],
                              cell=cell.ravel() + 1, distance_km=distance.ravel(),
                              vehicle_hours=hours.ravel()))
    cells['speed_kmh'] = np.divide(cells.distance_km, cells.vehicle_hours,
                                   out=np.full(len(cells), np.nan), where=cells.vehicle_hours > 1e-10)
    rho = z[:, :, k.RHO].mean(0).reshape(6, 20).mean(1)
    count = z[:, :, k.A]
    dp = np.divide(z[:, :, k.N] * z[:, :, k.P] - z[:, :, k.GC] * z[:, :, k.GP],
                   z[:, :, k.DC], out=np.ones_like(count), where=z[:, :, k.DC] > 1e-12)
    dp = np.clip(dp, 0, 1)
    kc = np.where(active, (count * critical(dp) + (lanes - count) * critical(z[:, :, k.GP])) / lanes,
                  critical(z[:, :, k.P])).mean(0).reshape(6, 20).mean(1)
    vehicle_time = z[:, :, k.N].sum(0).reshape(6, 20).sum(1) * 15 / 3600
    travelled = z[:, :, k.O].sum(0).reshape(6, 20).sum(1) / 3
    speed = np.divide(travelled, vehicle_time, out=np.full(6, np.nan), where=vehicle_time > 1e-12)
    units = pd.DataFrame(dict(unit=np.arange(6), density=rho, critical_density=kc, speed=speed))
    metrics = dict(density_std_units=float(np.sqrt(np.mean((rho - rho.mean())**2))),
                   critical_density_RMSE=float(np.sqrt(np.mean((rho - kc)**2))))
    return cells, units, metrics


class Recorder:
    def __init__(self, folder, stem, metadata, mode, clock_offset=0.):
        self.folder, self.stem = Path(folder), stem
        self.metadata, self.mode, self.clock_offset = metadata, mode, clock_offset
        self.metrics = {}

    def write(self, name, frame):
        self.folder.mkdir(parents=True, exist_ok=True)
        metadata = {('road_lanes' if key == 'lanes' else key): value
                    for key, value in self.metadata.items()}
        frame = frame.assign(**metadata)
        export_table(frame, self.folder / f'{self.stem}_{name}.csv', omit=('scenario_sha256',))

    def __call__(self, env, data):
        history = env._history[:env.t]
        cutoff = len(data['input_supply'])
        if self.mode == 'spatial':
            cells, units, self.metrics = spatial(history, cutoff, env.lane_num)
            self.write('cells', cells)
            self.write('units', units)
        elif self.mode == 'pems':
            z = history[120:cutoff]
            stock, flow = z[:, :, k.N], z[:, :, k.O]
            speed = np.divide(flow / 3, stock * 15 / 3600,
                              out=np.full_like(stock, 80.), where=stock > 1e-9)
            time = np.arange(120, cutoff, 4) * 15 / 3600 + self.clock_offset
            i, cell = np.indices(z[::4].shape[:2])
            self.write('states', pd.DataFrame(dict(time_h=time[i.ravel()], cell=cell.ravel() + 1,
                density=z[::4, :, k.RHO].astype(np.float32).ravel(),
                flow=(flow[::4] * 240).astype(np.float32).ravel(),
                speed=speed[::4].astype(np.float32).ravel())))
            self.write('trace', pd.DataFrame(dict(time_h=time,
                density=z[::4, 59, k.RHO], cumulative_outflow=np.cumsum(history[:cutoff, -1, k.O])[120::4])))
            controls = z[::60][:, [20, 40, 60, 80], k.A]
            t, unit = np.indices(controls.shape)
            self.write('controls', pd.DataFrame(dict(time_h=120 * 15 / 3600 + self.clock_offset + t.ravel() * .25,
                unit=unit.ravel() + 1, lanes=controls.ravel())))
