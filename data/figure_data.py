import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .profiles import MODES, hourly_profile

SEEDS = [202610150, 202610151, 202610152]
ARMS = ['Full', 'NoDensity', 'NoCapacity', 'NoSafety', 'NoTime', 'NoEarlyPenalty']
SPATIAL_ARMS = ['Full', 'Rule', 'FS1', 'FS2', 'FS3', 'FS4', 'FS5']


class Builder:
    def __init__(self, project, records, output, training=None):
        self.project, self.records, self.output = map(Path, [project, records, output])
        self.training = Path(training) if training else self.records / 'training'
        self.output.mkdir(parents=True, exist_ok=True)
        self.sources, self.tables = {}, []

    def read(self, path):
        path = Path(path).resolve()
        key = str(path.relative_to(self.project)) if path.is_relative_to(self.project) else str(path)
        self.sources[key] = hashlib.sha256(path.read_bytes()).hexdigest()
        return pd.read_csv(path)

    def raw(self, series):
        q = self.read(self.records / series / 'raw.csv')
        q = q.drop_duplicates('job_id', keep='last')
        if q.physical_failure.any(): raise ValueError(f'Failed records in {series}')
        return q

    def details(self, series, name):
        path = self.records / series / f'{name}.csv'
        if path.exists(): return self.read(path)
        files = sorted((self.records / series / 'details').glob(f'*_{name}.csv'))
        if not files: raise FileNotFoundError(f'Missing {series}/{name} records')
        return pd.concat([self.read(path) for path in files], ignore_index=True)

    def write(self, frame, name):
        path = self.output / name
        frame.to_csv(path, index=False)
        self.tables.append(dict(file=name, rows=len(frame), columns=list(frame.columns),
                                sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(f'{name}: {len(frame)} rows', flush=True)

    def curve(self, group, variant, size, ddof, clamp=False):
        values, lengths = [], []
        for seed in SEEDS:
            path = self.training / group / f'{variant}_seed{seed}.csv'
            if not path.exists():
                path = self.training / group / f'{variant}_seed{seed}' / 'episodes.csv'
            q = self.read(path).dropna(subset=['steps', 'common_reward', 'controls'])
            bins = ((q.steps - 1) // size).astype(int)
            if clamp: bins = bins.clip(upper=1000000 // size - 1)
            q = q.groupby(bins)[['common_reward', 'controls']].mean().reindex(range(1000000 // size))
            values.append(q.common_reward.to_numpy())
            lengths.append(q.controls.to_numpy())
        return np.array(values), np.array(lengths)

    def figure02(self):
        frames = []
        for mode in MODES:
            main, ramp = hourly_profile(91000000, mode)
            frames.append(pd.DataFrame(dict(mode=mode, hours=np.arange(len(main)) / 240,
                                           main_veh_h=main, ramp_veh_h=ramp)))
        self.write(pd.concat(frames, ignore_index=True), 'fig02.csv')

    def figure03(self):
        rows = []
        for algorithm in ['PPO', 'DQN', 'SAC']:
            rewards, lengths = self.curve('algorithms', algorithm, 2000, 1)
            for metric, a in [('reward5', rewards), ('controls', lengths)]:
                mean, sd = np.nanmean(a, axis=0), np.nanstd(a, axis=0, ddof=1)
                rows.append(pd.DataFrame(dict(algorithm=algorithm, metric=metric,
                    steps=(np.arange(500) + .5) * 2000, mean=mean, seed_sd=sd, raw_mean=mean)))
        self.write(pd.concat(rows, ignore_index=True), 'fig03.csv')

    def figure04(self):
        curves = []
        for arm in ARMS:
            a, _ = self.curve('ablation', arm, 20000, 0, True)
            curves.append(pd.DataFrame(dict(arm=arm, steps=np.arange(1, 51) * 20000,
                mean=np.nanmean(a, axis=0), seed_sd=np.nanstd(a, axis=0))))
        self.write(pd.concat(curves, ignore_index=True), 'fig04_training.csv')
        q = self.raw('synthetic')
        q = q[q.arm.isin(ARMS) & q.compliance.eq(1)]
        ids = sorted(q.case_id.unique())
        ps = q[['case_id', 'p']].drop_duplicates().set_index('case_id').loc[ids].p.to_numpy()
        rng = np.random.default_rng(2026102301)
        draw = np.concatenate([rng.choice(np.flatnonzero(ps == p), size=(5000, int((ps == p).sum())), replace=True)
                               for p in sorted(set(ps))], axis=1)
        seeds = rng.integers(0, 3, (5000, 3))
        rows = []
        fields = [('J_fixed_control', 1), ('complete_tts', 1e5), ('mean_entry_wait_h', 1 / 60),
                  ('fixed_departures', 1000), ('full_high_occupancy_vehicle_hours', 1), ('executed_lane_switches', 1)]
        for metric, scale in fields:
            for arm in ARMS:
                a = q[q.arm.eq(arm)].pivot(index='training_seed', columns='case_id', values=metric).loc[SEEDS, ids].to_numpy() / scale
                if not np.isfinite(a).all(): raise ValueError(f'Incomplete ablation: {arm}, {metric}')
                lo, hi = np.quantile(a[seeds[:, :, None], draw[:, None, :]].mean((1, 2)), [.025, .975])
                rows.append(dict(metric=metric, arm=arm, mean=a.mean(), low=lo, high=hi, scale=scale))
        self.write(pd.DataFrame(rows), 'fig04_test.csv')

    def figure05(self):
        for name in ['fig05.csv', 'fig05_input.csv', 'fig05_lanes.csv']:
            self.write(self.read(self.records / 'figure05' / name), name)

    def figure06(self):
        q = self.details('synthetic', 'cells')
        q = q[q.p.between(.4, .9) & q.arm.isin(SPATIAL_ARMS) & q.compliance.eq(1)]
        out = q.groupby(['p', 'arm', 'kind', 'cell'])[['distance_km', 'vehicle_hours']].sum().reset_index()
        out['speed_kmh'] = np.divide(out.distance_km, out.vehicle_hours,
            out=np.full(len(out), np.nan), where=out.vehicle_hours > 1e-10)
        self.write(out, 'fig06.csv')

    def figure07(self):
        q = self.details('synthetic', 'units')
        q = q[q.p.between(.4, .9) & q.arm.isin(SPATIAL_ARMS) & q.compliance.eq(1)]
        out = q.groupby(['p', 'arm', 'unit'])[['density', 'speed']].mean().reset_index()
        self.write(out, 'fig07.csv')

    def figure08(self):
        q = self.raw('synthetic')
        q = q[q.p.between(.4, .9) & q.arm.isin(SPATIAL_ARMS) & q.compliance.eq(1)]
        fields = ['density_std_units', 'critical_density_RMSE']
        if q[fields].isna().any().any(): raise ValueError('Missing unit-level density records')
        self.write(q.groupby(['p', 'arm'])[fields].mean().reset_index(), 'fig08.csv')

    def figure09(self):
        q = self.raw('synthetic')
        arms = ['Full', 'Rule', *[f'FS{i}' for i in range(1, 10)]]
        q = q[q.p.between(.3, .9) & q.arm.isin(arms) & q.compliance.eq(1)]
        fields = ['source_road_fuel_kg', 'source_queue_fuel_kg', 'source_admission_fuel_kg',
                  'source_total_fuel_kg', 'total_vehicles', 'total_departures', 'vehicle_km', 'complete_tts']
        if q[fields].isna().any().any(): raise ValueError('Missing fuel records')
        cases = q.groupby(['p', 'case_id', 'arm'])[fields].mean()
        out = cases.groupby(['p', 'arm']).sum()
        out['n_cases'] = cases.groupby(['p', 'arm']).size()
        self.write(out.reset_index(), 'fig09.csv')

    def figure10(self):
        q = pd.concat([self.raw('synthetic'), self.raw('compliance')], ignore_index=True)
        q = q[q.arm.isin(['Full', 'Rule', 'FS1'])]
        rows = []
        for (p, compliance), group in q.groupby(['p', 'compliance']):
            m = group.groupby(['case_id', 'arm'])[['complete_tts', 'J_fixed_control']].mean().groupby('arm').mean()
            row = dict(p=p, compliance=compliance)
            for arm, label in [('Full', 'PPO'), ('Rule', 'Rule'), ('FS1', 'FS1')]:
                row[label + '_TTS'] = m.loc[arm, 'complete_tts']
                row[label + '_reward'] = m.loc[arm, 'J_fixed_control']
            row.update(PPO_vs_FS1_TTS_pct=100 * (1 - row['PPO_TTS'] / row['FS1_TTS']),
                       Rule_vs_FS1_TTS_pct=100 * (1 - row['Rule_TTS'] / row['FS1_TTS']),
                       PPO_vs_Rule_TTS_pct=100 * (1 - row['PPO_TTS'] / row['Rule_TTS']))
            rows.append(row)
        self.write(pd.DataFrame(rows), 'fig10.csv')

    def figure11(self):
        path = self.project / 'assets/pems/daily_flows.npz'
        self.sources[str(path.relative_to(self.project))] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows = []
        with np.load(path, allow_pickle=False) as archive:
            for inlet, label in enumerate(['Mainline', 'On-ramp']):
                for date in sorted(archive.files):
                    day = f'{date[:4]}-{date[4:6]}-{date[6:]}'
                    rows.append(pd.DataFrame(dict(date=day, time_h=np.arange(288) / 12,
                        inlet=label, flow_veh_h=archive[date][inlet] * 12)))
        self.write(pd.concat(rows, ignore_index=True), 'fig11.csv')

    def figure12(self):
        a, _ = self.curve('pems', 'PPO', 2000, 1)
        self.write(pd.DataFrame(dict(steps=(np.arange(500) + .5) * 2000,
            mean=np.nanmean(a, axis=0), seed_sd=np.nanstd(a, axis=0, ddof=1))), 'fig12_training.csv')
        q = self.raw('pems_penetration')
        v = q.groupby(['p', 'case_id', 'arm']).native_episode_reward.mean().unstack('arm')
        rng = np.random.default_rng(20261023)
        rows = []
        for p, group in v.groupby(level='p'):
            a = group[['PPO', 'FS1']].to_numpy()
            boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
            for j, arm in enumerate(['PPO', 'FS1']):
                lo, hi = np.quantile(boot[:, j], [.025, .975])
                rows.append(dict(p=p, arm=arm, mean=a[:, j].mean(), lower95=lo, upper95=hi, n=len(a)))
        self.write(pd.DataFrame(rows), 'fig12_penetration.csv')
        q = self.raw('pems_dates')
        v = q.groupby(['date', 'rep', 'arm']).native_episode_reward.mean().unstack('arm').reset_index()
        v['weekday'] = pd.to_datetime(v.date).dt.dayofweek
        rng = np.random.default_rng(20261023)
        rows = []
        for scope in ['date', 'weekday']:
            for key, group in v.groupby(scope):
                a = group[['PPO', 'FS1']].to_numpy()
                boot = a[rng.integers(0, len(a), size=(10000, len(a)))].mean(axis=1)
                for j, arm in enumerate(['PPO', 'FS1']):
                    lo, hi = np.quantile(boot[:, j], [.025, .975])
                    rows.append(dict(scope=scope, key=str(key), arm=arm, mean=a[:, j].mean(), lower95=lo, upper95=hi, n=len(a)))
        self.write(pd.DataFrame(rows), 'fig12_dates.csv')

    def figure13(self):
        q = self.details('pems_spatial', 'trace')
        self.write(q[['arm', 'time_h', 'density', 'cumulative_outflow']], 'fig13.csv')
        controls = self.details('pems_spatial', 'controls')
        self.write(controls[controls.arm.eq('PPO')][['time_h', 'unit', 'lanes']], 'fig13_lanes.csv')
        wide = q.pivot(index='time_h', columns='arm', values='cumulative_outflow')
        self.write(pd.DataFrame(dict(time_h=wide.index, outflow_difference=wide.PPO - wide.FS1)), 'fig13_difference.csv')

    def figure14(self):
        q = self.details('pems_spatial', 'states')
        self.write(q[['arm', 'time_h', 'cell', 'density', 'flow', 'speed']], 'fig14.csv')

    def figure15(self):
        q = pd.concat([self.raw('lane4'), self.raw('lane5')], ignore_index=True)
        fields = ['native_episode_reward', 'complete_tts', 'mean_entry_wait_h', 'fixed_departures']
        q = q.groupby(['case_id', 'lanes', 'arm'])[fields].mean().groupby(['lanes', 'arm']).mean().reset_index()
        self.write(q, 'fig15.csv')

    def build(self, figures, metadata=False):
        for figure in sorted(figures):
            getattr(self, f'figure{figure:02d}')()
        manifest = dict(figures=sorted(figures), sources_sha256=self.sources, tables=self.tables)
        if metadata:
            (self.output / 'generated_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        return manifest
