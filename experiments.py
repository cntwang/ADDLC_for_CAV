import argparse
import hashlib
import json
import multiprocessing as mp
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from cav_dl.utils.io import append_record, export_table

SEEDS = [202610150, 202610151, 202610152]
ARMS = ['Full', 'NoDensity', 'NoCapacity', 'NoSafety', 'NoTime', 'NoEarlyPenalty']
KEY = ['case_id', 'arm', 'training_seed', 'lanes', 'compliance']
CACHE = {}
PROJECT = None
OUTPUT = None


def read_cases(project, filename, limit=None, daily=False):
    from cav_dl.data.cases import read_manifest
    table = read_manifest(project / 'configs' / filename)
    if limit:
        table = table.groupby('date' if daily else 'p', sort=False).head(limit)
    return [{key: value for key, value in row.items() if pd.notna(value)}
            for row in table.to_dict('records')]


def plan(project, figures, limit=None):
    tasks = {}
    def add(case, series, arm, seed=-1, lanes=3, compliance=1., native=False, mode=''):
        case_id = int(case['case_id'])
        key = (series, case_id, arm, seed, lanes, compliance)
        if key in tasks:
            if mode: tasks[key]['mode'] = mode
            return
        stem = f'case_{case_id:05d}_{arm}_seed{seed}_c{round(compliance * 100):03d}'
        tasks[key] = dict(case=case, series=series, case_id=case_id, arm=arm,
                         training_seed=seed, lanes=lanes, compliance=compliance,
                         native=native, mode=mode, job_id=stem)
    cases = read_cases(project, 'test_cases_3lane.csv', limit)
    for case in cases:
        p = round(float(case['p']), 8)
        spatial = bool(figures & {6, 7, 8}) and .4 <= p <= .9
        if 4 in figures:
            for arm in ARMS:
                for seed in SEEDS:
                    add(case, 'synthetic', arm, seed, mode='spatial' if spatial and arm == 'Full' else '')
        if spatial:
            for seed in SEEDS: add(case, 'synthetic', 'Full', seed, mode='spatial')
            for arm in ['Rule', 'FS1', 'FS2', 'FS3', 'FS4', 'FS5']:
                add(case, 'synthetic', arm, mode='spatial')
        if 9 in figures and .3 <= p <= .9:
            for seed in SEEDS: add(case, 'synthetic', 'Full', seed)
            for arm in ['Rule', *[f'FS{i}' for i in range(1, 10)]]:
                add(case, 'synthetic', arm)
        if 10 in figures:
            for compliance in [1., .9, .8, .7, .6, .5]:
                series = 'synthetic' if compliance == 1 else 'compliance'
                for seed in SEEDS: add(case, series, 'Full', seed, compliance=compliance)
                for arm in ['Rule', 'FS1']: add(case, series, arm, compliance=compliance)
    if 12 in figures:
        for series, filename, daily in [('pems_penetration', 'test_cases_pems.csv', False),
                                         ('pems_dates', 'pems_dates.csv', True)]:
            for case in read_cases(project, filename, limit, daily):
                for seed in SEEDS: add(case, series, 'PPO', seed, native=True)
                add(case, series, 'FS1', native=True)
    if figures & {13, 14}:
        for case in read_cases(project, 'pems_spatial.csv'):
            add(case, 'pems_spatial', 'PPO', SEEDS[0], mode='pems')
            add(case, 'pems_spatial', 'FS1', mode='pems')
    if 15 in figures:
        for lanes in [4, 5]:
            for case in read_cases(project, f'test_cases_{lanes}lane.csv', limit):
                for arm in ['Transfer', 'Retrained']:
                    for seed in SEEDS: add(case, f'lane{lanes}', arm, seed, lanes, native=True)
                add(case, f'lane{lanes}', 'FS1', lanes=lanes, native=True)
    return list(tasks.values())


def initialize(project, output):
    global PROJECT, OUTPUT
    import torch
    torch.set_num_threads(1)
    PROJECT, OUTPUT = Path(project), Path(output)


def model(task):
    from cav_dl.agents.policy import load_policy
    arm, seed, lanes = task['arm'], task['training_seed'], task['lanes']
    if arm in ['Rule', *[f'FS{i}' for i in range(1, 10)]]:
        algorithm, directory = arm, None
    elif arm in ARMS:
        algorithm = 'ppo'
        directory = PROJECT / f'models/synthetic/{arm}/PPO_seed{seed}'
    else:
        algorithm = 'ppo'
        width = lanes if arm == 'Retrained' else 3
        directory = PROJECT / f'models/pems{width}/PPO_seed{seed}'
    key = (arm, seed, lanes)
    if key not in CACHE:
        CACHE[key] = load_policy(algorithm, directory, lanes, transfer=arm == 'Transfer')
    return CACHE[key]


def execute(task):
    from cav_dl.data.synthetic import scene, MODES
    from cav_dl.data.pems import PeMSDemand
    from cav_dl.evaluation import native_episode, clearance_episode
    from cav_dl.records import Recorder
    case = task['case']
    metadata = {key: task[key] for key in ['series', 'job_id', *KEY]}
    metadata.update(p=float(case['p']), date=case.get('date'), rep=case.get('rep'),
                    traffic_seed=int(case['traffic_seed']),
                    scenario_sha256=case['scenario_sha256'])
    try:
        if 'date' in case:
            demand = PeMSDemand(PROJECT / 'assets/pems', task['lanes'], scale=case['boundary_scale'])
            data = demand.scene(metadata['traffic_seed'], case['p'], case['date'],
                                composition_scale=case.get('composition_scale'))
        else:
            data = scene(metadata['traffic_seed'], case['p'], MODES[int(case['main'])], MODES[int(case['ramp'])])
        if data['sha256'] != case['scenario_sha256']: raise ValueError('Scenario fingerprint mismatch')
        policy = model(task)
        result = {}
        if task['native']:
            native = native_episode(data, policy, task['lanes'])
            result.update(native_episode_reward=native['J_episode'], native_reason=native['reason'])
        if task['series'] not in ['pems_penetration', 'pems_dates']:
            recorder = Recorder(OUTPUT / task['series'] / 'details', task['job_id'], metadata,
                                task['mode'], clock_offset=3.) if task['mode'] else None
            result.update(clearance_episode(data, policy, task['lanes'],
                                            compliance=task['compliance'], recorder=recorder))
        return dict(**metadata, **result, physical_failure=False)
    except Exception as error:
        return dict(**metadata, physical_failure=True, success=False, error=repr(error))


def run(project, output, figures, workers=4, limit=None, resume=False):
    project, output = Path(project).resolve(), Path(output).resolve()
    tasks = plan(project, figures, limit)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(figures=sorted(figures), case_limit=limit, tasks=tasks,
                   models=json.loads((project / 'models/index.json').read_text()))
    for record in payload['models']:
        for name, expected in record['sha256'].items():
            actual = hashlib.sha256((project / record['path'] / name).read_bytes()).hexdigest()
            if actual != expected: raise ValueError(f'Model fingerprint mismatch: {record["path"]}/{name}')
    payload['code_sha256'] = {str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((project / 'src').rglob('*.py'))}
    if 5 in figures:
        payload['figure05'] = json.loads((project / 'examples/figure05/config.json').read_text())
        payload['code_sha256'].update({str(path.relative_to(project)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((project / 'examples/figure05').rglob('*.py'))})
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    protocol = output / 'run.json'
    if protocol.exists():
        if not resume: raise FileExistsError('Use another output directory or --resume')
        if json.loads(protocol.read_text())['sha256'] != digest:
            raise ValueError('Existing run has different cases or policies')
    else:
        protocol.write_text(json.dumps(dict(sha256=digest, figures=sorted(figures),
            case_limit=limit, expected=len(tasks), models=payload['models'],
            code_sha256=payload['code_sha256']), indent=2), encoding='utf-8')
        rows = [{**task['case'], **{key: value for key, value in task.items() if key not in ['case', 'case_id']},
                 'case_id': task['case_id']} for task in tasks]
        export_table(pd.DataFrame(rows), output / 'plan.csv', omit=('scenario_sha256',))
    seen = set()
    for series in sorted({task['series'] for task in tasks}):
        path = output / series / 'raw.csv'
        if path.exists():
            raw = pd.read_csv(path).drop_duplicates('job_id', keep='last')
            for row in raw[~raw.physical_failure].itertuples(): seen.add((series, row.job_id))
    pending = [task for task in tasks if (task['series'], task['job_id']) not in seen]
    completed, failures = len(tasks) - len(pending), 0
    if pending:
        initialize(project, output)
    pool = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
        initializer=initialize, initargs=(str(project), str(output))) if workers > 1 and pending else None
    iterator = pool.map(execute, pending) if pool else map(execute, pending)
    try:
        for row in iterator:
            folder = output / row['series']
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / 'raw.csv'
            append_record(row, path, omit=('scenario_sha256',))
            completed += 1
            failures += int(row['physical_failure'])
            (output / 'progress.json').write_text(json.dumps(dict(completed=completed,
                expected=len(tasks), failures=failures)), encoding='utf-8')
            if completed % 25 == 0 or completed == len(tasks):
                print(f'{completed}/{len(tasks)}; failures={failures}', flush=True)
    finally:
        if pool: pool.shutdown()
    if failures: raise RuntimeError(f'{failures} runs failed; see raw.csv')
    (output / 'complete.json').write_text(json.dumps(dict(completed=completed,
        expected=len(tasks), failures=failures, case_limit=limit)), encoding='utf-8')
    return tasks


def main(project=None):
    project = Path(project or Path.cwd()).resolve()
    parser = argparse.ArgumentParser(description='Replay saved scenarios and export plotting CSVs.')
    parser.add_argument('figures', nargs='*', default=['all'])
    parser.add_argument('--project', type=Path, default=project)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--from-records', type=Path)
    parser.add_argument('--training', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--case-limit', type=int)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    project = args.project.resolve()
    figures = set(range(2, 16)) if not args.figures or 'all' in args.figures else set(map(int, args.figures))
    if not figures.issubset(set(range(2, 16))):
        parser.error('Figures must be from 2 to 15')
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.case_limit is not None and args.case_limit < 1:
        parser.error('--case-limit must be positive')
    output = args.output or project / 'results'
    records = args.from_records
    if records is None:
        run(project, output, figures, args.workers, args.case_limit, args.resume)
        records = output
        if 5 in figures:
            subprocess.run([sys.executable, str(project / 'examples/figure05/generate.py'),
                            '--output', str(records / 'figure05')], check=True)
    from cav_dl.data.figure_data import Builder
    Builder(project, records, args.data or project / 'generated',
            args.training or project / 'records/training').build(figures)


if __name__ == '__main__':
    main()
