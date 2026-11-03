from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from cav_dl.agents.policy import load_policy
from cav_dl.data.synthetic import scene, weighted_quota
from cav_dl.evaluation import clearance_episode
from cav_dl.records import critical
from cav_dl.envs._legacy import kernel as k



def build_scene(config):
    seed, p = int(config['traffic_seed']), float(config['p'])
    data = scene(seed, p, config['main_mode'], config['ramp_mode'])
    if config['composition'] == 'quota15':
        return data
    if config['composition'] != 'block_random15':
        raise ValueError('Unknown illustrative composition protocol')
    minutes = float(config['composition_block_minutes'])
    length = int(minutes * 60 / 15)
    if length < 1 or not np.isclose(length * 15, minutes * 60):
        raise ValueError('Composition blocks must contain whole CTM steps')
    main, ramp = data['input_supply'], data['ramp_inflow']
    starts = np.arange(0, len(main), length)
    ends = np.minimum(starts + length, len(main))
    weights = [np.array([q[a:b].sum() for a, b in zip(starts, ends)])
               for q in [main, ramp]]
    rng = np.random.default_rng(np.random.SeedSequence([
        seed, 4026, int(config['composition_stream'])]))
    targets = weighted_quota(np.r_[weights[0], weights[1]],
                             rng.beta(.35, .35, 2 * len(starts)), p).reshape(2, -1)
    for inlet, (name, q) in enumerate([('input_cav_supply', main), ('ramp_cav_ratio', ramp)]):
        values = np.empty(len(q))
        for j, (a, b) in enumerate(zip(starts, ends)):
            values[a:b] = weighted_quota(q[a:b], rng.beta(.5, .5, b - a), targets[inlet, j])
        data[name] = values
    digest = hashlib.sha256()
    for name in ['init_car', 'init_cav_ratio', 'input_supply', 'input_cav_supply',
                 'ramp_inflow', 'ramp_cav_ratio']:
        digest.update(np.asarray(data[name], dtype='<f8').tobytes())
    data['sha256'] = digest.hexdigest()
    data['cav_mode'] = 'block_conditioned_random15'
    data['arrival_quota_error'] = float(main @ data['input_cav_supply']
        + ramp @ data['ramp_cav_ratio'] - p * (main.sum() + ramp.sum()))
    if abs(data['arrival_quota_error']) > 1e-7:
        raise ValueError('Whole-demand CAV quota failed')
    return data


class History:
    def __call__(self,env,data):
        self.z=env._history[:len(data['input_supply'])].copy()
        self.metrics={}


def generate(project, output):
    Q=Path(project)
    out=Path(output)
    import torch
    torch.set_num_threads(1)
    config=json.loads((Q/'examples/figure05/config.json').read_text(encoding='utf-8'))
    seed=int(config['traffic_seed'])
    p=float(config['p'])
    training_seed=int(config['training_seed'])
    data=build_scene(config)
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(dict(time_h=np.arange(2520)/240,main_CAV_share=data['input_cav_supply'],
                      ramp_CAV_share=data['ramp_cav_ratio'],main_vehicles=data['input_supply'],
                      ramp_vehicles=data['ramp_inflow'])).to_csv(out/'fig05_input.csv',index=False)
    ends=np.arange(60,2521,60)
    result={'time_h':ends/240}
    for inlet,q,c in [('main',data['input_supply'],data['input_cav_supply']),('ramp',data['ramp_inflow'],data['ramp_cav_ratio'])]:
        result[inlet+'_CAV_share']=[np.sum(q[j-60:j]*c[j-60:j])/np.sum(q[j-60:j]) for j in ends]
    rows=[]
    for arm,algorithm,model in [('Full','ppo',Q/f'models/synthetic/Full/PPO_seed{training_seed}'),('FS1','FS1',None)]:
        rec=History();policy=load_policy(algorithm,model)
        metrics=clearance_episode(data,policy,recorder=rec)
        rows.append(dict(arm=arm,traffic_seed=seed,p=p,scenario_sha256=data['sha256'],**metrics))
        z=rec.z; region=z[:,20:100]
        stock=region[:,:,k.N].sum(1)
        cv=(region[:,:,k.N]*region[:,:,k.P]).sum(1)
        flux=region[:,:,k.O].sum(1)
        n=region[:,:,k.A]
        kc=np.where(n>0,(n*critical(1.)+(3-n)*critical(region[:,:,k.GP]))/3,critical(region[:,:,k.P])).mean(1)
        result[arm+'_CAV_share']=[cv[j-60:j].sum()/stock[j-60:j].sum() for j in ends]
        result[arm+'_density']=[stock[j-60:j].mean()/80 for j in ends]
        result[arm+'_critical_density']=[kc[j-60:j].mean() for j in ends]
        result[arm+'_speed_kmh']=[flux[j-60:j].sum()/3/(stock[j-60:j].sum()/240) for j in ends]
        result[arm+'_cumulative_outflow']=np.cumsum(z[:,-1,k.O])[ends-1]
        if arm=='Full':
            lanes=z[:,[20,40,60,80],k.A]
            t,u=np.indices(lanes.shape)
            pd.DataFrame(dict(time_h=t.ravel()/240,unit=u.ravel()+1,lanes=lanes.ravel())).to_csv(out/'fig05_lanes.csv',index=False)
    result['Full_minus_FS1_outflow']=result['Full_cumulative_outflow']-result['FS1_cumulative_outflow']
    pd.DataFrame(result).to_csv(out/'fig05.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    print(pd.DataFrame(rows)[['arm','J_fixed_control','complete_tts','fixed_departures']].to_string(index=False))


