import hashlib
import numpy as np


def case_data(case):
    from cav_dl.data.synthetic import scene, MODES, quota15, weighted_quota
    seed, p = int(case['traffic_seed']), float(case['p'])
    data = scene(seed, .5, MODES[int(case['main'])], MODES[int(case['ramp'])])
    latent = data['init_cav_ratio'].copy()
    data['input_supply'] *= float(case['main_scale'])
    data['ramp_inflow'] *= float(case['ramp_scale'])
    data['input_cav_supply'], data['ramp_cav_ratio'] = quota15(
        data['input_supply'], data['ramp_inflow'], p, seed)
    data['init_cav_ratio'] = weighted_quota(data['init_car'], latent, p)
    h = hashlib.sha256()
    for key in ['init_car', 'init_cav_ratio', 'input_supply', 'input_cav_supply', 'ramp_inflow', 'ramp_cav_ratio']:
        h.update(np.asarray(data[key], dtype='<f8').tobytes())
    data.update(p=p, sha256=h.hexdigest(), offered_arrivals=float(data['input_supply'].sum() + data['ramp_inflow'].sum()))
    return data

