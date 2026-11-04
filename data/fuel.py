"""Equal-count demands with controlled total load and ramp share."""
import hashlib
import numpy as np
from cav_dl.data.synthetic import scene, quota15, weighted_quota, MODES

FIELDS = ['init_car', 'init_cav_ratio', 'input_supply', 'input_cav_supply',
          'ramp_inflow', 'ramp_cav_ratio']


def case_data(case):
    seed = int(case['traffic_seed'])
    p = float(case['p'])
    mode = MODES[int(case['family'])]
    data = scene(seed, .5, mode, mode)
    main_total = float(case['arrivals']) * (1 - float(case['ramp_share']))
    ramp_total = float(case['arrivals']) * float(case['ramp_share'])
    for key, total in [('input_supply', main_total), ('ramp_inflow', ramp_total)]:
        values = data[key] / data[key].sum() * total
        values[-1] += total - values.sum()
        assert abs(values.sum() - total) < 1e-8 and (values >= 0).all()
        data[key] = values
    data['input_cav_supply'], data['ramp_cav_ratio'] = quota15(
        data['input_supply'], data['ramp_inflow'], p, seed)
    data['init_cav_ratio'] = weighted_quota(data['init_car'], data['init_cav_ratio'], p)
    digest = hashlib.sha256()
    for key in FIELDS:
        digest.update(np.asarray(data[key], dtype='<f8').tobytes())
    data.update(p=p, sha256=digest.hexdigest(),
                offered_arrivals=float(data['input_supply'].sum() + data['ramp_inflow'].sum()),
                arrival_quota_error=float(data['input_supply'] @ data['input_cav_supply']
                    + data['ramp_inflow'] @ data['ramp_cav_ratio'] - p * case['arrivals']))
    return data
