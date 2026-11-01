import hashlib
import numpy as np
from cav_dl.envs._legacy.environment import EpisodeInputGenerator
from .profiles import MODES, hourly_profile

def weighted_quota(weights, latent, target):
    
    if target == 0 or target == 1:
        return np.full(len(weights), target, dtype=float)
    lo, hi = -float(np.max(latent))-1., 1.-float(np.min(latent))+1.
    total = weights.sum()
    for _ in range(75):
        mid = (lo+hi)/2
        if weights @ np.clip(latent+mid, 0., 1.) < target*total:
            lo = mid
        else:
            hi = mid
    p = np.clip(latent+(lo+hi)/2, 0., 1.)
    assert abs(weights @ p-target*total) < 1e-8
    return p



def quota15(main,ramp,p,seed):
    weights=np.r_[main,ramp]
    if p in (0.,1.):return np.full(len(main),p),np.full(len(ramp),p)
    children=np.random.SeedSequence([int(seed),151019]).spawn(2)
    generators=[np.random.default_rng(c) for c in children]
    sizes=[len(main),len(ramp)]
    for attempt in range(64):
        masks=[];pure=[]
        for rng,n in zip(generators,sizes):
            masks.append(rng.random(n)<.2);pure.append((rng.random(n)<p).astype(float))
        mask=np.r_[tuple(masks)];values=np.r_[tuple(pure)]
        fixed=float(weights[mask]@values[mask]);free=float(weights[~mask].sum())
        remaining=p*weights.sum()-fixed
        if free>0 and 0<=remaining<=free:break
    else:
        mask=np.zeros(len(weights),bool);values=np.zeros(len(weights));remaining=p*weights.sum();free=weights.sum()
    latent=np.r_[tuple(rng.beta(.5,.5,n) for rng,n in zip(generators,sizes))]
    values[~mask]=weighted_quota(weights[~mask],latent[~mask],remaining/free)
    assert abs(weights@values-p*weights.sum())<1e-7
    return values[:len(main)],values[len(main):]

def scene(seed,p,main_mode,ramp_mode):
    saved=np.random.get_state()
    try:
        np.random.seed(seed)
        data=EpisodeInputGenerator().generate_test_reset_config(seed,10600,6500,[p])
    finally:np.random.set_state(saved)
    main=hourly_profile(seed,main_mode)[0]*15/3600
    ramp=hourly_profile(seed,ramp_mode)[1]*15/3600
    cm,cr=quota15(main,ramp,p,seed)
    data.update(input_supply=main,ramp_inflow=ramp,input_cav_supply=cm,ramp_cav_ratio=cr)
    data['init_cav_ratio']=weighted_quota(data['init_car'],data['init_cav_ratio'],p)
    h=hashlib.sha256()
    for field in ('init_car','init_cav_ratio','input_supply','input_cav_supply','ramp_inflow','ramp_cav_ratio'):
        h.update(np.asarray(data[field],dtype='<f8').tobytes())
    error=float(main@cm+ramp@cr-p*(main.sum()+ramp.sum()))
    assert abs(error)<1e-7
    data.update(seed=int(seed),p=float(p),main_mode=main_mode,ramp_mode=ramp_mode,sha256=h.hexdigest(),
        offered_arrivals=float(main.sum()+ramp.sum()),arrival_quota_error=error,cav_mode='quota_conditioned_random15')
    return data
