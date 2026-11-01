import numpy as np

MODES=('D1_fixed_peak','D2_random_peak','D3_multi_peak','D4_random_steps','D5_correlated','D6_independent')

def envelope(hours, start, duration, rise, fall=None):
    fall = rise if fall is None else fall
    x = np.zeros(len(hours))
    up = (hours >= start-rise) & (hours < start)
    x[up] = .5*(1-np.cos(np.pi*(hours[up]-start+rise)/rise))
    x[(hours >= start) & (hours < start+duration)] = 1.
    down = (hours >= start+duration) & (hours < start+duration+fall)
    x[down] = .5*(1+np.cos(np.pi*(hours[down]-start-duration)/fall))
    return x

def hourly_profile(seed, mode, n=2520):
    
    streams = np.random.SeedSequence([seed, 6182026, MODES.index(mode)]).spawn(2)
    hours = np.arange(n)*15/3600
    result = []
    for inlet, (low, high, start, jitter) in enumerate(((6500.,10600.,1.5,50.), (300.,700.,2.5,5.))):
        rng = np.random.default_rng(streams[inlet])
        base = low+(high-low)*envelope(hours,start,2.,1.)
        if mode == MODES[0]:
            q = base+rng.uniform(-jitter,jitter,n)
        elif mode == MODES[1]:
            b = low*rng.uniform(.8,1.2)
            peak = max(1.05*b, high*rng.uniform(.8,1.2))
            s = rng.uniform(1.5, 3.5 if inlet == 0 else 4.5)
            duration, rise, fall = rng.uniform(1.,3.), rng.uniform(.25,1.5), rng.uniform(.25,1.5)
            q = b+(peak-b)*envelope(hours,s,duration,rise,fall)+rng.uniform(-jitter,jitter,n)
        elif mode == MODES[2]:
            count = int(rng.integers(2,4))
            centers = (np.arange(count)+.5)*10.5/count+rng.uniform(-.3,.3,count)
            q = np.full(n,low)
            for center in centers:
                duration, trans = rng.uniform(.2,.8), rng.uniform(.15,.45)
                peak = max(low, high*rng.uniform(.8,1.2))
                q = np.maximum(q,low+(peak-low)*envelope(hours,center-duration/2,duration,trans))
            q += rng.uniform(-jitter,jitter,n)
        elif mode == MODES[3]:
            q = np.empty(n)
            pos = 0
            while pos < n:
                end = min(n,pos+int(rng.integers(5,31))*4)
                q[pos:end] = rng.uniform(.75*low,1.2*high)
                pos = end
        elif mode == MODES[4]:
            rho = np.exp(-15/300)
            sigma = .25*(high-low)
            noise = np.empty(n)
            noise[0] = rng.normal(0,sigma)
            for t in range(1,n):
                noise[t] = rho*noise[t-1]+sigma*np.sqrt(1-rho*rho)*rng.normal()
            q = np.clip(base+noise,.75*low,1.2*high)
        else:
            q = rng.uniform(.75*low,1.2*high,n)
        result.append(np.maximum(q,0.))
    return result
