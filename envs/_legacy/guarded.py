from cav_dl.envs._legacy.kernel import *
@njit(cache=True)
def guarded_advance(history,main,mp,ramp,rp,ramp_pair,actual,cav_in,ramp_actual,
            t,count,lane_config,set_lanes,ramp_index,cumulative,initial_total,offered_total,terminated):
    early=False
    rejects=0
    acted=lane_config.copy()
    performed=0
    for _ in range(count):
        row=history[t]
        if set_lanes:
            row[:,A]=acted
        parameters(row)
        if set_lanes and np.any(row[20:100,EARLY]>0):
            rejects+=1
            n,p=row[:,N].copy(),row[:,P].copy()
            fallback=history[t-1,:,A].copy() if t>0 else np.zeros(120)
            row[:,:]=0.
            row[:,N],row[:,P],row[:,A]=n,p,fallback
            parameters(row)
            if np.any(row[20:100,EARLY]>0):
                fallback[:]=0.
                row[:,:]=0.
                row[:,N],row[:,P],row[:,A]=n,p,fallback
                parameters(row)
            acted=fallback.copy()
        flows(row,ramp_pair[t],ramp_index)
        rcell=row[ramp_index-1]
        if ramp[t]>rcell[RO] and t<len(main):
            remain=ramp[t]-rcell[RO]
            cv=ramp[t]*rp[t]-rcell[RC]+ramp[t+1]*rp[t+1]
            ramp[t+1]+=remain
            rp[t+1]=cv/ramp[t+1] if ramp[t+1]>0 else 0.
            ramp_pair[t+1,0],ramp_pair[t+1,1]=ramp[t+1],rp[t+1]
        cumulative[1]+=rcell[RO]
        ramp_actual[t]=rcell[RO] if ramp_index>0 else 0.
        cumulative[2]+=row[-1,O]
        
        
        offered_c=main[t]*mp[t]
        offered_h=main[t]*(1-mp[t])
        entered_c=min(offered_c,row[0,R]*mp[t])
        entered_h=min(offered_h,row[0,R]*(1-mp[t]))
        actual[t]=entered_c+entered_h
        cav_in[t]=entered_c
        cumulative[0]+=actual[t]
        if main[t]>actual[t] and t<len(main):
            remain=main[t]-actual[t]
            cv=remain*mp[t]+main[t+1]*mp[t+1]
            main[t+1]+=remain
            mp[t+1]=cv/main[t+1] if main[t+1]>0 else 0.
        
        for i in range(120):
            incoming=actual[t] if i==0 else row[i-1,O]
            incoming_cv=cav_in[t] if i==0 else row[i-1,OC]
            if i==ramp_index:
                incoming+=rcell[RO]
                incoming_cv+=rcell[RC]
            new_n=row[i,N]+incoming-row[i,O]
            new_cv=row[i,N]*row[i,P]+incoming_cv-row[i,OC]
            new_p=0. if new_n<1e-6 else new_cv/new_n
            history[t+1,i,N]=new_n
            history[t+1,i,P]=max(0.,new_p)
        
        early=False
        if cumulative[2]>=65000 or abs(initial_total+offered_total-cumulative[2])<10 or t>=3920:
            terminated=True
        performed+=1
        t+=1
        if early:
            break
    return t,performed,early,terminated,rejects
