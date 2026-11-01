from cav_dl.envs._legacy.kernel import *
from cav_dl.envs.kernels.lane3 import parameters_ulp
from cav_dl.envs.stay_helpers import mixed_nc
from cav_dl.envs.stay_helpers import HD,HDO,HDNEXT,STRANDED,RELEASED,retained,flows_stay
@njit(cache=True)
def parameters_stay(row):
    if not np.any(row[:,A]>0):
        parameters_ulp(row)
        return
    for i in range(120):
        x=row[i];n=x[A]
        x[EARLY]=0.
        if n==0:
            jam,q,w,crit,drop=fd(x[P])
            x[CAP]=jam;x[RHO]=x[N]
            x[S],x[R]=mixed_nc(q,jam,crit,x[RHO],x[N],w,drop,3.)
            continue
        c=x[N]*x[P];h=x[N]-c
        hd=x[HD]
        if hd < -1e-8 or hd > h+1e-8: raise ValueError("retained HV inventory outside class total")
        hd=min(h,max(0.,hd))
        
        cd=min(c,max(0.,fd(1.)[3]*n/3.-hd))
        for _ in range(12):
            dp=cd/(cd+hd) if cd+hd>0 else 1.
            cd=min(c,max(0.,fd(dp)[3]*n/3.-hd))
        chosen=False
        
        for j in range(int(np.ceil(c-cd))+1):
            dc=min(c,cd+j);total=dc+hd
            dp=dc/total if total>0 else 1.
            gc=x[N]-total;gp=(c-dc)/gc if gc>1e-12 else 0.
            if total<=fd(dp)[0]*n/3.+1e-10 and gc<=fd(gp)[0]*(3-n)/3.+1e-10:
                x[DC],x[GC],x[GP]=total,gc,gp
                chosen=True
                break
        if not chosen:
            x[EARLY]=1.
            continue
        dj,dq,dw,_,_=fd(dp);gj,gq,gw,gcrit,gdrop=fd(gp)
        x[DD]=total/(n/3.);x[GD]=gc/((3-n)/3.)
        x[CAP]=dj*n/3.+gj*(3-n)/3.;x[RHO]=x[N]
        x[DS]=max(0.,min(dq*15/3600*n,total))
        x[DR]=max(0.,min(dq,dw*(dj-x[DD])))*15/3600*n
        x[GS],x[GR]=mixed_nc(gq,gj,gcrit,x[GD],gc,gw,gdrop,3-n)
        x[S],x[R]=x[DS]+x[GS],x[DR]+x[GR]

@njit(cache=True)
def advance_stay(history,main,mp,ramp,rp,ramp_pair,actual,cav_in,ramp_actual,
            t,count,lane_config,set_lanes,ramp_index,cumulative,initial_total,offered_total,terminated):
    early=False
    rejects=0
    acted=lane_config.copy()
    performed=0
    for _ in range(count):
        row=history[t]
        previous=history[t-1,:,A].copy() if t>0 else np.zeros(120)
        before_h=row[:,HD].copy()
        if set_lanes:
            row[:,A]=acted
        for i in range(120):
            row[i,HD],row[i,STRANDED],row[i,RELEASED]=retained(before_h[i],previous[i],row[i,A],row[i,N],row[i,P],0.75)
        parameters_stay(row)
        if set_lanes and np.any(row[20:100,EARLY]>0):
            rejects+=1
            n,p=row[:,N].copy(),row[:,P].copy()
            fallback=history[t-1,:,A].copy() if t>0 else np.zeros(120)
            row[:,:]=0.
            row[:,N],row[:,P],row[:,A]=n,p,fallback
            for i in range(120):
                row[i,HD],row[i,STRANDED],row[i,RELEASED]=retained(before_h[i],previous[i],row[i,A],row[i,N],row[i,P],0.75)
            parameters_stay(row)
            if np.any(row[20:100,EARLY]>0):
                fallback[:]=0.
                row[:,:]=0.
                row[:,N],row[:,P],row[:,A]=n,p,fallback
                for i in range(120):
                    row[i,HD],row[i,STRANDED],row[i,RELEASED]=retained(before_h[i],previous[i],row[i,A],row[i,N],row[i,P],0.75)
                parameters_stay(row)
            acted=fallback.copy()
        flows_stay(row,ramp_pair[t],ramp_index,0.75)
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
            if new_p < -1e-8 or new_p > 1.+1e-8: raise ValueError("CTM cell CAV fraction outside [0,1]")
            new_p=min(1.,max(0.,new_p))
            incoming_hd=0. if i==0 else row[i-1,HDNEXT]
            next_hd=row[i,HD]-row[i,HDO]+incoming_hd
            if next_hd < -1e-8 or next_hd > new_n-new_cv+1e-8: raise ValueError("persistent HV cohort conservation failure")
            history[t+1,i,HD]=max(0.,next_hd)
            history[t+1,i,N]=new_n
            history[t+1,i,P]=max(0.,new_p)
        
        early=False
        if False:
            terminated=True
        performed+=1
        t+=1
        if early:
            break
    return t,performed,early,terminated,rejects
