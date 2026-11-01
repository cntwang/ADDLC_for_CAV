from cav_dl.envs._legacy.kernel import *
@njit(cache=True)
def mixed_ulp(q,jam,crit,rho,cars,wave,drop,lanes):
    q_jam=drop*.4
    scale=(drop-q_jam)/(jam-crit)
    supply=max(0.,min(q*15/3600*lanes,cars)) if rho<=np.nextafter(crit,np.inf) else max(0.,min(cars,(drop-scale*(rho-crit))*15/3600*lanes))
    receive=max(0.,min(q,wave*(jam-rho))*15/3600*lanes)
    return supply,receive

@njit(cache=True)
def parameters_ulp(row):
    jam_c,q_c,w_c,crit_c,_=fd(1.)
    for i in range(120):
        x=row[i]
        n=x[A]
        if n==0:
            jam,q,w,crit,drop=fd(x[P])
            x[CAP]=jam*(1/3)*4
            x[RHO]=x[N]/((1/3)*4)
            x[S],x[R]=mixed_ulp(q,jam,crit,x[RHO],x[N],w,drop,4)
        else:
            x[EARLY]=1.
            gl_n=4-n
            cav=x[N]*x[P]
            hv=x[N]*(1-x[P])
            dc=min(cav,crit_c*n*(1/3))
            gl_cav=cav-dc
            gc=gl_cav+hv
            gp=gl_cav/gc if gc>0 else 0.
            gj,_,_,_,_=fd(gp)
            max_gc=gj*(1/3)*gl_n
            if gc>max_gc:
                if gl_cav>0:
                    end=min(cav,jam_c*n*(1/3))
                    
                    for trial in np.arange(dc,end,1.):
                        gl_cav=cav-trial
                        gc=gl_cav+hv
                        gp=gl_cav/gc if gc>0 else 0.
                        gj,_,_,_,_=fd(gp)
                        if gc<=gj*(1/3)*gl_n:
                            x[DC],x[GC],x[GP]=trial,gc,gp
                            x[EARLY]=0.
                            break
            else:
                x[DC],x[GC],x[GP]=dc,gc,gp
                x[EARLY]=0.
            gj,gq,gw,gcrit,gdrop=fd(x[GP])
            x[DD]=x[DC]/((1/3)*n) if n>0 else 0.
            x[GD]=x[GC]/((1/3)*gl_n) if gl_n>0 else 0.
            x[CAP]=jam_c*(1/3)*n+gj*(1/3)*gl_n
            x[RHO]=x[N]/((1/3)*4)
            x[DS]=max(0.,min(q_c*15/3600*n,x[DC]))
            x[DR]=max(0.,min(q_c,w_c*(jam_c-x[DD])))*15/3600*n
            x[GS],x[GR]=mixed_ulp(gq,gj,gcrit,x[GD],x[GC],gw,gdrop,gl_n)
            x[S],x[R]=x[DS]+x[GS],x[DR]+x[GR]

@njit(cache=True)
def flows_ulp(row,ramp_t,ramp_index):
    for i in range(120):
        x=row[i]
        cs=x[S]*x[P] if x[A]==0 else x[DS]+x[GS]*x[GP]
        hs=x[S]*(1-x[P]) if x[A]==0 else x[GS]*(1-x[GP])
        cr,hr=np.inf,np.inf
        if i<119:
            d=row[i+1]
            if x[A]>d[A]:
                if d[A]==0:
                    fraction=cs/(cs+hs) if cs+hs>0 else d[P]
                    cr,hr=d[R]*fraction,d[R]*(1-fraction)
                elif cs-d[DR]>0:
                    fraction=(cs-d[DR])/max(cs+hs-d[DR],1e-12)
                    cr,hr=d[DR]+d[GR]*fraction,d[GR]*(1-fraction)
                else:
                    cr,hr=d[DR],d[GR]
            elif x[A]==d[A]:
                if x[A]==0:
                    cr,hr=d[R]*x[P],d[R]*(1-x[P])
                elif cs-d[DR]>0:
                    fraction=(cs-d[DR])/max(cs+hs-d[DR],1e-12)
                    cr,hr=d[DR]+d[GR]*fraction,d[GR]*(1-fraction)
                else:
                    cr,hr=d[DR],d[GR]
            else:
                if cs>d[DR]:
                    cr=d[DR]+d[GR]*(cs-d[DR])/max(cs+hs-d[DR],1e-12)
                    hr=d[GR]*(1-(cs-d[DR])/max(cs+hs-d[DR],1e-12))
                else:
                    cr,hr=d[DR],d[GR]
        if i==ramp_index-1:
            rs,rp=ramp_t
            rcs=rs*rp
            rhs=rs-rcs
            d=row[i+1]
            if d[A]==0:
                fraction=(rcs+cs)/(cs+hs+rs) if cs+hs+rs>0 else d[P]
                cr,hr=d[R]*fraction,d[R]*(1-fraction)
            elif cs+rcs>d[DR]:
                cr=d[DR]+d[GR]*(cs+rcs-d[DR])/max(cs+hs+rs-d[DR],1e-12)
                hr=d[GR]*(1-(cs+rcs-d[DR])/max(cs+hs+rs-d[DR],1e-12))
            
            fc=0. if cs+rcs==0 else cr/(cs+rcs)
            fh=0. if hs+rhs==0 else hr/(hs+rhs)
            co=cs if fc>=1 else cs*fc
            ho=hs if fh>=1 else hs*fh
            x[RC]=rcs if fc>=1 else rcs*fc
            x[RH]=rhs if fh>=1 else rhs*fh
            x[RO]=x[RC]+x[RH]
        else:
            co,ho=min(cs,cr),min(hs,hr)
        if co==cs and x[A]>0:
            x[DO]=x[DS]
            x[DF]=x[DO]/((15/3600)*x[A]) if x[A]>0 else 0.
            x[GCO]=x[GS]*x[GP]
        elif co<cs and x[A]>0:
            fraction=co/cs if cs>0 else 0.
            x[DO]=x[DS]*fraction
            x[DF]=x[DO]/((15/3600)*x[A]) if x[A]>0 else 0.
            x[GCO]=x[GS]*x[GP]*fraction
        x[GHO]=ho
        x[GO]=x[GCO]+x[GHO]
        x[GF]=x[GO]/((15/3600)*(4-x[A])) if x[A]<3 else 0.
        x[OC],x[O]=co,co+ho

@njit(cache=True)
def advance_ulp(history,main,mp,ramp,rp,ramp_pair,actual,cav_in,ramp_actual,
            t,count,lane_config,set_lanes,ramp_index,cumulative,initial_total,offered_total,terminated):
    early=False
    performed=0
    for _ in range(count):
        row=history[t]
        if set_lanes:
            row[:,A]=lane_config
        parameters_ulp(row)
        flows_ulp(row,ramp_pair[t],ramp_index)
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
            history[t+1,i,N]=new_n
            history[t+1,i,P]=max(0.,new_p)
        
        early=bool(np.any(row[20:99,EARLY]>0))
        if cumulative[2]>=65000 or abs(initial_total+offered_total-cumulative[2])<10 or t>=3920:
            terminated=True
        performed+=1
        t+=1
        if early:
            break
    return t,performed,early,terminated

@njit(cache=True)
def ulp_clear_advance(history,main,mp,ramp,rp,ramp_pair,actual,cav_in,ramp_actual,
            t,count,lane_config,set_lanes,ramp_index,cumulative,initial_total,offered_total,terminated):
    early=False
    rejects=0
    acted=lane_config.copy()
    performed=0
    for _ in range(count):
        row=history[t]
        if set_lanes:
            row[:,A]=acted
        parameters_ulp(row)
        if set_lanes and np.any(row[20:100,EARLY]>0):
            rejects+=1
            n,p=row[:,N].copy(),row[:,P].copy()
            fallback=history[t-1,:,A].copy() if t>0 else np.zeros(120)
            row[:,:]=0.
            row[:,N],row[:,P],row[:,A]=n,p,fallback
            parameters_ulp(row)
            if np.any(row[20:100,EARLY]>0):
                fallback[:]=0.
                row[:,:]=0.
                row[:,N],row[:,P],row[:,A]=n,p,fallback
                parameters_ulp(row)
            acted=fallback.copy()
        flows_ulp(row,ramp_pair[t],ramp_index)
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

@njit(cache=True)
def reward_components(history,t,state):
    _,dcap,_,_,_=fd(1.)
    density=0.
    flow=0.
    for j in range(t-60,t):
        dtotal=0.
        ftotal=0.
        for i in range(120):
            x=history[j,i]
            if x[A]>0:
                dl=density_reward(x[DD],1.)
                gl=density_reward(x[GD],x[GP])
                dtotal+=(dl*x[A]+gl*(4-x[A]))/4
                _,gc,_,_,_=fd(x[GP])
                cap=dcap*x[A]+gc*(4-x[A])
            else:
                dtotal+=density_reward(x[RHO],x[P])
                _,gc,_,_,_=fd(x[P])
                cap=gc*4
            ftotal+=cap/(dcap*4)
        density+=dtotal/120
        flow+=ftotal/120
    rd,rf=density/60,flow/60
    security=max(state[4],state[9],state[14],state[19],state[24],state[29])
    penalty=-(security-.5) if .5<security<=1 else 0.
    return .5*rd+.5*rf+penalty,.5*rd,.5*rf,penalty

@njit(cache=True)
def observation(history,t,current_action,actual,ramp_index):
    k=t-3
    state=np.zeros(38)
    for seg in range(6):
        x=history[k,seg*20+10]
        if x[A]>0:
            state[seg*5:seg*5+4]=np.array([x[DD],x[GD],1.,x[GP]])
        else:
            state[seg*5:seg*5+4]=np.array([0.,x[RHO],0.,x[P]])
        security=0.
        for i in range(seg*20,(seg+1)*20):
            y=history[k,i]
            if y[A]>0:
                jam,_,_,_,_=fd(y[GP])
                maximum=jam*(1/3)*(4-y[A])
                value=y[GC]/maximum if maximum>0 else 0.
            else:
                jam,_,_,_,_=fd(y[P])
                maximum=jam*4*(1/3)
                value=y[N]/maximum if maximum>0 else 0.
            security=max(security,value)
        state[seg*5+4]=security
    state[30:34]=current_action
    main_window=np.zeros(60)
    ramp_window=np.zeros(60)
    out_window=np.zeros(60)
    
    
    for j in range(60):
        index=k-59+j
        main_window[j]=actual[index]
        ramp_window[j]=history[index,ramp_index-1,RO]
        out_window[j]=history[index,119,O]
    state[34]=main_window.mean()
    state[35]=ramp_window.mean()
    state[36]=out_window.mean()
    state[37]=t/5000
    return np.round(state,2).astype(np.float32)
