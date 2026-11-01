from ._legacy.kernel import *
from .kernels.lane3 import flows_ulp
HD,HDO,HDNEXT,STRANDED,RELEASED=range(32,37)

@njit(cache=True)
def dl_share(x):
    return min(1.,max(0.,(x[N]*x[P]-x[GC]*x[GP])/x[DC])) if x[DC]>1e-12 else 1.

@njit(cache=True)
def mixed_nc(q,jam,crit,rho,cars,wave,drop,lanes):
    return mixed(q,jam,crit,rho,cars,wave,drop,lanes) if rho>np.nextafter(crit,np.inf) else (max(0.,min(q*15/3600*lanes,cars)),max(0.,min(q,wave*(jam-rho))*15/3600*lanes))

@njit(cache=True)
def retained(h,oldn,newn,Ntotal,p,eta):
    
    if newn<=oldn:
        after=h*newn/oldn if oldn>0 else 0.
        return after,0.,h-after
    exposed=max(0.,Ntotal*(1-p)-h)*(newn-oldn)/(3-oldn)
    return h+eta*exposed,eta*exposed,0.

@njit(cache=True)
def flows_stay(row,ramp_t,ramp_index,eta):
    if not np.any(row[:,A]>0):
        flows_ulp(row,ramp_t,ramp_index)
        return
    for i in range(120):
        x=row[i];dp=dl_share(x) if x[A]>0 else 1.
        cs=x[S]*x[P] if x[A]==0 else x[DS]*dp+x[GS]*x[GP]
        hds=x[DS]*(1-dp) if x[A]>0 else 0.
        hgs=x[S]*(1-x[P]) if x[A]==0 else x[GS]*(1-x[GP])
        rs=ramp_t[0] if i==ramp_index-1 else 0.;rc=rs*ramp_t[1];rh=rs-rc
        cd=cs+rc
        overlap=0.;exposed=0.;fdc=1.;fg=1.;fc=1.
        if i<119:
            d=row[i+1]
            overlap=min(x[A],d[A])/x[A] if x[A]>0 else 0.
            exposed=eta*max(0.,d[A]-x[A])/(3-x[A])
            hdemand=hds*overlap+hgs*exposed
            hgdemand=hds*(1-overlap)+hgs*(1-exposed)+rh
            if d[A]==0:
                
                scale=min(1.,d[R]/max(cd+hgdemand,1e-12))
                fc=scale;fg=scale
            else:
                
                den=cd+hdemand
                reserve_c=d[DR]*cd/den if den>0 else 0.
                reserve_h=d[DR]*hdemand/den if den>0 else 0.
                fdc=min(1.,reserve_h/hdemand) if hdemand>0 else 1.
                remc=max(0.,cd-reserve_c)
                frac=remc/(remc+hgdemand) if remc+hgdemand>0 else 0.
                fc=min(1.,(reserve_c+d[GR]*frac)/cd) if cd>0 else 0.
                fg=min(1.,d[GR]*(1-frac)/hgdemand) if hgdemand>0 else 1.
        hdo=hds*(overlap*fdc+(1-overlap)*fg) if i<119 else hds
        hgo=hgs*(exposed*fdc+(1-exposed)*fg) if i<119 else hgs
        co=cs*fc;ho=hdo+hgo
        x[RC]=rc*fc;x[RH]=rh*fg;x[RO]=x[RC]+x[RH]
        x[HDO]=hdo
        x[HDNEXT]=(hds*overlap+hgs*exposed)*fdc if i<119 else 0.
        x[DO]=x[DS]*dp*fc+hdo if x[A]>0 else 0.
        x[GCO]=x[GS]*x[GP]*fc if x[A]>0 else co
        x[GHO]=hgo;x[GO]=x[GCO]+hgo
        x[DF]=x[DO]/((15/3600)*x[A]) if x[A]>0 else 0.
        x[GF]=x[GO]/((15/3600)*(3-x[A]))
        x[OC],x[O]=co,co+ho
