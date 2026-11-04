"""Plot matched compliance outcomes from CSV."""
import numpy as np
import pandas as pd

def render(data, output):
    from cav_dl.utils.style import configure,save
    import matplotlib.pyplot as plt
    q=pd.read_csv(data)
    with plt.rc_context({'font.size':17,'axes.titlesize':19,'axes.labelsize':18,'xtick.labelsize':16,'ytick.labelsize':16}):
        fig, axes=plt.subplots(2,3,figsize=(15,8.8),sharex=True,sharey=True)
        for ax,c in zip(axes.flat,[1,.9,.8,.7,.6,.5]):
            s=q[q.compliance.eq(c)&q.p.between(.1,.9)].sort_values('p')
            ax.plot(s.p,s.PPO_vs_FS1_TTS_pct,'o-',color='#49666d',lw=2,ms=5)
            ax.fill_between(s.p,s.PPO_vs_FS1_low95_pct,s.PPO_vs_FS1_high95_pct,color='#49666d',alpha=.16,lw=0)
            ax.axhline(0,color='gray',lw=.8,ls='--')
            ax.set_title(f'HV compliance = {c:.1f}',pad=10)
            ax.set_xticks([.1,.3,.5,.7,.9]); ax.grid(alpha=.18)
        fig.supxlabel('CAV penetration rate',y=.02)
        fig.supylabel('Complete TTS reduction relative to FS1 (%)',x=.01)
        fig.tight_layout(rect=(.025,.055,1,1))
        save(fig,output,'TTS_compliance')

