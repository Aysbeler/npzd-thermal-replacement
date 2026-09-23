import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import model
C1,C2 = model.C1, model.C2                 # blue (rate), red (thermal)

d=np.load('mc_results.npz'); N=int(d['N'])
A1,A2 = d['A1'],d['A2']; B1,B2=d['B1'],d['B2']
Aw=A2-A1; Bw=B2-B1
refs=(15.37,20.10,4.73)
nA=int(np.sum(np.isfinite(A1)&np.isfinite(A2)))
nB=int(np.sum(np.isfinite(B1)&np.isfinite(B2)))

fig,ax=plt.subplots(1,3,figsize=(13,3.7))
panels=[('$T_1$ (lower edge) ($^\\circ$C)',A1,B1,refs[0]),
        ('$T_2$ (upper edge) ($^\\circ$C)',A2,B2,refs[1]),
        ('window width ($^\\circ$C)',Aw,Bw,refs[2])]
for a,(xlab,xa,xb,ref) in zip(ax,panels):
    va=xa[np.isfinite(xa)]; vb=xb[np.isfinite(xb)]
    lo=min(va.min(),vb.min()); hi=max(va.max(),vb.max())
    bins=np.linspace(lo,hi,22)
    a.hist(va,bins=bins,color=C1,alpha=0.65,label=f'rates $\\pm15\\%$ (n={nA})',edgecolor='white',linewidth=0.4)
    a.hist(vb,bins=bins,color=C2,alpha=0.55,label=f'+ thermal (n={nB})',edgecolor='white',linewidth=0.4)
    a.axvline(ref,ls='--',color='0.25',lw=1.5,label=f'reference {ref:g}')
    a.set_xlabel(xlab); a.set_ylabel('count')
    a.legend(fontsize=8,frameon=False)
fig.suptitle('Monte Carlo distributions of bifurcation thresholds under parameter uncertainty',
             fontsize=11,y=1.02)
plt.tight_layout()
plt.savefig('fig9_mc_robustness.png',dpi=300,bbox_inches='tight')
print("saved. nA=%d nB=%d"%(nA,nB))
print("width rate  95%% [%.2f,%.2f] ref 4.73"%(np.nanpercentile(Aw,2.5),np.nanpercentile(Aw,97.5)))
print("width therm 95%% [%.2f,%.2f]"%(np.nanpercentile(Bw,2.5),np.nanpercentile(Bw,97.5)))
