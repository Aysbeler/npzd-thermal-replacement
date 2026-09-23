import numpy as np, warnings; warnings.filterwarnings('ignore')
import xarray as xr, model
from calib_cobs import make_Tfun, seasonal_rhs, model_monthly
from scipy.integrate import solve_ivp

import os
U=os.path.join(os.path.dirname(os.path.abspath(__file__)),'data','raw_cmems')+os.sep
phy=xr.open_dataset(U+'cmems_mod_med_phy-temp_my_4_2km_P1M-m_1782971879835.nc')
pla=xr.open_dataset(U+'cmems_mod_med_bgc-plankton_my_4_2km_P1M-m_1782972759117.nc')
lat=phy['latitude'].values; lon=phy['longitude'].values
sst_all=phy['thetao'].isel(depth=0).values      # (264,9,14)
phyc_all=pla['phyc'].isel(depth=0).values        # (264,9,14)
nt=sst_all.shape[0]; nyr=nt//12                   # 22 years
def clim(a): return np.nanmean(a[:nyr*12].reshape(nyr,12,*a.shape[1:]),axis=0)  # (12,9,14)
sstC=clim(sst_all); phycC=clim(phyc_all)

ci,cj=6,1                                          # center 38.4792N, 26.375E
print("center lat=%.4f lon=%.4f"%(lat[ci],lon[cj]))
# collect sea cells in a 5x5 neighborhood (finite phyc), exclude center
cells=[]
for di in range(-2,3):
    for dj in range(-2,3):
        i,j=ci+di,cj+dj
        if 0<=i<9 and 0<=j<14 and not (di==0 and dj==0):
            if np.all(np.isfinite(phycC[:,i,j])) and np.all(np.isfinite(sstC[:,i,j])) and phycC[:,i,j].mean()>0:
                cells.append((i,j))
print("sea neighbor cells found:",len(cells))

def coarse_fit(sst,phyc):
    Tfun=make_Tfun(sst); best=None
    for o1 in (14,15,16,17):
        for o2 in (18,19,20,21,22):
            p=model.make_p(Topt1=o1,Topt2=o2); Ph=model_monthly(p,Tfun,years=5)
            c=np.sum(Ph*phyc)/np.sum(phyc**2); rss=np.sum((Ph-c*phyc)**2)
            r=np.corrcoef(Ph,phyc)[0,1]
            if best is None or rss<best[0]: best=(rss,r,o1,o2,c,int(np.argmax(Ph))+1)
    return best  # rss,r,Topt1,Topt2,c_obs,model_peak_month

# center reference
cr=coarse_fit(sstC[:,ci,cj],phycC[:,ci,cj])
print("\nCENTER coarse-fit: r=%.3f Topt=(%d,%d) c_obs=%.2f phyc_peak=M%d model_peak=M%d"%(
    cr[1],cr[2],cr[3],cr[4],int(np.argmax(phycC[:,ci,cj]))+1,cr[5]))

print("\nneighbor  lat     lon     phyc_peak  model_peak  r      Topt1 Topt2")
rs=[]; peaks=[]
for (i,j) in cells:
    b=coarse_fit(sstC[:,i,j],phycC[:,i,j])
    pk=int(np.argmax(phycC[:,i,j]))+1
    rs.append(b[1]); peaks.append(pk)
    print("  %5.4f %7.4f   M%-2d       M%-2d       %.3f  %d    %d"%(lat[i],lon[j],pk,b[5],b[1],b[2],b[3]))
rs=np.array(rs); peaks=np.array(peaks)
print("\nSUMMARY over %d sea neighbors:"%len(cells))
print("  r: min=%.3f median=%.3f max=%.3f"%(rs.min(),np.median(rs),rs.max()))
print("  phyc peak month: values=%s (mode=M%d)"%(sorted(set(peaks.tolist())),int(np.bincount(peaks).argmax())))
