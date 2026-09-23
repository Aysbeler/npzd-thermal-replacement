import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import model

def run_face(T, I0, absent, y0, Tend=4000):
    # absent: which phyto set to 0 (1 -> P1=0 face E2 ; 2 -> P2=0 face E1)
    p=model.make_p(T=T, I0=I0); rhs=model.make_rhs(p)
    y0=np.array(y0,float)
    if absent==1: y0[1]=0.0
    else:         y0[2]=0.0
    sol=solve_ivp(rhs,[0,Tend],y0,method='RK45',rtol=1e-8,atol=1e-10,
                  t_eval=np.linspace(Tend*0.75,Tend,3000))
    return sol.y

def face_test(absent, Ts, I0=0.5):
    print("\n--- Face %s=0  (single-type subsystem, checking global attraction + no cycle) ---"
          %('P1' if absent==1 else 'P2'))
    print(" T     #ICs  max_spread(endpts)   max_osc_amp   verdict")
    ICs=[]
    for N0 in (0.5,3.0): 
        for Pp in (0.3,2.5):
            for Z0 in (0.2,3.0):
                for D0 in (1.0,8.0):
                    ICs.append([N0,Pp,Pp,Z0,D0])
    for T in Ts:
        ends=[]; oscs=[]
        for ic in ICs:
            Y=run_face(T,I0,absent,ic)
            present_idx = 2 if absent==1 else 1   # the surviving phyto index
            comp=Y[present_idx]
            ends.append(Y[:,-1])
            oscs.append(comp.max()-comp.min())    # late-window peak-to-peak
        ends=np.array(ends)
        spread=np.max(np.ptp(ends,axis=0))        # spread of endpoints across ICs
        maxosc=max(oscs)
        ok = (spread<1e-3) and (maxosc<1e-3)
        print("  %4.1f   %3d    %.2e            %.2e     %s"%(
            T,len(ICs),spread,maxosc,"OK (global eq, no cycle)" if ok else "CHECK"))

# window ~ [15.37, 20.10]; test interior points
face_test(absent=2, Ts=[15.5,17.0,18.5,20.0])   # E1 face (P2=0), cold type survives
face_test(absent=1, Ts=[15.5,17.0,18.5,20.0])   # E2 face (P1=0), warm type survives
