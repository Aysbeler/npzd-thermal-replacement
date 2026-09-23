import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.optimize import brentq, fsolve
import model
from mc_edges import G, Nstar   # G(i,N,T,p), Nstar via Phi

def predator_free_eq(i, T, p):
    # Z=0, single type i: N solves G_i(N)=nu_i (break-even); then N-balance gives P_i, D
    d = p['d1'] if i==1 else p['d2']
    # N_c where G_i(N_c)=d  (G increasing in N)
    try:
        Nc=brentq(lambda N: G(i,N,T,p)-d, 1e-9, 50.0)
    except Exception:
        return None
    # equilibrium: dP=0 gives G_i=d (satisfied). N,D from N and D balances with Z=0:
    # dD=0: d*P = (rho+lD)*D -> D=d*P/(rho+lD); dN=0: I0 - lN*N - G_i(N)*P + rho*D=0
    from mc_edges import rho as _r
    rh=_r(T,p)
    def Nbal(P):
        D=d*P/(rh+p['lD']); return p['I0']-p['lN']*Nc - d*P + rh*D  # G_i(Nc)=d
    # solve for P: linear in P
    # I0 - lN*Nc - d*P + rho*d*P/(rho+lD) = 0 -> P*(d - rho*d/(rho+lD)) = I0 - lN*Nc
    coef=d*(1 - rh/(rh+p['lD'])); P=(p['I0']-p['lN']*Nc)/coef
    if P<=0: return None
    D=d*P/(rh+p['lD'])
    return np.array([Nc, P if i==1 else 0.0, P if i==2 else 0.0, 0.0, D]), P

def grazer_eig_at(yf,P_i,i,T,p):
    KP=p['KP1'] if i==1 else p['KP2']
    return p['e']*p['mZ']*(P_i/(KP+P_i)) - p['dZ']   # phi_i=1 (single type)

print("--- TRUE {Z=0} predator-free single-type equilibria + grazer invasion eigenvalue ---")
print(" T     winner  N_c     P_free   grazer_eig   (need >0 for Z-face repelling)")
for T in [15.5,16,17,18,19,20.0]:
    p=model.make_p(T=T,I0=0.5)
    cands=[]
    for i in (1,2):
        r=predator_free_eq(i,T,p)
        if r is not None:
            yf,P=r; cands.append((yf[0],i,yf,P))   # (N_c, i, state, P)
    if not cands: print("  %4.1f  none"%T); continue
    # attractor = lower break-even N_c (wins competition)
    Nc,i,yf,P=min(cands,key=lambda c:c[0])
    eig=grazer_eig_at(yf,P,i,T,p)
    print("  %4.1f    P%d    %.4f  %.4f   %+.4f"%(T,i,Nc,P,eig))
