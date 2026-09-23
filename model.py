import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif'],
    'font.size':11,'axes.labelsize':12,'axes.titlesize':12,
    'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':9.5,
    'axes.linewidth':0.9,'lines.linewidth':1.8,
    'xtick.direction':'in','ytick.direction':'in',
    'xtick.major.size':4,'ytick.major.size':4,
    'xtick.minor.visible':True,'ytick.minor.visible':True,
    'figure.dpi':120,'savefig.dpi':300,'savefig.bbox':'tight',
    'mathtext.fontset':'dejavuserif'})

C1='#1f5fa8'; C2='#c0392b'; CZ='#27865a'; CN='#7d5ba6'; CD='#b8860b'

def theta(T,Topt,w,wskew): return np.exp(-((T-Topt)/w)**2) if T<=Topt else np.exp(-((T-Topt)/wskew)**2)
def rho(T,rho0,Q10,Tref): return rho0*Q10**((T-Tref)/10)
def f(N,KN): return N/(KN+N)
def g(P,KP): return P/(KP+P)
def hD(D,KD): return D/(KD+D)
def make_p(**o):
    p=dict(I0=0.5,lN=0.12,lD=0.05,mu1=1.1,mu2=1.1,KN1=1.2,KN2=0.5,mZ=0.4,KP1=1.0,KP2=1.0,
        d1=0.1,d2=0.1,e=0.6,dZ=0.15,Topt1=15.0,w1=7.0,wskew1=4.0,Topt2=25.0,w2=7.0,wskew2=4.0,
        rho0=0.05,Q10=2.0,Tref=20.0,KD=0.5,T=20.0,switch=2.0); p.update(o); return p
def make_rhs(p2):
    def rhs(t,y):
        N,P1,P2,Z,D=[max(x,1e-12) for x in y]; T=p2['T']
        th1=theta(T,p2['Topt1'],p2['w1'],p2['wskew1']);th2=theta(T,p2['Topt2'],p2['w2'],p2['wskew2'])
        rh=rho(T,p2['rho0'],p2['Q10'],p2['Tref']); fN1=f(N,p2['KN1']);fN2=f(N,p2['KN2'])
        s=p2['switch'];w1s=P1**s;w2s=P2**s;tot=w1s+w2s+1e-12;phi1=w1s/tot;phi2=w2s/tot
        gP1=phi1*g(P1,p2['KP1']);gP2=phi2*g(P2,p2['KP2'])
        gr1=p2['mu1']*th1*fN1*P1;gr2=p2['mu2']*th2*fN2*P2;remin=rh*D
        return [p2['I0']-p2['lN']*N-gr1-gr2+remin,
                gr1-p2['mZ']*gP1*Z-p2['d1']*P1,
                gr2-p2['mZ']*gP2*Z-p2['d2']*P2,
                p2['e']*p2['mZ']*(gP1+gP2)*Z-p2['dZ']*Z,
                p2['d1']*P1+p2['d2']*P2+(1-p2['e'])*p2['mZ']*(gP1+gP2)*Z+p2['dZ']*Z-remin-p2['lD']*D]
    return rhs

def run(T,I0,y0=[1.0,0.8,0.8,0.4,0.4],Tend=9000,switch=2.0):
    p2=make_p(T=T,I0=I0,switch=switch);rhs=make_rhs(p2)
    sol=solve_ivp(rhs,[0,Tend],y0,method='RK45',rtol=1e-9,atol=1e-11,t_eval=np.linspace(Tend*0.7,Tend,300))
    return sol
