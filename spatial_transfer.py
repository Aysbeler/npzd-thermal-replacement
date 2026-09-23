"""
spatial_transfer.py

Out-of-sample test of the calibrated thermal optima across neighbouring reanalysis cells.

The optima are estimated at one cell, so a good fit there is a within-sample result: with two
free optima and a free scale, a twelve-point seasonal cycle can be matched for reasons that need
not generalize.  The reanalysis covers many cells around the calibration site with independent
seasonal cycles, which supplies a genuine held-out test.

The protocol is a transfer, not a refit.  The optima fitted at the central cell
(38.4792 N, 26.3750 E) are applied unchanged at every other cell; only the observation scale
c_obs is recomputed there, since it absorbs the local mean biomass and the nitrogen-to-carbon
conversion and carries no information about thermal niches.  Each cell is driven by its own SST
climatology, so the model output differs from cell to cell entirely through the forcing.

Two baselines make the result interpretable:

  refit    - optima re-estimated at each cell, an upper bound on achievable fit;
  harmonic - a three-parameter single harmonic fitted at each cell, the descriptive null that
             outperformed the mechanistic model in leave-one-month-out prediction at the
             central cell.

If the transferred optima track the neighbouring cycles nearly as well as locally refitted ones,
the calibration is not an artefact of fitting two parameters to twelve points at a single
location.
"""
import numpy as np, warnings; warnings.filterwarnings('ignore')
from scipy.integrate import solve_ivp
import xarray as xr
import model

import os
_HERE = os.path.dirname(os.path.abspath(__file__))
_RAW = os.path.join(_HERE, 'data', 'raw_cmems')
PL = os.path.join(_RAW, 'cmems_mod_med_bgc-plankton_my_4_2km_P1M-m_1782972759117.nc')
TE = os.path.join(_RAW, 'cmems_mod_med_phy-temp_my_4_2km_P1M-m_1782971879835.nc')
CLAT, CLON = 38.4792, 26.3750
NYR = 22
DC = (np.arange(12) + 0.5)*365/12.0
P0 = model.make_p()


def load():
    """Monthly climatologies of SST and phytoplankton carbon on the common grid."""
    pl = xr.open_dataset(PL); te = xr.open_dataset(TE)
    la, lo = pl.latitude.values, pl.longitude.values
    lat_t, lon_t = te.latitude.values, te.longitude.values
    P = pl['phyc'].isel(depth=0).values[:NYR*12]
    T = te['thetao'].isel(depth=0).values[:NYR*12]
    cells = {}
    for a in range(len(la)):
        for b in range(len(lo)):
            if not np.isfinite(P[:, a, b]).all():
                continue
            i = int(np.argmin(np.abs(lat_t - la[a])))
            j = int(np.argmin(np.abs(lon_t - lo[b])))
            if abs(lat_t[i]-la[a]) > 0.01 or abs(lon_t[j]-lo[b]) > 0.01:
                continue
            if not np.isfinite(T[:, i, j]).all():
                continue
            cells[(float(la[a]), float(lo[b]))] = (
                np.nanmean(T[:, i, j].reshape(NYR, 12), axis=0),
                np.nanmean(P[:, a, b].reshape(NYR, 12), axis=0))
    return cells


def monthly(sst, o1, o2, years=6):
    """Monthly means of the annually periodic solution under the given SST climatology."""
    def Tf(t):
        return float(np.interp(np.mod(t, 365.0), DC, sst, period=365.0))

    def rhs(t, y):
        N, P1, P2, Z, D = [max(v, 1e-12) for v in y]
        T = Tf(t)
        th1 = model.theta(T, o1, P0['w1'], P0['wskew1'])
        th2 = model.theta(T, o2, P0['w2'], P0['wskew2'])
        rh = model.rho(T, P0['rho0'], P0['Q10'], P0['Tref'])
        s = P0['switch']; a = P1**s; b = P2**s; tot = a + b + 1e-12
        gP1 = (a/tot)*model.g(P1, P0['KP1']); gP2 = (b/tot)*model.g(P2, P0['KP2'])
        gr1 = P0['mu1']*th1*model.f(N, P0['KN1'])*P1
        gr2 = P0['mu2']*th2*model.f(N, P0['KN2'])*P2
        return [P0['I0'] - P0['lN']*N - gr1 - gr2 + rh*D,
                gr1 - P0['mZ']*gP1*Z - P0['d1']*P1,
                gr2 - P0['mZ']*gP2*Z - P0['d2']*P2,
                P0['e']*P0['mZ']*(gP1+gP2)*Z - P0['dZ']*Z,
                P0['d1']*P1 + P0['d2']*P2 + (1-P0['e'])*P0['mZ']*(gP1+gP2)*Z
                + P0['dZ']*Z - rh*D - P0['lD']*D]

    s = solve_ivp(rhs, [0, years*365], [1, .5, .5, .4, 2.], method='RK45',
                  rtol=1e-7, atol=1e-9, max_step=5.0, dense_output=True)
    t0 = (years-1)*365.0
    tt = np.linspace(t0, t0+365, 1464)
    Y = s.sol(tt)
    mon = (np.floor(((tt-t0)/365.0)*12).astype(int)) % 12
    tot = Y[1] + Y[2]
    return np.array([tot[mon == m].mean() for m in range(12)])


def rscore(pred, obs):
    """Correlation after the least-squares scale, which is the only refitted quantity."""
    if pred.max() < 1e-9:
        return np.nan
    c = (pred @ obs)/(pred @ pred)
    return float(np.corrcoef(c*pred, obs)[0, 1])


def harmonic_r(sst, obs):
    ph = 2*np.pi*(np.arange(12) + 0.5)/12
    H = np.column_stack([np.ones(12), np.cos(ph), np.sin(ph)])
    beta = np.linalg.lstsq(H, obs, rcond=None)[0]
    return float(np.corrcoef(H @ beta, obs)[0, 1])


def refit_r(sst, obs, G1=np.arange(14.5, 17.1, 0.75), G2=np.arange(18.5, 21.1, 0.75)):
    best = -9
    for o1 in G1:
        for o2 in G2:
            best = max(best, rscore(monthly(sst, o1, o2), obs))
    return best


if __name__ == '__main__':
    cells = load()
    print(f"cells with both SST and phytoplankton fully recorded: {len(cells)}")
    key = min(cells, key=lambda k: abs(k[0]-CLAT) + abs(k[1]-CLON))
    print(f"calibration cell: ({key[0]:.4f}, {key[1]:.4f})\n")

    sst0, obs0 = cells[key]
    O1, O2 = 15.70, 19.20                       # optima fitted at the central cell
    print(f"transferring optima ({O1}, {O2}) unchanged; only c_obs is rescaled\n")

    dist = {k: np.hypot((k[0]-key[0])*111, (k[1]-key[1])*87) for k in cells}
    order = sorted(cells, key=lambda k: dist[k])
    sel = order[:10]

    print(f"{'cell':>22} {'km':>6} {'transfer r':>11} {'refit r':>9} {'harmonic r':>11}")
    print("-"*64)
    tr, rf, hm = [], [], []
    for k in sel:
        sst, obs = cells[k]
        a = rscore(monthly(sst, O1, O2), obs)
        b = refit_r(sst, obs)
        c = harmonic_r(sst, obs)
        tr.append(a); rf.append(b); hm.append(c)
        print(f"({k[0]:8.4f},{k[1]:8.4f}) {dist[k]:6.1f} {a:11.3f} {b:9.3f} {c:11.3f}")
    tr, rf, hm = np.array(tr), np.array(rf), np.array(hm)
    d = np.array([dist[k] for k in sel])
    print("-"*64)
    # The ten nearest fully recorded cells fall into two groups separated by a gap in the
    # grid: those within NEAR_KM of the calibration cell, and a second group about 20 km away.
    NEAR_KM = 12.0
    for label, m in (("within %.0f km" % NEAR_KM, d <= NEAR_KM), ("beyond %.0f km" % NEAR_KM, d > NEAR_KM)):
        if not m.any():
            continue
        print(f"\n{label}: {m.sum()} cells, {d[m].min():.1f}-{d[m].max():.1f} km")
        print(f"  median  transfer {np.median(tr[m]):.3f}  refit {np.median(rf[m]):.3f}  harmonic {np.median(hm[m]):.3f}")
        print(f"  range   transfer {tr[m].min():.3f}-{tr[m].max():.3f}")
        print(f"  transfer beats harmonic at {int((tr[m] > hm[m]).sum())}/{m.sum()} cells")
