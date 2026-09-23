# Warming-driven transcritical replacement of phytoplankton types in an NPZD model with prey switching

Code and data reproducing every quantitative result in the paper. Each script reproduces one
result and states in its header which section, table or figure it belongs to.

## Requirements

Python 3.12. Install the pinned versions with

```
pip install -r requirements.txt
```

`xarray`, `netCDF4` and `pandas` are needed only by the three scripts that read the raw
reanalysis.

Every stochastic routine fixes its seed, listed below. Run the scripts from the repository root.

## Layout

```
model.py            model equations and the reference parameters of Table 1
*.py                one script per result, listed below
data/raw_cmems/     CMEMS reanalysis subsets (temperature, plankton, nutrients)
data/derived/       climatologies extracted from them
data/README_DATA.md provenance of every data file
```

`model.py` is imported by every other script, so the reference parameters are defined in exactly
one place.

## Quick check

```
python invasion_edges.py     # coexistence window [15.37, 20.10] C
python calib_cobs.py         # fitted optima 15.7 and 19.2 C, c_obs = 6.08, r = 0.952
python zerohopf.py           # codimension-two point (20.153834, 0.520075)
```

Each runs in under a minute.

## Scripts by section

### Section 5 - Boundary equilibria and invasion thresholds

| Script | Result |
|---|---|
| `invasion_edges.py` | window edges $T_1=15.37$, $T_2=20.10$ as roots of the invasion fitness |
| `invasion_criterion_check.py` | analytic invasion criterion, Proposition 7.1 |
| `c3_zface_true.py`, `c4_verify.py` | boundary equilibria and their transverse eigenvalues |

### Section 6 - Bifurcation analysis

| Script | Result |
|---|---|
| `sotomayor_coefficients.py` | Sotomayor coefficients at both edges, Table 2 |
| `continuation.py` | pseudo-arclength continuation of the coexistence branch |
| `mc_hopf.py`, `warm_hopf_cycle.py` | enrichment and warming Hopf points, Table 3 |
| `lyapunov_coefficient_exact.py` | first Lyapunov coefficients from symbolic derivatives, Table 3 |
| `jacobian_symbolic.py` | symbolic Jacobian and multilinear forms used above |
| `fig6_spectrum.py` | eigenvalue crossing at the Hopf points |
| `face_hopf_check.py` | face-Hopf transitions at 18.15 and 19.79 C |
| `zerohopf.py` | codimension-two point and its spectrum, Section 6.3 |

### Section 7 - Persistence

| Script | Result |
|---|---|
| `predator_free_face.py` | separating-hyperplane test of Lemma 7.2 and the failure band |
| `breakeven_crossing.py` | degenerate equilibrium family at $T_\times=17.5949$ C |
| `mean_invasion.py` | mean invasion rates on the face attractors |
| `face_cycle_invasion_P1.py`, `face_cycle_invasion_P2.py` | invasion rates on the face cycles, Table 7 |
| `face_cycle_simplicity.py` | face cycles are simple limit cycles, hypothesis (P4) |

### Section 8 - Robustness and mechanism

| Script | Result | Seed |
|---|---|---|
| `solver_independence.py` | five integrators agree to $2.6\times10^{-12}$ | -- |
| `floquet_cycles.py` | Floquet multipliers of the face and interior cycles | -- |
| `periodic_continuation.py` | collocation check behind the sampling caveat | -- |
| `plateau_hysteresis.py` | single-type plateau and the hysteresis sweep | -- |
| `mc_run.py`, `mc_edges.py` | Monte Carlo robustness, Table 4 | 2024 |
| `prcc_large.py`, `prcc_check.py` | partial rank correlations, Table 5 | 7 |
| `cond_robust.py` | conditioning of the sensitivity analysis | -- |
| `robust_extra_variants.py`, `table6_osc.py` | alternative functional forms, Table 6 | -- |
| `epsilon_continuation.py` | window width against background grazing $\varepsilon$ | -- |
| `q10_plane.py` | window width over the $(Q_{10,g},Q_{10,m})$ plane | -- |
| `mechanism_factorial.py` | factorial trait design and three grazing responses, Table 8 | -- |

### Section 9 - Reanalysis data

| Script | Result | Seed |
|---|---|---|
| `cmems_extract.py` | builds the twelve-month climatology from `data/raw_cmems/` | -- |
| `calib_cobs.py` | fitted optima and observation scale | -- |
| `neighbor_check.py` | four adjacent cells, $r$ between 0.91 and 0.94 | -- |
| `spatial_transfer.py` | transfer test: within 10 km $0.952/0.952/0.896$ (transferred/refitted/harmonic); beyond 20 km the transfer fails | -- |
| `identifiability_collinearity.py` | collinearity indices | -- |
| `identifiability_uncertainty.py` | profile and bootstrap intervals, Table 9 | 20260718 |
| `null_models.py`, `aic_bic_modelselection.py` | null-model comparison, Table 10 | -- |
| `table7_recompute.py` | table entries recomputed from the fitted residuals | -- |
| `mld_null.py` | mixed-layer nulls | -- |
| `onetype_fairness.py` | one-type null under quadratic closure | -- |
| `quasistatic_check.py` | seasonal edge crossings and recovery lag | -- |
| `warming_projection.py` | cold-type share under a uniform temperature shift | -- |

## Figures

| Figure | Script | Output |
|---|---|---|
| 2 | `fig_thermal_niches.py` | `fig_thermal_niches.png` |
| 3 | `fig_bifurcation_merged.py` | `fig_bifurcation_merged.png` |
| 4 | `fig01_enrichment_hopf.py` | `hoph.png` |
| 5 | `fig3_TI0_plane.py` | `fig3_TI0_plane.png` |
| 6 | `fig_rk4_longrun.py` | `fig_rk4_longrun.png` |
| 7 | `fig7_phase_corrected.py` | `fig7_phase_corrected.pdf` |
| 8 | `fig12_calibration.py` | `fig12_calibration.png` |
| 9 | `fig13_modelselection.py` | `fig13_modelselection.png` |

Figure 1 is a schematic drawn in the manuscript source. `fig07_mc.py` draws the Monte Carlo
distributions summarized in Table 4.

## Running times

Most scripts finish in under a minute on one core. A few, such as `invasion_criterion_check.py`
and `fig13_modelselection.py`, take several minutes, and `prcc_large.py` and
`identifiability_uncertainty.py` longer still because of the bootstrap.

## Data

The reanalysis subsets are public Copernicus Marine Service products; `data/README_DATA.md`
gives their identifiers and the extraction. They are redistributed here only so that the results
can be reproduced without an account.
