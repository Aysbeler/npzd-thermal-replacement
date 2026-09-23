# Data

Source data and derived products for the calibration section.

## Provenance

Copernicus Marine Service Mediterranean Sea reanalysis, monthly means, surface level,
2001-2022, at the grid cell nearest 38.48 N, 26.38 E (realized cell 38.4792 N, 26.3750 E),
eastern Aegean. Products:

| Variable | Product family | File |
|---|---|---|
| Temperature `thetao` | MEDSEA physics reanalysis (phy-temp) | `raw_cmems/cmems_mod_med_phy-temp_my_4_2km_P1M-m_1782971879835.nc` |
| Chlorophyll `chl`, phytoplankton carbon `phyc` | MEDSEA biogeochemistry (bgc-plankton) | `raw_cmems/cmems_mod_med_bgc-plankton_my_4_2km_P1M-m_1782972759117.nc` |
| Nitrate `no3` | MEDSEA biogeochemistry (bgc-nut) | `raw_cmems/cmems_mod_med_bgc-nut_my_4_2km_P1M-m_1782972904103.nc` |

These are spatial subsets of the public products, downloaded through the Copernicus Marine
Toolbox. The full products are redistributable from Copernicus, not from this archive.

## The dataset actually used

`derived/station_climatology_point.csv` is the twelve-point monthly climatology used for every
fitted result in the paper: columns `month, sst, sst_max, chl, no3, phyc`. Each entry is the mean
over the available years of that calendar month at the single reference cell. It is regenerated
exactly by

```
python cmems_extract.py  raw_cmems/...phy-temp...nc  raw_cmems/...bgc-plankton...nc  raw_cmems/...bgc-nut...nc
```

Regeneration reproduces the stored file to 2e-6, which is the float32 rounding of the stored CSV.

## Other derived files

These are extractions made while selecting the reference cell and checking spatial
representativeness. They are not inputs to any result in the paper and are included only so the
selection can be retraced.

| File | Contents |
|---|---|
| `station_climatology.csv` | Twelve-point climatology from an area-averaged box rather than the single cell |
| `box_timeseries.csv` | Monthly time series, box-averaged, 2001-2022 |
| `surface_3comp.csv` | Monthly surface series with additional variables (phosphate, oxygen, primary production, mixed-layer depth) |
| `izmir_bgc_all.csv`, `izmir_chl_timeseries.csv` | Earlier extractions starting in 2000 |
| `temperature_climatology_izmir.csv`, `temperature_surface_izmir.csv` | Temperature-only extractions |
| `temperature_regional_monthly_stats.csv` | Regional monthly temperature statistics |

Where `station_climatology.csv` and `station_climatology_point.csv` differ, the difference is
single cell versus area average; the paper uses the single cell throughout so that temperature,
nutrients and biology all come from the same water.
