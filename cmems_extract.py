"""
CMEMS single-cell climatology extraction (Section 8 calibration input).

Reproduces data/station_climatology_point.csv from the Copernicus Marine Service
Mediterranean reanalysis .nc files (publicly available; see manuscript Data
availability). All variables are extracted at the SAME grid cell and surface layer
for internal consistency: 38.4792 N, 26.3750 E (nearest cell to 38.48 N/26.38 E).

Usage:  python cmems_extract.py  /path/to/phy-temp.nc  /path/to/plankton.nc  /path/to/nut.nc
Requires: xarray, numpy, pandas
"""
import sys, numpy as np, pandas as pd, xarray as xr

CENTER_LAT, CENTER_LON = 38.4792, 26.3750   # realized reference cell

def monthly_climatology(da):
    a = da.isel(depth=0).values                 # (time, lat, lon)
    nyr = a.shape[0] // 12
    return np.nanmean(a[:nyr*12].reshape(nyr, 12, *a.shape[1:]), axis=0)  # (12, lat, lon)

def nearest(coord, val):
    return int(np.argmin(np.abs(coord - val)))

def main(phy_nc, pla_nc, nut_nc, out='station_climatology_point.csv'):
    phy = xr.open_dataset(phy_nc); pla = xr.open_dataset(pla_nc); nut = xr.open_dataset(nut_nc)
    lat = phy['latitude'].values; lon = phy['longitude'].values
    i, j = nearest(lat, CENTER_LAT), nearest(lon, CENTER_LON)
    sst  = monthly_climatology(phy['thetao'])[:, i, j]
    chl  = monthly_climatology(pla['chl'])[:, i, j]
    phyc = monthly_climatology(pla['phyc'])[:, i, j]
    no3  = monthly_climatology(nut['no3'])[:, i, j]
    # SST monthly maximum (across years) for reference
    smax = np.nanmax(phy['thetao'].isel(depth=0).values.reshape(-1,12,*phy['thetao'].isel(depth=0).shape[1:])[:,:, i, j], axis=0)
    df = pd.DataFrame({'month': np.arange(1,13), 'sst': sst, 'sst_max': smax,
                       'chl': chl, 'no3': no3, 'phyc': phyc})
    df.to_csv(out, index=False)
    print("wrote %s at cell lat=%.4f lon=%.4f (idx %d,%d)" % (out, lat[i], lon[j], i, j))
    return df

if __name__ == '__main__':
    if len(sys.argv) >= 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
