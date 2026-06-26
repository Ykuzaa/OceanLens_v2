# Data

OceanLens is designed for ocean forecasting and super-resolution experiments on gridded ocean reanalysis or forecast fields.

The current workflow targets Copernicus Marine GLORYS daily fields.

## Preprocessing

```text
GLORYS HR 1/12 degree daily
  -> coarsen to 1.5 degree
  -> interpolate to 1/4 degree experimental LR
  -> save HR, LR, masks and stats to Zarr
```

The low-resolution tensor is represented on the target grid after degradation, so models can learn residual corrections on aligned grids.

## Local Layout

Raw NetCDF and processed Zarr stores are not committed to Git.

Suggested layout:

```text
data/
├── raw_daily/
│   └── YYYY/
│       └── glorys_YYYY-MM-DD.nc
└── processed/
    └── v1.zarr
```

## Notes

- Do not commit Copernicus credentials.
- Do not commit large NetCDF/Zarr datasets.
- Keep data splits and preprocessing configs versioned.
