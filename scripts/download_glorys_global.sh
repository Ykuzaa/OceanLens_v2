#!/bin/bash
# Download the global GLORYS12V1 daily surface dataset required by OceanLens_v2.
#
# LIR note: Copernicus Marine needs Internet access. According to the LIR wiki,
# Internet access is available from the frontal node, not from regular compute
# nodes. Run this script from lir-frontal with nohup/tmux/screen, not sbatch.

set -euo pipefail

export OUT_DIR="${OUT_DIR:-/scratch/emboulaalam/data/glorys/raw_daily}"
export DATASET_ID="${DATASET_ID:-cmems_mod_glo_phy_my_0.083deg_P1D-m}"
export LON_MIN="${LON_MIN:--180}"
export LON_MAX="${LON_MAX:-179.92}"
export LAT_MIN="${LAT_MIN:--80}"
export LAT_MAX="${LAT_MAX:-90}"

mkdir -p logs

echo "Global GLORYS download"
echo "OUT_DIR=${OUT_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "BBOX lon=[${LON_MIN}, ${LON_MAX}] lat=[${LAT_MIN}, ${LAT_MAX}]"

echo "Downloading train/validation years: 1994-01-01 to 2004-12-31"
START_DATE=1994-01-01 END_DATE=2004-12-31 bash scripts/download_glorys_daily.sh

echo "Downloading test year: 2019-01-01 to 2019-12-31"
START_DATE=2019-01-01 END_DATE=2019-12-31 bash scripts/download_glorys_daily.sh

echo "Done."
