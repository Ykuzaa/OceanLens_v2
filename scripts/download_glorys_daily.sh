#!/bin/bash
# Download GLORYS12V1 daily surface fields with Copernicus Marine Toolbox.

set -euo pipefail

OUT_DIR="${OUT_DIR:-/scratch/emboulaalam/data/glorys/raw_daily}"
DATASET_ID="${DATASET_ID:-cmems_mod_glo_phy_my_0.083deg_P1D-m}"

START_DATE="${START_DATE:-1994-01-01}"
END_DATE="${END_DATE:-2004-12-31}"

LON_MIN="${LON_MIN:--20}"
LON_MAX="${LON_MAX:-20}"
LAT_MIN="${LAT_MIN:-30}"
LAT_MAX="${LAT_MAX:-60}"

mkdir -p "${OUT_DIR}"

end_exclusive="$(date -I -d "${END_DATE} + 1 day")"
current="${START_DATE}"
while [[ "${current}" < "${end_exclusive}" ]]; do
  year="$(date -d "${current}" +%Y)"
  mkdir -p "${OUT_DIR}/${year}"

  output_file="glorys_${current}.nc"
  output_path="${OUT_DIR}/${year}/${output_file}"

  if [[ -f "${output_path}" ]]; then
    echo "Skip existing ${output_path}"
  else
    echo "Download ${current} -> ${output_path}"
    copernicusmarine subset \
      --dataset-id "${DATASET_ID}" \
      --variable thetao \
      --variable so \
      --variable zos \
      --variable uo \
      --variable vo \
      --minimum-longitude "${LON_MIN}" \
      --maximum-longitude "${LON_MAX}" \
      --minimum-latitude "${LAT_MIN}" \
      --maximum-latitude "${LAT_MAX}" \
      --minimum-depth 0 \
      --maximum-depth 1 \
      --start-datetime "${current}" \
      --end-datetime "${current}" \
      --output-directory "${OUT_DIR}/${year}" \
      --output-filename "${output_file}" \
      --overwrite
  fi

  current="$(date -I -d "${current} + 1 day")"
done
