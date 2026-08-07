#!/usr/bin/env bash
set -euo pipefail

job_manifest="$1"
source_archive="$2"
expected_source_sha="$3"
image_path="$4"
expected_image_sha="$5"
job_id="$6"

actual_source_sha="$(sha256sum "${source_archive}" | awk '{print $1}')"
test "${actual_source_sha}" = "${expected_source_sha}"
actual_image_sha="$(sha256sum "${image_path}" | awk '{print $1}')"
test "${actual_image_sha}" = "${expected_image_sha}"

rm -rf source_locked
mkdir -p source_locked
tar -xzf "${source_archive}" -C source_locked
output_dir="raw_outputs/paper_i_hh_geo_comparators_macro_projected_singleton_all_six_r50_20260718_v1_chtc/${job_id}"
mkdir -p "${output_dir}"

apptainer exec \
  --bind "${PWD}:${PWD}" \
  --pwd "${PWD}" \
  --env "PYTHONPATH=${PWD}/source_locked" \
  "${image_path}" \
  python run_job.py "${job_manifest}" "${output_dir}"

tar -czf "raw_outputs/paper_i_hh_geo_comparators_macro_projected_singleton_all_six_r50_20260718_v1_chtc/${job_id}_transfer.tar.gz" \
  -C "raw_outputs/paper_i_hh_geo_comparators_macro_projected_singleton_all_six_r50_20260718_v1_chtc" \
  "${job_id}"
