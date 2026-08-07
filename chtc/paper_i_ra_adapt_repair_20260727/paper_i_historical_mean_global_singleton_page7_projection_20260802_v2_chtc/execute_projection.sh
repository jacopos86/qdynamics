#!/usr/bin/env bash
set -euo pipefail

export PYTHONDONTWRITEBYTECODE=1

if [[ "$#" -ne 9 ]]; then
  echo "usage: $0 EXECUTION_ID REGIME SOURCE_PROC ARCHIVE ARCHIVE_SHA ARCHIVE_SIZE SOURCE_ARCHIVE IMAGE IJSON_VENDOR" >&2
  exit 64
fi

execution_id="$1"
regime="$2"
source_proc="$3"
archive_path="$4"
expected_archive_sha256="$5"
expected_archive_size_bytes="$6"
source_archive="$7"
image_path="$8"
ijson_vendor="$9"

validator="chtc/paper_i_ra_adapt_repair_20260727/validate_nph3_v3_attempt_archive.py"
builder="pipelines/reporting/build_paper_i_historical_mean_global_singleton_ra_projection.py"
append_adapter="output/pdf/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving/paper_i_ra_adapt_stationary_core_full48_r50_20260728_evolving_append_singleton_r70_all6_adapter.json"
transfer_dir="transfer"
validation_output="${transfer_dir}/${regime}__cluster_9400779__proc_${source_proc}.archive_validation.json"
projection_output="${transfer_dir}/${regime}__cluster_9400779__proc_${source_proc}.page7_projection.json"
remote_archive_path="/home/jsstrobel/Holstein_phase3_optuna_chtc/${archive_path}"

case "$regime" in
  weak_weak|intermediate_weak|strong_weak_u8) ;;
  *) echo "unsupported nph=3 regime" >&2; exit 65 ;;
esac
if [[ ! "$source_proc" =~ ^[0-2]$ \
   || ! "$expected_archive_sha256" =~ ^[0-9a-f]{64}$ \
   || ! "$expected_archive_size_bytes" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid source archive binding" >&2
  exit 65
fi
case "$archive_path" in
  /*|*".."*|*"//"*) echo "unsafe archive path" >&2; exit 65 ;;
esac

for required in \
  "$archive_path" \
  "$source_archive" \
  "$image_path" \
  "$ijson_vendor" \
  "$validator" \
  "$builder" \
  "$append_adapter"
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or unsafe projection input: $required" >&2
    exit 66
  fi
done

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

if [[ "$(hash_file "$ijson_vendor")" != "820030b23d1e2c6a37d1ccc02d44c313c3305396bf30c1fdfecab3273a98733e" ]]; then
  echo "pure-Python ijson vendor SHA-256 mismatch" >&2
  exit 67
fi
if [[ "$(stat -c '%s' "$archive_path")" != "$expected_archive_size_bytes" ]]; then
  echo "source archive size mismatch" >&2
  exit 67
fi

runtime_bin="$(command -v apptainer || command -v singularity || true)"
if [[ -z "$runtime_bin" ]]; then
  echo "apptainer/singularity runtime unavailable" >&2
  exit 68
fi

# The source lock supplies the scientific runtime.  The transferred reporting
# files are an additive, hash-bound projection layer and are not in that source
# archive, so extraction cannot overwrite them.
tar -xzf "$source_archive"
mkdir vendor
tar -xzf "$ijson_vendor" -C vendor
mkdir "$transfer_dir"
sandbox_root="$(pwd -P)"

"$runtime_bin" exec \
  --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env IJSON_BACKEND=python \
  --env "PYTHONPATH=${sandbox_root}/vendor:${sandbox_root}" \
  --bind "${sandbox_root}:${sandbox_root}" \
  --pwd "$sandbox_root" \
  "$image_path" \
  python3 "$validator" \
    --archive "$archive_path" \
    --execution-id "$execution_id" \
    --proc-id "$source_proc" \
  >"$validation_output"

"$runtime_bin" exec \
  --cleanenv \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env IJSON_BACKEND=python \
  --env "PYTHONPATH=${sandbox_root}/vendor:${sandbox_root}" \
  --bind "${sandbox_root}:${sandbox_root}" \
  --pwd "$sandbox_root" \
  "$image_path" \
  python3 "$builder" \
    --append-adapter "$append_adapter" \
    --regime "$regime" \
    --archive "$archive_path" \
    --archive-validation "$validation_output" \
    --remote-archive-path "$remote_archive_path" \
    --remote-archive-sha256 "$expected_archive_sha256" \
    --remote-archive-size-bytes "$expected_archive_size_bytes" \
    --output "$projection_output"

for output in "$validation_output" "$projection_output"; do
  if [[ ! -s "$output" || -L "$output" ]]; then
    echo "projection output is missing or unsafe: $output" >&2
    exit 69
  fi
done

printf '{"archive_sha256":"%s","archive_size_bytes":%s,"projection_sha256":"%s","regime_id":"%s","status":"passed","validation_sha256":"%s"}\n' \
  "$expected_archive_sha256" \
  "$expected_archive_size_bytes" \
  "$(hash_file "$projection_output")" \
  "$regime" \
  "$(hash_file "$validation_output")"
