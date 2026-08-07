#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 13 ]]; then
  echo "usage: $0 JOB_SPEC JOB_SPEC_SHA256 SOURCE_ARCHIVE SOURCE_SHA256 IMAGE IMAGE_SHA256 PACKAGE_MANIFEST AUTHORIZATION AUTHORIZATION_SHA256 V7_FINAL OBJECTIVE_GATE_AUTHORITY EXECUTION_PLAN OUTPUT_ARCHIVE" >&2
  exit 64
fi

job_spec="$1"
expected_job_spec_sha256="$2"
source_archive="$3"
expected_source_sha256="$4"
image_path="$5"
expected_image_sha256="$6"
package_manifest="$7"
authorization_receipt="$8"
expected_authorization_sha256="$9"
v7_final_receipt="${10}"
objective_gate_authority="${11}"
execution_plan="${12}"
output_archive="${13}"

hash_file() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

for required in \
  "$job_spec" \
  "$source_archive" \
  "$image_path" \
  "$package_manifest" \
  "$authorization_receipt" \
  "$v7_final_receipt" \
  "$objective_gate_authority" \
  "$execution_plan"
do
  if [[ ! -f "$required" || -L "$required" ]]; then
    echo "missing or symlinked required input: $required" >&2
    exit 65
  fi
done

actual_source_sha256="$(hash_file "$source_archive")"
if [[ "$actual_source_sha256" != "$expected_source_sha256" ]]; then
  echo "source archive SHA-256 mismatch" >&2
  exit 66
fi
actual_job_spec_sha256="$(hash_file "$job_spec")"
if [[ "$actual_job_spec_sha256" != "$expected_job_spec_sha256" ]]; then
  echo "job-spec SHA-256 mismatch" >&2
  exit 66
fi
actual_authorization_sha256="$(hash_file "$authorization_receipt")"
if [[ "$actual_authorization_sha256" != "$expected_authorization_sha256" ]]; then
  echo "authorization-receipt SHA-256 mismatch" >&2
  exit 66
fi
actual_image_sha256="$(hash_file "$image_path")"
if [[ "$actual_image_sha256" != "$expected_image_sha256" ]]; then
  echo "execution image SHA-256 mismatch" >&2
  exit 67
fi

package_dir="$(dirname "$package_manifest")"
verify_authorization_bound_control_plane() {
python3 - "$package_dir" "$authorization_receipt" "$1" <<'PY'
import hashlib
import json
import pathlib
import sys

package_dir = pathlib.Path(sys.argv[1])
authorization_path = pathlib.Path(sys.argv[2])
wrapper_source = pathlib.Path(sys.argv[3])
files = (
    "build_attempt_selection.py",
    "build_package.py",
    "package_contract.py",
    "run_cell.py",
    "execute_source_locked_job.sh",
    "run_scientific_preflight_smokes.py",
    "stage_transferred_executable.py",
    "submit.sub",
    "validate_package.py",
    "validate_fetched.py",
    "link_shared_append.py",
    "objective_gates.py",
)

def canonical_sha256(value):
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

rows = []
for relative in files:
    path = (
        wrapper_source
        if relative == "execute_source_locked_job.sh"
        else package_dir / relative
    )
    if not path.is_file() or path.is_symlink():
        raise SystemExit(
            f"authorization-bound control-plane file is unsafe: {relative}"
        )
    rows.append(
        {
            "path": relative,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_bytes": path.stat().st_size,
        }
    )
control = {
    "schema": "paper_i_ra_adapt_study1_package_control_plane_v2",
    "package_id": "paper_i_ra_adapt_study1_minimal_20260728_v2_chtc",
    "files": rows,
    "file_count": len(rows),
    "all_files_verified": True,
}
control_sha256 = canonical_sha256(control)
authorization = json.loads(
    authorization_path.read_text(encoding="utf-8")
)
authorization_unsigned = dict(authorization)
authorization_sha256 = authorization_unsigned.pop("sha256", None)
if authorization_sha256 != canonical_sha256(authorization_unsigned):
    raise SystemExit("authorization self digest does not match")
if authorization.get("package_control_plane_sha256") != control_sha256:
    raise SystemExit("authorization-bound package control plane drifted")
PY
}

# HTCondor may rename transfer_executable.  Verify the complete authorization-
# bound aggregate first, substituting the actual $0 bytes only for the wrapper
# row.  The now-authenticated helper may then stage those exact bytes at the
# canonical package path.  Recompute the same aggregate from that final layout.
verify_authorization_bound_control_plane "$0"
python3 "$package_dir/stage_transferred_executable.py" \
  --wrapper-source "$0" \
  --package-dir "$package_dir" >/dev/null
verify_authorization_bound_control_plane \
  "$package_dir/execute_source_locked_job.sh"

scratch_root="$(pwd -P)"
python3 - "$source_archive" "$scratch_root" <<'PY'
import pathlib
import shutil
import sys
import tarfile

archive_path = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()
with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()
    if not members:
        raise SystemExit("source archive is empty")
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        if (
            name.is_absolute()
            or ".." in name.parts
            or "." in name.parts
            or not member.isfile()
            or member.issym()
            or member.islnk()
        ):
            raise SystemExit(f"unsafe source archive member: {member.name}")
    for member in members:
        name = pathlib.PurePosixPath(member.name)
        target = destination.joinpath(*name.parts)
        if target.exists():
            raise SystemExit(f"source member would overwrite a file: {member.name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source = archive.extractfile(member)
        if source is None:
            raise SystemExit(f"cannot read source member: {member.name}")
        with source, target.open("xb") as output:
            shutil.copyfileobj(source, output)
        target.chmod(member.mode & 0o777)
PY

case "$output_archive" in
  /*|*".."*|*"//"*)
    echo "unsafe output archive path: $output_archive" >&2
    exit 68
    ;;
esac
if [[ -e "$output_archive" ]]; then
  echo "refusing to overwrite output archive: $output_archive" >&2
  exit 69
fi

run_cell="${package_dir}/run_cell.py"
if [[ ! -f "$run_cell" || -L "$run_cell" ]]; then
  echo "packaged worker is unavailable: $run_cell" >&2
  exit 70
fi

package_job_outputs() {
  local worker_status="$1"
  python3 - \
    "$job_spec" \
    "$scratch_root" \
    "$output_archive" \
    "$worker_status" <<'PY'
import gzip
import json
import os
import pathlib
import sys
import tarfile

job_path = pathlib.Path(sys.argv[1])
root = pathlib.Path(sys.argv[2]).resolve()
destination = pathlib.Path(sys.argv[3])
worker_status = int(sys.argv[4])
job = json.loads(job_path.read_text(encoding="utf-8"))

names = [
    str(job["artifact_paths"][role])
    for role in (
        "execution_manifest",
        "checkpoint",
        "estimator_ledger",
        "result",
        "summary",
    )
]
names.append(str(job["worker_receipt_path"]))
members = []
for raw in names:
    name = pathlib.PurePosixPath(raw)
    if name.is_absolute() or ".." in name.parts or "." in name.parts:
        raise SystemExit(f"unsafe output member: {raw}")
    source = (root / pathlib.Path(*name.parts)).resolve()
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise SystemExit(f"output member escapes scratch: {raw}") from exc
    if source.exists():
        if not source.is_file() or source.is_symlink():
            raise SystemExit(f"output member is not a regular file: {raw}")
        members.append((name.as_posix(), source))

if worker_status == 0 and len(members) != 6:
    raise SystemExit(
        f"successful worker produced {len(members)} of 6 narrow outputs"
    )
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_name(f".{destination.name}.tmp")
if temporary.exists():
    raise SystemExit(f"stale output temporary exists: {temporary}")
with temporary.open("xb") as raw:
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(
            mode="w", fileobj=gz, format=tarfile.PAX_FORMAT
        ) as archive:
            for name, source in sorted(members):
                info = tarfile.TarInfo(name)
                info.size = source.stat().st_size
                info.mode = 0o644
                info.uid = 0
                info.gid = 0
                info.uname = ""
                info.gname = ""
                info.mtime = 0
                with source.open("rb") as stream:
                    archive.addfile(info, stream)
    raw.flush()
    os.fsync(raw.fileno())
temporary.replace(destination)
PY
}

worker_status=1
trap 'saved_status=$?; package_job_outputs "$worker_status" || true; exit "$saved_status"' EXIT

container_runtime=""
if command -v apptainer >/dev/null 2>&1; then
  container_runtime="apptainer"
elif command -v singularity >/dev/null 2>&1; then
  container_runtime="singularity"
else
  echo "apptainer/singularity is unavailable" >&2
  exit 71
fi

"$container_runtime" exec \
  --cleanenv \
  --bind "${scratch_root}:/work" \
  "$image_path" \
  /bin/bash -lc '
    set -euo pipefail
    cd /work
    export PYTHONNOUSERSITE=1
    export PYTHONDONTWRITEBYTECODE=1
    export PYTHONHASHSEED=0
    export OMP_NUM_THREADS=4
    export OPENBLAS_NUM_THREADS=4
    export MKL_NUM_THREADS=4
    export VECLIB_MAXIMUM_THREADS=4
    export NUMEXPR_NUM_THREADS=4
    exec python3 "$1" \
      --mode execute \
      --source-root /work \
      --job-spec "$2" \
      --package-manifest "$3" \
      --authorization-receipt "$4" \
      --v7-final-receipt "$5" \
      --objective-gate-authority "$6" \
      --execution-plan "$7" \
      --source-archive-sha256 "$8" \
      --verified-image-sha256 "$9"
  ' _ \
  "$run_cell" \
  "$job_spec" \
  "$package_manifest" \
  "$authorization_receipt" \
  "$v7_final_receipt" \
  "$objective_gate_authority" \
  "$execution_plan" \
  "$actual_source_sha256" \
  "$actual_image_sha256"

worker_status=0
package_job_outputs "$worker_status"
trap - EXIT
