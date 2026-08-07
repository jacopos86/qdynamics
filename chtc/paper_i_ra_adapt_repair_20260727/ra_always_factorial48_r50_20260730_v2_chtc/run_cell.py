#!/usr/bin/env python3
"""Execute one factorial cell only with explicit external authorization."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["STATIC_ADAPT_HH_POOL_CACHE"] = "off"
os.environ["STATIC_ADAPT_CANDIDATE_RECORD_CACHE"] = "off"
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from package_contract import (  # noqa: E402
    JOB_SCHEMA,
    PACKAGE_ID,
    SOURCE_ARCHIVE_MANIFEST_NAME,
    SOURCE_ARCHIVE_NAME,
    PackageContractError,
    canonical_json_bytes,
    load_json,
    safe_relative_path,
    sha256_file,
    verify_self_digest,
)


AUTHORIZATION_SCHEMA = (
    "paper_i_ra_always_factorial_execution_authorization_v1"
)


def _validate_authorization(
    path: Path,
    *,
    execution_id: str,
    job_sha256: str,
) -> dict[str, Any]:
    authority = load_json(path, label="external execution authorization")
    verify_self_digest(authority, label="external execution authorization")
    if (
        authority.get("schema") != AUTHORIZATION_SCHEMA
        or authority.get("package_id") != PACKAGE_ID
        or authority.get("execution_id") != execution_id
        or authority.get("job_sha256") != job_sha256
        or authority.get("execution_authorized") is not True
        or authority.get("submission_authorized") is not True
    ):
        raise PackageContractError(
            "External execution authorization does not bind this job."
        )
    return authority


def _extract_source_archive(destination: Path) -> None:
    manifest = load_json(
        PACKAGE_DIR / SOURCE_ARCHIVE_MANIFEST_NAME,
        label="source archive manifest",
    )
    verify_self_digest(manifest, label="source archive manifest")
    archive_binding = manifest.get("archive")
    rows = manifest.get("members")
    archive_path = PACKAGE_DIR / SOURCE_ARCHIVE_NAME
    if (
        not isinstance(archive_binding, Mapping)
        or archive_binding.get("path") != SOURCE_ARCHIVE_NAME
        or sha256_file(archive_path) != archive_binding.get("sha256")
        or archive_path.stat().st_size
        != int(archive_binding.get("size_bytes", -1))
        or not isinstance(rows, list)
    ):
        raise PackageContractError("Source archive binding drifted.")
    declared = {
        safe_relative_path(row["path"], label="source member").as_posix():
        row
        for row in rows
        if isinstance(row, Mapping)
    }
    if len(declared) != len(rows):
        raise PackageContractError("Source archive member set duplicates.")
    destination.mkdir(parents=True, exist_ok=False)
    observed: set[str] = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive:
            relative = safe_relative_path(
                member.name, label="tar member"
            ).as_posix()
            if (
                relative not in declared
                or relative in observed
                or not member.isfile()
                or member.issym()
                or member.islnk()
            ):
                raise PackageContractError(
                    f"Unsafe source archive member: {relative}"
                )
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise PackageContractError(
                    f"Unreadable source archive member: {relative}"
                )
            with target.open("xb") as output:
                for block in iter(
                    lambda: stream.read(1024 * 1024), b""
                ):
                    output.write(block)
            row = declared[relative]
            if (
                sha256_file(target) != row["sha256"]
                or target.stat().st_size != int(row["size_bytes"])
            ):
                raise PackageContractError(
                    f"Extracted source member drifted: {relative}"
                )
            observed.add(relative)
    if observed != set(declared):
        raise PackageContractError("Extracted source member closure failed.")


def _activate_source_root(source_root: Path) -> None:
    root = source_root.resolve()
    for name in list(sys.modules):
        if (
            name == "pipelines"
            or name.startswith("pipelines.")
            or name == "src"
            or name.startswith("src.")
        ):
            del sys.modules[name]
    sys.path[:] = [
        item
        for item in sys.path
        if not (
            (Path(item or ".").resolve() / "pipelines").exists()
            or (Path(item or ".").resolve() / "src").exists()
        )
    ]
    sys.path.insert(0, root.as_posix())
    importlib.invalidate_caches()
    module = importlib.import_module("pipelines.static_adapt.ra_adapt")
    origin = Path(str(module.__file__)).resolve()
    try:
        origin.relative_to(root)
    except ValueError as exc:
        raise PackageContractError(
            "Runtime implementation escaped the source archive."
        ) from exc


def _problem_from_protocol(protocol: Any) -> Any:
    from pipelines.contracts.problem import ProblemRequest
    from pipelines.static_adapt.builders.problem_registry import (
        resolve_problem_context,
    )

    receipt = protocol.problem
    return resolve_problem_context(
        ProblemRequest(
            problem_key=str(receipt.problem_key),
            num_sites=int(receipt.num_sites),
            t=float(receipt.t),
            u=float(receipt.u),
            dv=float(receipt.dv),
            omega0=float(receipt.omega0),
            g_ep=float(receipt.g_ep),
            n_ph_max=int(receipt.n_ph_max),
            boson_encoding=str(receipt.boson_encoding),
            ordering=str(receipt.ordering),
            boundary=str(receipt.boundary),
            include_zero_point=bool(receipt.include_zero_point),
            v_nn=float(receipt.v_nn),
            t_prime=float(receipt.t_prime),
            n_fermions=(
                None
                if receipt.n_fermions is None
                else int(receipt.n_fermions)
            ),
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument(
        "--execution-authorization", type=Path, required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to overwrite output: {output}")
    job = load_json(args.job.resolve(), label="job")
    verify_self_digest(job, label="job")
    if (
        job.get("schema") != JOB_SCHEMA
        or job.get("package_id") != PACKAGE_ID
        or job.get("execution_authorized") is not False
        or job.get("submission_authorized") is not False
    ):
        raise PackageContractError("Job is not the inert factorial contract.")
    _validate_authorization(
        args.execution_authorization.resolve(),
        execution_id=str(job["execution_id"]),
        job_sha256=str(job["sha256"]),
    )

    with tempfile.TemporaryDirectory(
        prefix="paper-i-ra-always-factorial-source."
    ) as temporary_name:
        source_root = Path(temporary_name) / "source"
        _extract_source_archive(source_root)
        _activate_source_root(source_root)
        from pipelines.static_adapt.ra_adapt import run_ra_adapt
        from pipelines.static_adapt.ra_adapt.bundles import (
            load_validated_bundle_protocol,
        )

        binding = job["protocol"]
        protocol_path = source_root / safe_relative_path(
            binding["path"], label="protocol path"
        )
        if (
            sha256_file(protocol_path) != binding["sha256"]
            or protocol_path.stat().st_size
            != int(binding["size_bytes"])
        ):
            raise PackageContractError("Source-locked protocol drifted.")
        protocol = load_validated_bundle_protocol(protocol_path)
        if protocol.sha256 != binding["canonical_sha256"]:
            raise PackageContractError(
                "Source-locked protocol canonical digest drifted."
            )
        if (
            protocol.active_gradient_policy
            != job["active_gradient_policy"]
            or protocol.resource_weighting_scope
            != job["resource_weighting_scope"]
        ):
            raise PackageContractError(
                "Runtime protocol axes drifted from the job."
            )
        payload = run_ra_adapt(
            _problem_from_protocol(protocol), protocol
        ).to_dict()

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("xb") as stream:
        stream.write(canonical_json_bytes(payload) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, output)
    temporary.unlink()
    print(
        json.dumps(
            {
                "status": "passed",
                "execution_id": job["execution_id"],
                "output": output.as_posix(),
                "output_sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
