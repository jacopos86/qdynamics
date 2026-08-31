#!/usr/bin/env python3
"""Materialize the AAVQE r50->r75 continuation package (six cells)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]
ANCHOR_ROOT = Path(
    "/Users/jakestrobel/Documents/Holstein_implementation/Holstein_test_fullclone_3/chtc/"
    "paper_i_ra_adapt_repair_20260727/"
    "paper_i_append_adapt_stationary_singleton6_r50_powell_tolmatch_20260816_v1_chtc_local_runtime/"
    "20260816T180727Z"
)
JOB_SCHEMA = "paper_i_aavqe_r75_continuation_job_v1"
REGIMES = (
    ("weak_weak", 3),
    ("intermediate_weak", 3),
    ("strong_weak_u8", 3),
    ("weak_strong", 7),
    ("intermediate_strong", 7),
    ("strong_strong_u8", 7),
)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    body = {k: v for k, v in payload.items() if k != "sha256"}
    return hashlib.sha256(
        (json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n").encode()
    ).hexdigest()


def digested(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result["sha256"] = canonical_sha256(result)
    return result


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_archive(path: Path) -> str:
    members = []
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for root_name in ("docs", "pipelines", "src"):
            root = REPO_ROOT / root_name
            for source in sorted(root.rglob("*")):
                if (
                    not source.is_file()
                    or source.is_symlink()
                    or "__pycache__" in source.parts
                    or source.suffix in {".pyc", ".pyo"}
                ):
                    continue
                data = source.read_bytes()
                info = tarfile.TarInfo(source.relative_to(REPO_ROOT).as_posix())
                info.size = len(data)
                info.mode = 0o755 if data.startswith(b"#!") else 0o644
                info.mtime = 0
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                archive.addfile(info, io.BytesIO(data))
                members.append(info.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0, compresslevel=6) as zipped:
            zipped.write(buffer.getvalue())
    return file_sha256(path)


def main() -> int:
    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    source_sha = build_source_archive(PACKAGE_DIR / "source/source_locked.tar.gz")
    queue_rows = []
    jobs = []
    for regime, nph in REGIMES:
        cell_dir = None
        for candidate in ANCHOR_ROOT.iterdir():
            if candidate.name.startswith(f"core__{regime}__nph"):
                cell_dir = candidate
                break
        if cell_dir is None:
            raise RuntimeError(f"anchor missing for {regime}")
        payload_dir = cell_dir / "payload"
        anchor_src = payload_dir / "checkpoint.json"
        summary = json.loads((payload_dir / "summary.json").read_text())
        expected_energy = float(summary["accepted_history"][-1]["energy_after"])
        eid = f"aavqe_r75cont__{regime}__nph{nph}"
        anchor_rel = f"anchors/{eid}.checkpoint.json"
        anchor_dst = PACKAGE_DIR / anchor_rel
        anchor_dst.parent.mkdir(parents=True, exist_ok=True)
        anchor_dst.write_bytes(anchor_src.read_bytes())
        import gzip as _gzip
        anchor_raw = json.loads(
            _gzip.decompress((payload_dir / "result.json.gz").read_bytes())
        )
        history_rel = f"anchors/{eid}.history.json"
        (PACKAGE_DIR / history_rel).write_text(
            json.dumps(
                anchor_raw["result_payload"]["history"],
                sort_keys=True,
                default=str,
            )
        )
        job = digested(
            {
                "schema": JOB_SCHEMA,
                "execution_id": eid,
                "regime_id": regime,
                "nph": nph,
                "target_horizon": 75,
                "anchor_path": anchor_rel,
                "anchor_sha256": file_sha256(anchor_dst),
                "anchor_history_path": history_rel,
                "anchor_history_sha256": file_sha256(
                    PACKAGE_DIR / history_rel
                ),
                "anchor_source": str(anchor_src),
                "expected_prefix_energy": expected_energy,
                "energy_reproduction_tolerance": 1.0e-14,
                "source_archive_sha256": source_sha,
                "execution_authorized": True,
                "authority_basis": (
                    "explicit_user_order_20260819_true_continuation_"
                    "from_iteration_50_on_chtc"
                ),
            }
        )
        job_rel = f"jobs/{eid}.json"
        (PACKAGE_DIR / job_rel).parent.mkdir(parents=True, exist_ok=True)
        (PACKAGE_DIR / job_rel).write_text(
            json.dumps(job, indent=2, sort_keys=True) + "\n"
        )
        jobs.append({"execution_id": eid, "sha256": job["sha256"]})
        memory = 8192 if nph == 3 else 16384
        queue_rows.append(
            f"{eid}\t{job_rel}\tjobs/{eid}.json\t1\t{memory}\t20480\t172800"
        )
    (PACKAGE_DIR / "queue.tsv").write_text("\n".join(queue_rows) + "\n")
    manifest = digested(
        {
            "schema": "paper_i_aavqe_r75_continuation_package_v1",
            "created_at": created_at,
            "source_archive_sha256": source_sha,
            "worker_file_sha256": file_sha256(PACKAGE_DIR / "worker.py"),
            "execute_file_sha256": file_sha256(PACKAGE_DIR / "execute_job.sh"),
            "jobs": jobs,
        }
    )
    (PACKAGE_DIR / "package_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"cells": len(jobs), "source_sha256": source_sha}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
