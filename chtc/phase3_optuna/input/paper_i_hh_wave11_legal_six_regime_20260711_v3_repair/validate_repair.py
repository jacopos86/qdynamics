#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    repo_root = Path.cwd()
    repair_root = Path(__file__).resolve().parent
    manifest = json.loads((repair_root / "repair_manifest.json").read_text())
    checks: list[dict[str, object]] = []
    for record in manifest["required_runtime_provenance_inputs"]:
        path = repo_root / record["path"]
        actual = sha256(path) if path.is_file() else None
        checks.append(
            {
                "role": record["role"],
                "path": record["path"],
                "expected_sha256": record["sha256"],
                "actual_sha256": actual,
                "ok": actual == record["sha256"],
            }
        )
    submit = repo_root / manifest["submit_file"]
    submit_text = submit.read_text(encoding="utf-8")
    checks.append(
        {
            "path": manifest["submit_file"],
            "expected_sha256": manifest["submit_sha256"],
            "actual_sha256": sha256(submit),
            "ok": sha256(submit) == manifest["submit_sha256"],
        }
    )
    for record in manifest["required_runtime_provenance_inputs"]:
        checks.append(
            {
                "check": "runtime_provenance_is_transferred",
                "path": record["path"],
                "ok": record["path"] in submit_text,
            }
        )
    wave_path = repo_root / manifest["immutable_input_bundle"] / "wave_manifest.json"
    wave = json.loads(wave_path.read_text(encoding="utf-8"))
    checks.append(
        {
            "check": "scientific_contract_unchanged",
            "expected": manifest["scientific_contract_hash"],
            "actual": wave["scientific_contract_hash"],
            "ok": wave["scientific_contract_hash"]
            == manifest["scientific_contract_hash"],
        }
    )
    failed = [check for check in checks if not check["ok"]]
    payload = {
        "schema": "paper_i_hh_wave11_legal_chtc_operational_repair_validation_v1",
        "status": "pass" if not failed else "fail",
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
