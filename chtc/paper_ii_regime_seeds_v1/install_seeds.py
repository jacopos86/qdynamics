#!/usr/bin/env python3
"""Install fetched stage-1 regime seeds under the registered registry paths.

``paper_ii_runs.REGIMES`` resolves every regime row to
``chtc/paper_ii_regime_seeds_v1/input/seeds/<regime_id>.json``.  This script is
the only bridge from a fetched CHTC output tree
(``<fetched>/raw_outputs/paper_ii_regime_seeds_v1/<regime_id>/``) to those
paths.  It refuses anything the receipt does not vouch for: the receipt must be
sealed, ``status == "qualified"``, the artifact bytes must hash to the recorded
digest, the physics and optimizer seed must match ``input/regimes.tsv``, and the
gate must be the frozen ``1e-6``.  Qualified seeds are installed; missing or
unqualified rows are reported and left absent so ``resolve_regime`` keeps
raising ``SeedNotBuiltError`` for them.  Exit 0 only when every requested row is
installed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BUNDLE = Path("chtc/paper_ii_regime_seeds_v1")
RAW_SUBDIR = Path("raw_outputs/paper_ii_regime_seeds_v1")
RECEIPT_SCHEMA = "paper_ii_regime_seed_receipt_v1"
ARTIFACT_SCHEMA = "paper_ii_regime_seed_artifact_v1"
ENERGY_GATE = 1.0e-6
INSTALL_RECEIPT_SCHEMA = "paper_ii_regime_seed_install_receipt_v1"


def _canonical_sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regime_rows(regimes_tsv: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with regimes_tsv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows[row["regime_id"]] = {
                "u": float(row["u"]),
                "g_ep": float(row["g_ep"]),
                "nph": int(row["nph"]),
                "dv": float(row.get("dv", 0.0)),
                "optimizer_seed": int(row["optimizer_seed"]),
            }
    return rows


def verify_receipt(
    receipt: Mapping[str, Any], *, regime_id: str, row: Mapping[str, Any], raw_dir: Path
) -> tuple[Path, dict[str, Any]]:
    """Return the verified artifact path or raise ValueError with the reason."""

    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError(f"receipt schema is {receipt.get('schema')!r}")
    core = {k: v for k, v in receipt.items() if k != "seed_receipt_sha256"}
    if receipt.get("seed_receipt_sha256") != _canonical_sha(core):
        raise ValueError("receipt seal does not match its content")
    if receipt.get("regime_id") != regime_id:
        raise ValueError(f"receipt names regime {receipt.get('regime_id')!r}")
    if receipt.get("status") != "qualified":
        raise ValueError(f"status={receipt.get('status')!r}: {receipt.get('failure')}")
    if float(receipt.get("energy_gate", -1.0)) != ENERGY_GATE:
        raise ValueError(f"energy gate {receipt.get('energy_gate')} is not {ENERGY_GATE}")
    error = float(receipt["energy_error"])
    if not error <= ENERGY_GATE:
        raise ValueError(f"energy_error {error} exceeds the gate")
    physical = receipt.get("physical") or {}
    expected = {
        "U": row["u"], "g": row["g_ep"], "n_ph_max": row["nph"],
        "staggered_potential": row["dv"],
    }
    actual = {
        "U": float(physical.get("U", float("nan"))),
        "g": float(physical.get("g", float("nan"))),
        "n_ph_max": int(physical.get("n_ph_max", -1)),
        "staggered_potential": float(physical.get("staggered_potential", float("nan"))),
    }
    if actual != expected:
        raise ValueError(f"receipt physics {actual} != regimes.tsv {expected}")
    if int(receipt.get("optimizer_seed", -1)) != int(row["optimizer_seed"]):
        raise ValueError("optimizer seed differs from regimes.tsv")
    artifact_row = receipt.get("artifact") or {}
    artifact = raw_dir / Path(str(artifact_row.get("path", ""))).name
    if not artifact.is_file():
        raise ValueError(f"artifact missing beside receipt: {artifact}")
    if _file_sha(artifact) != artifact_row.get("sha256"):
        raise ValueError("artifact bytes do not match the receipt digest")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if payload.get("schema") != ARTIFACT_SCHEMA:
        raise ValueError(f"artifact schema is {payload.get('schema')!r}")
    core = {k: v for k, v in payload.items() if k != "seed_artifact_sha256"}
    if payload.get("seed_artifact_sha256") != _canonical_sha(core):
        raise ValueError("artifact seal does not match its content")
    if payload.get("seed_artifact_sha256") != artifact_row.get("seed_artifact_sha256"):
        raise ValueError("artifact seal differs from the receipt")
    return artifact, {
        "sha256": artifact_row["sha256"],
        "seed_artifact_sha256": payload["seed_artifact_sha256"],
        "energy_error": error,
        "exact_energy": receipt.get("exact_energy"),
        "num_parameters": receipt.get("num_parameters"),
        "code_revision": receipt.get("code_revision"),
        "wall_seconds": receipt.get("wall_seconds"),
    }


def install(
    *,
    fetched_root: Path,
    regime_ids: Sequence[str],
    regimes_tsv: Path,
    dest: Path,
) -> dict[str, Any]:
    rows = read_regime_rows(regimes_tsv)
    installed: dict[str, Any] = {}
    problems: dict[str, str] = {}
    for regime_id in regime_ids:
        if regime_id not in rows:
            problems[regime_id] = "not in regimes.tsv"
            continue
        raw_dir = fetched_root / RAW_SUBDIR / regime_id
        receipt_path = raw_dir / "seed_receipt.json"
        if not receipt_path.is_file():
            problems[regime_id] = f"no receipt at {receipt_path}"
            continue
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            artifact, summary = verify_receipt(
                receipt, regime_id=regime_id, row=rows[regime_id], raw_dir=raw_dir
            )
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            problems[regime_id] = str(exc)
            continue
        target = dest / f"{regime_id}.json"
        if target.exists() and _file_sha(target) != summary["sha256"]:
            problems[regime_id] = f"refusing to overwrite a different seed at {target}"
            continue
        dest.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(artifact, target)
        installed[regime_id] = {
            **summary,
            "installed_path": str(target),
            "source_receipt": str(receipt_path),
        }
    receipt = {
        "schema": INSTALL_RECEIPT_SCHEMA,
        "installed_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "fetched_root": str(fetched_root),
        "regimes_tsv_sha256": _file_sha(regimes_tsv),
        "installed": installed,
        "problems": problems,
    }
    if installed:
        dest.mkdir(parents=True, exist_ok=True)
        sums = dest / "SHA256SUMS"
        lines = {}
        if sums.exists():
            for line in sums.read_text(encoding="utf-8").splitlines():
                digest, _, name = line.partition("  ")
                if name:
                    lines[name] = digest
        for regime_id, row in installed.items():
            lines[f"{regime_id}.json"] = row["sha256"]
        sums.write_text(
            "".join(f"{lines[name]}  {name}\n" for name in sorted(lines)),
            encoding="utf-8",
        )
        (dest / "install_receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fetched-root", required=True, type=Path,
                        help="directory holding raw_outputs/paper_ii_regime_seeds_v1/")
    parser.add_argument("--regime-ids", type=Path, default=BUNDLE / "input" / "regime_ids.txt")
    parser.add_argument("--regimes-tsv", type=Path, default=BUNDLE / "input" / "regimes.tsv")
    parser.add_argument("--dest", type=Path, default=BUNDLE / "input" / "seeds")
    parser.add_argument("--only", action="append", default=[],
                        help="install only this regime id; repeatable")
    args = parser.parse_args(argv)
    regime_ids = [
        line.strip() for line in args.regime_ids.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.only:
        regime_ids = [r for r in regime_ids if r in set(args.only)]
    receipt = install(
        fetched_root=args.fetched_root,
        regime_ids=regime_ids,
        regimes_tsv=args.regimes_tsv,
        dest=args.dest,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if not receipt["problems"] else 2


if __name__ == "__main__":
    sys.exit(main())
