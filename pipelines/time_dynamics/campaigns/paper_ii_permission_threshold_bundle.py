"""Locks and CHTC packages for the pre-permission ray-threshold campaigns.

A bundle is the corrected exchange campaign's lock (``paper_ii_corrected_exchange_cells_v1``)
with ``--exchange-permission-ray-distance-max <literal>`` declared in its
``registry_bindings.extra_flags``, over either one cell or all eleven.  The
package is templated from the corrected campaign package
(``chtc/paper_ii_corrected_exchange_v1_a43bcd86_20260905``): the same
``run_action.py``, container launcher, submit file, and validator, with the
campaign identity, action count, source revision, and source-archive digest
rewritten, and the validator additionally requiring the declared threshold on
every action argv.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Sequence

from pipelines.time_dynamics.campaigns import (
    paper_ii_final_estimator_telemetry_v1 as telemetry,
    paper_ii_final_selector_comparison_v1 as selector,
)

CORRECTED_LOCK = Path(selector.__file__).with_name("paper_ii_corrected_exchange_cells_v1.json")
TEMPLATE_PACKAGE = Path("chtc/paper_ii_corrected_exchange_v1_a43bcd86_20260905")
SCOPES = {"strongfast_dtheta": ("strongfast__delta_theta_5e-5",), "eleven": None, "cells": ()}
_LITERAL = re.compile(r"^[0-9.]+e-[0-9]+$")


def token_for(literal: str) -> str:
    """``5e-3`` -> ``5em03``; ``2e-2`` -> ``2em02``: the id-safe spelling."""

    if not _LITERAL.match(literal):
        raise ValueError(f"literal must look like 5e-3; got {literal!r}")
    mantissa, exponent = literal.split("e-")
    return f"{mantissa.replace('.', 'p')}em{int(exponent):02d}"


def campaign_id_for(literal: str | None, scope: str) -> str:
    if scope == "cells":
        raise ValueError("scope 'cells' needs an explicit campaign_id")
    if literal is None:
        raise ValueError("an undeclared threshold needs an explicit campaign_id")
    token = token_for(literal)
    if scope == "strongfast_dtheta":
        return f"paper_ii_permission_threshold_{token}_v1"
    if scope == "eleven":
        return f"paper_ii_corrected_exchange_tau_{token}_v1"
    raise ValueError(f"unknown scope {scope!r}; known: {sorted(SCOPES)}")


def write_lock(
    *,
    literal: str | None,
    scope: str,
    output: Path | None = None,
    created: str | None = None,
    cell_ids: Sequence[str] | None = None,
    apm_cut_overrides: Mapping[str, float] | None = None,
    campaign_id: str | None = None,
    purpose: str | None = None,
) -> Path:
    """Write the lock for ``literal`` over ``scope`` and return its path.

    ``literal=None`` keeps the manuscript's permission bound (no extra flag).
    ``scope="cells"`` takes ``cell_ids`` explicitly; ``apm_cut_overrides``
    moves a cell to another activation-threshold rung, which the purpose text
    must say.
    """

    payload = json.loads(CORRECTED_LOCK.read_text(encoding="utf-8"))
    parent = hashlib.sha256(CORRECTED_LOCK.read_bytes()).hexdigest()
    cell_ids = tuple(cell_ids) if scope == "cells" else SCOPES[scope]
    if scope == "cells" and not cell_ids:
        raise ValueError("scope 'cells' needs cell_ids")
    cells = [json.loads(json.dumps(c)) for c in payload["cells"] if cell_ids is None or c["cell_id"] in cell_ids]
    if cell_ids is not None and len(cells) != len(cell_ids):
        raise ValueError("scope names a cell absent from the corrected lock")
    for cell in cells:
        override = (apm_cut_overrides or {}).get(cell["cell_id"])
        if override is not None:
            cell["apm_cut"] = float(override)
    campaign_id = campaign_id or campaign_id_for(literal, scope)
    lock = dict(payload)
    lock["campaign_id"] = campaign_id
    lock["lock_id"] = f"{campaign_id[:-3]}_cells_v1"  # paper_ii_..._<tok>_v1 -> ..._<tok>_cells_v1
    lock["created"] = created or _dt.date.today().isoformat()
    lock["parent_matched_cell_lock_sha256"] = parent
    lock["purpose"] = purpose or (
        f"Pre-permission ray threshold {literal}: the corrected exchange campaign's "
        f"{'strong--fast (2.4,3) delta-theta cell' if cell_ids else 'eleven cells'} "
        "re-run with the ray bound of Paper II Eq. deletion_ray_bound tested at "
        f"{literal} instead of the post-refit tolerance 2e-3. Post-refit ray and "
        "velocity certificates unchanged."
    )
    lock["adoption_note"] = (
        "Not adopted paper evidence until the user promotes it. Cell rows, cuts, "
        "sources, and metrics copied verbatim from the corrected lock."
    )
    lock["cells"] = cells
    if cell_ids is None:
        lock["exclusion_note"] = payload.get("exclusion_note", "")
    else:
        lock["exclusions"] = []
        lock["exclusion_note"] = "Single-cell grid; nothing excluded."
        lock["grid"] = {
            "cell_count": len(cells),
            "controllers": sorted({c["controller"] for c in cells}),
            "drives": sorted({c["drive"] for c in cells}),
            "excluded_count": 0,
            "relation": "cells + exclusions = drives x controllers",
        }
    bindings = dict(payload["registry_bindings"])
    bindings["extra_flags"] = (
        ["--record-statevector"]
        if literal is None
        else ["--record-statevector", selector.PERMISSION_RAY_FLAG, literal]
    )
    lock["registry_bindings"] = bindings
    path = output or CORRECTED_LOCK.with_name(f"{lock['lock_id']}.json")
    path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selector.load_lock(path, **selector.lock_parameters(path))  # fail closed before use
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def build_package(
    *,
    lock_path: Path,
    package_dir: Path,
    source_revision: str,
    repo_root: Path = Path("."),
    template: Path = TEMPLATE_PACKAGE,
    compile_workers: int = 8,
) -> Path:
    """Template the corrected package for ``lock_path`` at ``source_revision``.

    Built in a temporary directory so the working tree stays clean while the
    packager resolves the code revision, then moved to ``package_dir``.
    """

    repo_root = Path(repo_root)
    template = repo_root / template
    lock_path = Path(lock_path)
    lock = selector.load_lock(lock_path, **selector.lock_parameters(lock_path))
    literal = lock.permission_ray_distance_max  # None keeps the manuscript's bound
    campaign_id = telemetry.campaign_id_for(lock)
    with tempfile.TemporaryDirectory(prefix="paper_ii_bundle_") as scratch:
        pkg = Path(scratch) / package_dir.name
        (pkg / "input").mkdir(parents=True)
        (pkg / "logs").mkdir()
        archive = pkg / "input" / f"source_{source_revision[:8]}.tar.gz"
        subprocess.run(
            ["git", "-C", str(repo_root), "archive", "--format=tar.gz", "-o", str(archive.resolve()),
             source_revision, "pipelines", "src", "chtc/paper_ii_production_v2/input/seeds"],
            check=True,
        )
        source_sha = _sha256(archive)
        shutil.copy(lock_path, pkg / "input/matched_cells_lock.json")
        for name in ("calibration_receipt.json", "report_refresh_resolution.json"):
            shutil.copy(template / "input" / name, pkg / "input" / name)
        summary = telemetry.prepare(
            output_root="raw_outputs",
            actions_jsonl=pkg / "input/actions.jsonl",
            lock_json=lock_path,
            source_package_sha256=source_sha,
            compile_workers=compile_workers,
            comparator_mode=telemetry.COMPARATOR_MODE_REUSE,
        )
        rows = [json.loads(line) for line in (pkg / "input/actions.jsonl").read_text().splitlines() if line]
        if len(rows) != len(lock.cells) or any(r["code_revision"] != source_revision for r in rows):
            raise RuntimeError("actions do not carry the requested source revision (dirty tree?)")
        (pkg / "input/prepare_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        ids = "".join(r["action_id"] + "\n" for r in rows)
        (pkg / "input/action_ids.txt").write_text(ids)
        (pkg / "input/action_ids_apm.txt").write_text(ids)
        manifest = json.loads((template / "input/manifest.json").read_text())
        template_rev = manifest["source"]["code_revision"]
        template_sha = manifest["source"]["archive_sha256"]
        template_campaign = manifest["campaign_id"]

        def rewrite(name: str, extra: Sequence[tuple[str, str]] = ()) -> None:
            text = (template / name).read_text()
            text = text.replace(template.name, package_dir.name)
            text = text.replace(f"source_{template_rev[:8]}.tar.gz", archive.name)
            text = text.replace(template_rev, source_revision).replace(template_sha, source_sha)
            text = text.replace(f'"{template_campaign}"', f'"{campaign_id}"')
            for old, new in extra:
                if old not in text:
                    raise RuntimeError(f"{name}: template anchor missing: {old!r}")
                text = text.replace(old, new)
            (pkg / name).write_text(text)
            if name.endswith(".sh"):
                (pkg / name).chmod(0o755)

        n = len(rows)
        # Campaigns share action ids, and every job stages its archive into the
        # submit directory's archives/: prefix the archive and receipt names with
        # the campaign id so a later stage-out cannot overwrite an earlier one.
        rewrite(
            "run_action.py",
            [
                ("EXPECTED_ACTION_COUNT = 11", f"EXPECTED_ACTION_COUNT = {n}"),
                ('target = archive_dir / f"{action_id}.tar.gz"', 'target = archive_dir / f"{CAMPAIGN_ID}__{action_id}.tar.gz"'),
                ('archive_dir / f"{action_id}.archive.json"', 'archive_dir / f"{CAMPAIGN_ID}__{action_id}.archive.json"'),
            ],
        )
        rewrite("run_action_apptainer.sh")
        rewrite("submit_apm.sub", [('"paper-ii-corrected-exchange-apm-v1"', f'"{campaign_id.replace("_", "-")}"')])
        validator = (template / "validate_package.py").read_text()
        validator = re.sub(r"== 11\b", "== EXPECTED_ACTION_COUNT", validator)
        validator = validator.replace(
            "PACKAGE = Path(__file__).resolve().parent",
            "PACKAGE = Path(__file__).resolve().parent\n"
            f"EXPECTED_ACTION_COUNT = {n}\nEXPECTED_PERMISSION_RAY = {literal!r}",
        )
        anchor = '            "declared exchange frontier absent",\n        )\n'
        if anchor not in validator:
            raise RuntimeError("validator template anchor missing")
        validator = validator.replace(
            anchor,
            anchor
            + '        require(\n            flag_pair(runner, "--exchange-permission-ray-distance-max")\n'
            '            == (None if EXPECTED_PERMISSION_RAY is None else ("--exchange-permission-ray-distance-max", EXPECTED_PERMISSION_RAY)),\n'
            '            "declared pre-permission threshold drift",\n        )\n',
        )
        validator = validator.replace(
            '"pipelines/time_dynamics/campaigns/paper_ii_corrected_exchange_cells_v1.json",',
            '"pipelines/time_dynamics/campaigns/paper_ii_corrected_exchange_cells_v1.json",\n'
            f'        "{lock_path.relative_to(repo_root) if lock_path.is_absolute() else lock_path}",',
        )
        validator = validator.replace(template_rev, source_revision).replace(template_sha, source_sha)
        validator = validator.replace(f'"{template_campaign}"', f'"{campaign_id}"')
        (pkg / "validate_package.py").write_text(validator)

        manifest["campaign_id"] = campaign_id
        manifest["created_utc"] = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        manifest["purpose"] = json.loads(lock_path.read_text())["purpose"]
        manifest["scope"].update({"action_count": n, "cell_count": n, "apm_action_count": n,
                                  "purpose": f"pre-permission ray threshold {literal}"})
        manifest["routes"] = {route: n for route in manifest["routes"]}
        manifest["search"]["exchange_permission_ray_distance_max"] = literal
        manifest["search"]["apm_cuts"] = {row["cell_id"]: row["apm_cut"] for row in rows}
        manifest["source"].update({
            "code_revision": source_revision, "archive": f"input/{archive.name}", "archive_sha256": source_sha,
            "archive_command": f"git archive --format=tar.gz {source_revision} pipelines src chtc/paper_ii_production_v2/input/seeds",
            "template_package": template.name,
        })
        for key, rel in (("actions_jsonl", "input/actions.jsonl"), ("action_ids", "input/action_ids.txt"),
                         ("action_ids_apm", "input/action_ids_apm.txt"), ("prepare_summary", "input/prepare_summary.json"),
                         ("matched_cells_lock", "input/matched_cells_lock.json"),
                         ("calibration_receipt", "input/calibration_receipt.json"),
                         ("report_refresh_resolution", "input/report_refresh_resolution.json")):
            manifest["inputs"][key]["path"] = rel
            manifest["inputs"][key]["sha256"] = _sha256(pkg / rel)
        manifest["package_files"] = {
            name: _sha256(pkg / name)
            for name in ("run_action.py", "run_action_apptainer.sh", "submit_apm.sub", "validate_package.py")
        }
        manifest["adoption"]["purpose"] = "pre-permission threshold campaign; adoption is the user's decision"
        (pkg / "input/manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
        if package_dir.exists():
            shutil.rmtree(package_dir)
        shutil.move(str(pkg), str(package_dir))
    return package_dir


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--literal", default=None, help="threshold literal as it appears on the argv, e.g. 5e-3; omit to keep the manuscript's bound")
    parser.add_argument("--scope", choices=sorted(SCOPES), default="eleven")
    parser.add_argument("--cell", action="append", default=[], help="scope 'cells': a cell id, repeatable")
    parser.add_argument("--apm-cut", action="append", default=[], help="cell_id=cut to move a cell to another activation-threshold rung")
    parser.add_argument("--campaign-id", default=None)
    parser.add_argument("--purpose", default=None)
    parser.add_argument("--source-revision", default=None, help="commit to archive; default HEAD (must be clean)")
    parser.add_argument("--build", action="store_true", help="also build the package under chtc/")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.apm_cut:
        cell_id, _, value = item.partition("=")
        overrides[cell_id] = float(value)
    lock_path = write_lock(
        literal=args.literal, scope=args.scope, cell_ids=args.cell or None,
        apm_cut_overrides=overrides or None, campaign_id=args.campaign_id, purpose=args.purpose,
    )
    print("lock", lock_path)
    if args.build:
        rev = args.source_revision or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        date = _dt.date.today().strftime("%Y%m%d")
        campaign_id = args.campaign_id or campaign_id_for(args.literal, args.scope)
        pkg = Path("chtc") / f"{campaign_id}_{rev[:8]}_{date}"
        print("package", build_package(lock_path=lock_path, package_dir=pkg, source_revision=rev))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
