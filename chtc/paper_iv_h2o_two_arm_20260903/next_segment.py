#!/usr/bin/env python3
"""Turn a retrieved segment's results/ into the next segment's submit file.

Never submits. Given the local copy of `<arm>/segNN/results/` it:

  1. reads the segment's stop from result.json (`run.stop.primary_reason`,
     `run.stop.fired_reasons`; summary.json mirrors them as `stop_reason`,
     `stop_fired_reasons`). A wall-stopped segment exits 0 and writes
     result.json exactly like a maximum_controller_rounds stop
     (h2o_application.py, `--max-wall-seconds`). The verdict is:
       continue  iff primary_reason == "wall_clock_budget_v1"
                 (sr_snake/_controller.py WALL_CLOCK_BUDGET_STOP_REASON);
       done      if "exact_ed_target_reached" or "maximum_controller_rounds"
                 fired (the 1.6 mHa stop or the 50-round horizon), or
                 summary.json says chemical_accuracy_reached;
       refused   for anything else (no result.json, an unknown reason),
                 with the reason printed. A protocol stop that fires in the
                 same round as the wall budget keeps precedence
                 (`fired_reasons[0]`), so a segment is never continued past
                 a scientific stop.
  2. verifies current.json: reason `iteration_done`, `complete: false`,
     the estimator-ledger pointer in `checkpoint` and `adapt_vqe` agree, the
     pointer names one sibling file whose sha256 matches, and the sidecar
     carries the checkpoint-sidecar schema (`sr_snake/_resume.py:4047-4073`
     hard-fails otherwise); every other content-addressed sidecar pointer
     (e.g. verified_singleton_resume_sidecar) is verified the same way;
  3. stages current.json plus every referenced sidecar into
     `<arm>/seg{NN+1}/input/` with a `resume_manifest.sha256`;
  4. renders `<arm>/seg{NN+1}/submit.sub` with `--resume-checkpoint` and a
     fresh --output-dir (results/ of the new segment directory).

The stop-reason strings below are copied from
`pipelines/static_adapt/sr_snake/_controller.py` (the engine that ships in
source.tgz); `runtime_preflight.py` checks the archived source still spells
them the same way, so this file never guesses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_package as bp  # noqa: E402

LEDGER_POINTER_SCHEMA = "paper_i_estimator_call_ledger_checkpoint_pointer_v2"
LEDGER_SIDECAR_SCHEMA = "paper_i_estimator_call_ledger_checkpoint_sidecar_v2"

# sr_snake/_controller.py: WALL_CLOCK_BUDGET_STOP_REASON and the two protocol
# stop conditions evaluated by _evaluate_stop. The wall budget has the lowest
# precedence, so primary_reason == WALL_CLOCK_STOP_REASON already implies no
# protocol stop fired in that round; the fired_reasons check below is belt
# and braces.
WALL_CLOCK_STOP_REASON = "wall_clock_budget_v1"
EXACT_TARGET_STOP_REASON = "exact_ed_target_reached"
HORIZON_STOP_REASON = "maximum_controller_rounds"
TERMINAL_STOP_REASONS = (EXACT_TARGET_STOP_REASON, HORIZON_STOP_REASON)

DISPOSITION_CONTINUE = "continue"
DISPOSITION_DONE_EXACT_TARGET = "done_exact_target_reached"
DISPOSITION_DONE_HORIZON = "done_horizon_reached"
DISPOSITION_DONE_CHEMICAL_ACCURACY = "done_chemical_accuracy"
DISPOSITION_ABNORMAL = "abnormal_exit"
DISPOSITION_UNKNOWN_STOP = "unknown_stop_reason"


class Refusal(SystemExit):
    def __init__(self, message: str) -> None:
        super().__init__(f"REFUSED: {message}")


def sha256(path: Path) -> str:
    return bp.sha256(path)


def load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise Refusal(f"{label} missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise Refusal(f"{label} is not valid JSON ({exc}): {path}")
    if not isinstance(payload, dict):
        raise Refusal(f"{label} is not a JSON object: {path}")
    return payload


def _stop_block(results: Path) -> dict[str, Any]:
    """Read the stop receipt from result.json (authoritative) and summary.json.

    h2o_application writes result.json (`run.stop`) first and summary.json
    (`stop_reason`, `stop_fired_reasons`, `wall_clock_elapsed_seconds`,
    `chemical_accuracy_reached`) second, both only after the engine returned
    normally. A segment that crashed, was held, or was killed has neither.
    """
    result_json = results / "result.json"
    summary_json = results / "summary.json"
    out: dict[str, Any] = {
        "result_json_present": result_json.is_file(),
        "summary_json_present": summary_json.is_file(),
        "primary_reason": None,
        "fired_reasons": None,
        "completed_controller_rounds": None,
        "wall_clock_budget_seconds": None,
        "wall_clock_elapsed_seconds": None,
        "chemical_accuracy_reached": None,
        "summary_stop_reason": None,
    }
    if result_json.is_file():
        result = load_json(result_json, label="result.json")
        run = result.get("run") if isinstance(result.get("run"), dict) else {}
        stop = run.get("stop") if isinstance(run.get("stop"), dict) else {}
        out["primary_reason"] = stop.get("primary_reason")
        fired = stop.get("fired_reasons")
        out["fired_reasons"] = [str(r) for r in fired] if isinstance(fired, list) else None
        out["completed_controller_rounds"] = stop.get("completed_controller_rounds")
        out["wall_clock_budget_seconds"] = stop.get("wall_clock_budget_seconds")
        out["wall_clock_elapsed_seconds"] = stop.get("wall_clock_elapsed_seconds")
    if summary_json.is_file():
        summary = load_json(summary_json, label="summary.json")
        out["summary_stop_reason"] = summary.get("stop_reason")
        out["chemical_accuracy_reached"] = summary.get("chemical_accuracy_reached")
        if out["fired_reasons"] is None and isinstance(summary.get("stop_fired_reasons"), list):
            out["fired_reasons"] = [str(r) for r in summary["stop_fired_reasons"]]
        if out["completed_controller_rounds"] is None:
            out["completed_controller_rounds"] = summary.get("completed_controller_rounds")
        if out["wall_clock_elapsed_seconds"] is None:
            out["wall_clock_elapsed_seconds"] = summary.get("wall_clock_elapsed_seconds")
        if out["primary_reason"] is None and summary.get("stop_reason") is not None:
            out["primary_reason"] = summary.get("stop_reason")
        elif (
            out["primary_reason"] is not None
            and summary.get("stop_reason") is not None
            and str(summary.get("stop_reason")) != str(out["primary_reason"])
        ):
            raise Refusal(
                f"result.json primary_reason={out['primary_reason']!r} disagrees "
                f"with summary.json stop_reason={summary.get('stop_reason')!r}; "
                "the retrieved results/ is inconsistent."
            )
    return out


def completion_state(results: Path) -> dict[str, Any]:
    """Classify how the previous segment stopped.

    Returns a state whose `disposition` is one of the DISPOSITION_* values;
    only DISPOSITION_CONTINUE is resumable. The caller decides whether to
    refuse (staging) or merely report (--verify-only).
    """
    receipt_json = results / "segment_receipt.json"
    receipt = load_json(receipt_json, label="segment_receipt.json") if receipt_json.is_file() else None
    stop = _stop_block(results)
    fired = set(stop["fired_reasons"] or ())
    primary = None if stop["primary_reason"] is None else str(stop["primary_reason"])
    exit_code = None if receipt is None else receipt.get("python_exit_code")

    if stop["chemical_accuracy_reached"] is True or EXACT_TARGET_STOP_REASON in fired or primary == EXACT_TARGET_STOP_REASON:
        disposition = DISPOSITION_DONE_EXACT_TARGET
        why = (
            "the exact-target stop fired (summary chemical_accuracy_reached="
            f"{stop['chemical_accuracy_reached']!r}, fired_reasons={sorted(fired)}); "
            "the arm is finished, there is no next segment."
        )
        if stop["chemical_accuracy_reached"] is True and EXACT_TARGET_STOP_REASON not in fired:
            disposition = DISPOSITION_DONE_CHEMICAL_ACCURACY
    elif HORIZON_STOP_REASON in fired or primary == HORIZON_STOP_REASON:
        disposition = DISPOSITION_DONE_HORIZON
        why = (
            f"maximum_controller_rounds fired after {stop['completed_controller_rounds']} "
            "rounds (the campaign horizon); the arm is finished, there is no next segment."
        )
    elif not stop["result_json_present"]:
        disposition = DISPOSITION_ABNORMAL
        why = (
            "no result.json: the segment did not return normally (python_exit_code="
            f"{exit_code!r}). A wall-clock stop writes result.json with "
            f"primary_reason={WALL_CLOCK_STOP_REASON!r}; read segment_receipt.json and "
            "logs/ before deciding anything. Pass --resume-after-abnormal-exit to stage "
            "the last accepted checkpoint anyway."
        )
    elif primary == WALL_CLOCK_STOP_REASON:
        disposition = DISPOSITION_CONTINUE
        why = (
            f"stopped by the wall-clock budget after {stop['completed_controller_rounds']} "
            f"rounds ({stop['wall_clock_elapsed_seconds']}s of {stop['wall_clock_budget_seconds']}s)."
        )
    else:
        disposition = DISPOSITION_UNKNOWN_STOP
        why = (
            f"result.json primary_reason={primary!r} (fired_reasons={sorted(fired)}) is "
            f"neither {WALL_CLOCK_STOP_REASON!r} nor a known terminal stop "
            f"{list(TERMINAL_STOP_REASONS)}; refusing to guess."
        )
    return {
        "disposition": disposition,
        "why": why,
        "stop_reason": primary,
        "fired_reasons": sorted(fired),
        "completed_controller_rounds": stop["completed_controller_rounds"],
        "wall_clock_budget_seconds": stop["wall_clock_budget_seconds"],
        "wall_clock_elapsed_seconds": stop["wall_clock_elapsed_seconds"],
        "chemical_accuracy_reached": stop["chemical_accuracy_reached"],
        "result_json_present": stop["result_json_present"],
        "summary_json_present": stop["summary_json_present"],
        "python_exit_code": exit_code,
        "receipt": receipt,
    }


def verify_pointer(
    pointer: Any, *, owner: str, current: Path
) -> tuple[Path, str]:
    if not isinstance(pointer, dict):
        raise Refusal(f"{owner} is not a pointer object")
    name = pointer.get("path")
    digest = str(pointer.get("sha256", "")).lower()
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise Refusal(f"{owner}.path must name one sibling file; got {name!r}")
    if len(digest) != 64:
        raise Refusal(f"{owner}.sha256 is not a sha256 hex digest")
    sidecar = current.with_name(name)
    if sidecar.is_symlink() or not sidecar.is_file():
        raise Refusal(f"{owner} sidecar is not a regular sibling file: {sidecar}")
    actual = sha256(sidecar)
    if actual != digest:
        raise Refusal(
            f"{owner} sidecar sha256 mismatch: pointer {digest[:16]} vs file "
            f"{actual[:16]} ({sidecar.name}). Resume would hard-fail."
        )
    return sidecar, digest


def verify_checkpoint(results: Path) -> dict[str, Any]:
    current = results / "current.json"
    payload = load_json(current, label="current.json")
    checkpoint = payload.get("checkpoint")
    adapt = payload.get("adapt_vqe")
    if not isinstance(checkpoint, dict) or not isinstance(adapt, dict):
        raise Refusal("current.json lacks checkpoint/adapt_vqe blocks")
    if checkpoint.get("reason") != "iteration_done":
        raise Refusal(
            f"checkpoint.reason={checkpoint.get('reason')!r}; only "
            "iteration_done checkpoints are resumable"
        )
    if checkpoint.get("complete") is not False:
        raise Refusal("checkpoint.complete is not false; this is a finished run")
    depth = checkpoint.get("depth")
    if not isinstance(depth, int) or depth < 1:
        raise Refusal(f"checkpoint.depth={depth!r} is not a positive integer")

    ledger_field = "estimator_call_ledger_checkpoint"
    top_pointer = checkpoint.get(ledger_field)
    adapt_pointer = adapt.get(ledger_field)
    if top_pointer is None or adapt_pointer is None:
        raise Refusal(
            "current.json has no estimator-ledger pointer; the segment must "
            "run with the estimator ledger enabled for resume to validate"
        )
    if top_pointer != adapt_pointer:
        raise Refusal("checkpoint and adapt_vqe estimator-ledger pointers disagree")
    if top_pointer.get("schema") != LEDGER_POINTER_SCHEMA:
        raise Refusal(f"ledger pointer schema {top_pointer.get('schema')!r} unsupported")
    if top_pointer.get("status") != "complete" or top_pointer.get("current_round_finalized") is not True:
        raise Refusal("ledger pointer is not a complete, finalized-round pointer")
    if top_pointer.get("checkpoint_depth") != depth:
        raise Refusal("ledger pointer checkpoint_depth disagrees with checkpoint.depth")
    ledger_sidecar, ledger_sha = verify_pointer(
        top_pointer, owner=f"checkpoint.{ledger_field}", current=current
    )
    sidecar_payload = load_json(ledger_sidecar, label="estimator-ledger sidecar")
    if sidecar_payload.get("schema") != LEDGER_SIDECAR_SCHEMA:
        raise Refusal(
            f"sidecar schema {sidecar_payload.get('schema')!r} != "
            f"{LEDGER_SIDECAR_SCHEMA!r}; resume refuses it"
        )
    sidecar_ckpt = sidecar_payload.get("checkpoint") or {}
    if (
        sidecar_ckpt.get("reason") != "iteration_done"
        or sidecar_ckpt.get("current_round_finalized") is not True
        or sidecar_ckpt.get("depth") != depth
    ):
        raise Refusal("estimator-ledger sidecar is not closed at the accepted round")

    staged: dict[str, str] = {"current.json": sha256(current), ledger_sidecar.name: ledger_sha}
    for key, value in adapt.items():
        if key == ledger_field or not isinstance(value, dict):
            continue
        if "path" in value and "sha256" in value and str(value.get("path", "")).startswith("current."):
            path, digest = verify_pointer(value, owner=f"adapt_vqe.{key}", current=current)
            staged[path.name] = digest

    energy = adapt.get("energy")
    return {
        "depth": depth,
        "ansatz_depth": checkpoint.get("ansatz_depth"),
        "energy_hartree": energy,
        "route_contract_sha256": checkpoint.get("sr_route_profile_contract_sha256"),
        "current_sha256": staged["current.json"],
        "staged_files": staged,
    }


def chemical_accuracy_from_protocol(results: Path, energy: Any) -> dict[str, Any]:
    protocol_json = results / "protocol.json"
    if not protocol_json.is_file() or not isinstance(energy, (int, float)):
        return {"checked": False}
    protocol = load_json(protocol_json, label="protocol.json")
    target = (protocol.get("stopping_rule") or {}).get("exact_ed_target") or {}
    exact = target.get("energy")
    tol = target.get("absolute_tolerance")
    if not isinstance(exact, (int, float)) or not isinstance(tol, (int, float)):
        return {"checked": False}
    error = abs(float(energy) - float(exact))
    reached = error <= float(tol)
    return {
        "checked": True,
        "exact_energy_hartree": exact,
        "tolerance": tol,
        "abs_error_hartree": error,
        "reached": reached,
    }


def infer_previous(results: Path, arm: str | None, segment: int | None) -> tuple[str, int, dict[str, Any] | None]:
    receipt_path = results / "segment_receipt.json"
    receipt = load_json(receipt_path, label="segment_receipt.json") if receipt_path.is_file() else None
    if receipt is not None:
        r_arm, r_seg = receipt.get("arm"), receipt.get("segment")
        if arm is not None and arm != r_arm:
            raise Refusal(f"--arm {arm} but segment_receipt.json says {r_arm}")
        if segment is not None and segment != r_seg:
            raise Refusal(f"--previous-segment {segment} but receipt says {r_seg}")
        arm, segment = r_arm, int(r_seg)
    if arm is None or segment is None:
        raise Refusal("no segment_receipt.json; pass --arm and --previous-segment")
    if arm not in bp.ARMS:
        raise Refusal(f"unknown arm {arm!r}")
    return arm, int(segment), receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--previous-results", type=Path, required=True, help="local copy of the retrieved <arm>/segNN/results/")
    parser.add_argument("--arm", choices=sorted(bp.ARMS), default=None)
    parser.add_argument("--previous-segment", type=int, default=None)
    parser.add_argument("--cost-mode", choices=bp.COST_MODES, default=None, help="default: the previous segment's cost mode from its receipt, else the manifest")
    parser.add_argument("--max-wall-seconds", type=int, default=None, help="default: package/input_manifest.json segment_budget")
    parser.add_argument("--results-mode", choices=bp.RESULTS_MODES, default=None)
    parser.add_argument("--verify-only", action="store_true", help="run the checks and print the verdict; stage nothing. Exit 0 when the segment is resumable, 3 when the arm is finished or the stop is not a wall-clock stop, 1 when the checkpoint fails verification.")
    parser.add_argument("--resume-after-abnormal-exit", action="store_true", help="stage the last accepted checkpoint of a segment that returned no result.json (crash/hold/kill). Never continues past a terminal protocol stop.")
    parser.add_argument("--force", action="store_true", help="overwrite an existing next-segment directory that has no results/")
    args = parser.parse_args(argv)

    results = args.previous_results.expanduser().resolve()
    if not results.is_dir():
        raise Refusal(f"previous results directory does not exist: {results}")
    manifest = load_json(bp.PACKAGE_DIR / "package" / "input_manifest.json", label="package/input_manifest.json")

    arm, prev_seg, receipt = infer_previous(results, args.arm, args.previous_segment)
    state = completion_state(results)
    disposition = state["disposition"]
    if disposition == DISPOSITION_ABNORMAL and args.resume_after_abnormal_exit:
        print(f"WARNING: {state['why']} Staging anyway (--resume-after-abnormal-exit).", file=sys.stderr)
        disposition = DISPOSITION_CONTINUE
    resumable = disposition == DISPOSITION_CONTINUE

    verified = verify_checkpoint(results)
    accuracy = chemical_accuracy_from_protocol(results, verified["energy_hartree"])
    horizon = int(manifest["segment_budget"]["horizon_rounds"])
    if resumable and accuracy.get("reached") is True:
        # Belt and braces: the engine's exact stop should already have fired.
        disposition, resumable = DISPOSITION_DONE_CHEMICAL_ACCURACY, False
        state["why"] = (
            f"checkpoint energy {verified['energy_hartree']} is within {accuracy['tolerance']} Ha "
            f"of the exact target {accuracy['exact_energy_hartree']} "
            f"(|dE|={accuracy['abs_error_hartree']:.3e}); chemical accuracy reached, no next segment."
        )
    if resumable and verified["depth"] >= horizon:
        disposition, resumable = DISPOSITION_DONE_HORIZON, False
        state["why"] = f"checkpoint depth {verified['depth']} already at the {horizon}-round horizon."

    cost_mode = args.cost_mode or (receipt or {}).get("cost_mode") or manifest["arms"][arm]["cost_mode"]
    max_wall = args.max_wall_seconds or int(manifest["segment_budget"]["max_wall_seconds"])
    results_mode = args.results_mode or (receipt or {}).get("results_mode") or manifest["results_mode"]

    verdict = {
        "arm": arm,
        "previous_segment": prev_seg,
        "next_segment": prev_seg + 1 if resumable else None,
        "disposition": disposition,
        "why": state["why"],
        "stop_reason": state["stop_reason"],
        "fired_reasons": state["fired_reasons"],
        "completed_controller_rounds": state["completed_controller_rounds"],
        "wall_clock_budget_seconds": state["wall_clock_budget_seconds"],
        "wall_clock_elapsed_seconds": state["wall_clock_elapsed_seconds"],
        "python_exit_code": state["python_exit_code"],
        "result_json_present": state["result_json_present"],
        "summary_json_present": state["summary_json_present"],
        "checkpoint_depth": verified["depth"],
        "energy_hartree": verified["energy_hartree"],
        "chemical_accuracy": accuracy,
        "staged_files": verified["staged_files"],
        "cost_mode": cost_mode,
        "max_wall_seconds": max_wall,
        "results_mode": results_mode,
        "resumable": resumable,
    }
    if args.verify_only:
        print(json.dumps(verdict, indent=2, sort_keys=True))
        if not resumable:
            print(f"NOT RESUMABLE ({disposition}): {state['why']}", file=sys.stderr)
            return 3
        return 0
    if not resumable:
        raise Refusal(f"({disposition}) {state['why']}")

    next_seg = prev_seg + 1
    seg_dir = bp.PACKAGE_DIR / arm / bp.segment_dirname(next_seg)
    if seg_dir.exists():
        if (seg_dir / "results").exists() or not args.force:
            raise Refusal(f"{seg_dir} already exists" + ("" if args.force else " (pass --force to rewrite an unsubmitted one)"))
        shutil.rmtree(seg_dir)
    input_dir = seg_dir / "input"
    input_dir.mkdir(parents=True)
    (seg_dir / "logs").mkdir()
    (seg_dir / "logs" / ".keep").touch()

    staged_names: list[str] = []
    lines: list[str] = []
    for name, digest in verified["staged_files"].items():
        shutil.copyfile(results / name, input_dir / name)
        if sha256(input_dir / name) != digest:
            raise Refusal(f"copy of {name} did not preserve its digest")
        staged_names.append(name)
        lines.append(f"{digest}  {name}\n")
    (input_dir / "resume_manifest.sha256").write_text("".join(lines), encoding="utf-8")
    resume_files = tuple(["current.json"] + [n for n in staged_names if n != "current.json"] + ["resume_manifest.sha256"])

    submit_text = bp.render_submit(
        arm=arm,
        segment=next_seg,
        cost_mode=cost_mode,
        max_wall_seconds=max_wall,
        results_mode=results_mode,
        resume_files=resume_files,
        remote_root=manifest.get("remote_root", bp.REMOTE_HOME_ROOT),
        image_path=manifest.get("image_path", bp.DEFAULT_IMAGE_PATH),
        horizon_rounds=horizon,
    )
    problems = bp.check_submit_text(submit_text, label=f"{arm}/{bp.segment_dirname(next_seg)}/submit.sub")
    if problems:
        shutil.rmtree(seg_dir)
        raise Refusal("rendered submit file failed self-check: " + "; ".join(problems))
    (seg_dir / "submit.sub").write_text(submit_text, encoding="utf-8")
    bp.write_json(
        seg_dir / "segment.json",
        {
            "schema": "paper_iv_h2o_two_arm_segment_v1",
            "campaign": bp.CAMPAIGN,
            "arm": arm,
            "segment": next_seg,
            "candidate_representation": bp.ARMS[arm]["candidate_representation"],
            "cost_mode": cost_mode,
            "max_wall_seconds": max_wall,
            "horizon_rounds": horizon,
            "results_mode": results_mode,
            "resume_from": {
                "previous_segment": prev_seg,
                "previous_results": str(results),
                "current_sha256": verified["current_sha256"],
                "checkpoint_depth": verified["depth"],
                "previous_stop_reason": state["stop_reason"],
                "previous_cluster_id": (receipt or {}).get("cluster_id"),
            },
            "staged_utc": datetime.now(timezone.utc).isoformat(),
            "submitted_cluster": None,
            "submit_file": "submit.sub",
        },
    )
    verdict["next_segment_dir"] = str(seg_dir)
    print(json.dumps(verdict, indent=2, sort_keys=True))
    remote_root = manifest.get("remote_root", bp.REMOTE_HOME_ROOT)
    rel = seg_dir.relative_to(bp.PACKAGE_DIR)
    print(
        "\nNOT SUBMITTED. User-gated next steps (see README.md 'What is user-gated'):\n"
        f"  rsync -av {seg_dir}/ ap2001:{remote_root}/{rel}/\n"
        f"  ssh ap2001 'cd {remote_root}/{rel} && condor_submit submit.sub'\n"
        "then record the ClusterId in segment.json (submitted_cluster).",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
