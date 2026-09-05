"""Run and score the registered SIAM Nb=2 proof pair on the compiled-Q route."""

import json, os, resource, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SEED = HERE / "input" / "siam_nb2_seed.json"
OUT = HERE / "output"
REF = ROOT / "output" / "siam_nb2_proof_qc_v1_reference"


def main() -> int:
    from pipelines.time_dynamics.paper_ii_runs import build_siam_nb2_proof_run
    from pipelines.time_dynamics.runners.ap_append_from_adapt_artifact import main as run_main
    from pipelines.time_dynamics.siam_screening import FIXTURE_NB2
    from pipelines.time_dynamics.siam_seed import write_siam_seed_manifest
    from pipelines.time_dynamics.diagnostics.siam_offline_scoring import (
        classify_siam_matched_pair, score_siam_raw_run,
    )

    SEED.parent.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    REF.mkdir(parents=True, exist_ok=True)
    write_siam_seed_manifest(SEED, FIXTURE_NB2)

    revision = os.environ.get("CODE_REVISION", "unset")
    timing = {}
    for arm in ("avqds", "exchange_q_compiled"):
        run_json = OUT / f"{arm}.run.json"
        cmd = build_siam_nb2_proof_run(
            seed_path=str(SEED), arm=arm, output_json=str(run_json),
            code_revision=revision, extra_flags=("--compile-workers", "1"),
        )
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        start = time.time()
        rc = run_main(list(cmd.argv()))
        wall = time.time() - start
        peak = max(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        )
        assert rc == 0, (arm, rc)
        timing[arm] = {"wall_seconds": wall, "peak_rss_mb": peak / (1024 * 1024),
                       "rss_before_mb": before / (1024 * 1024)}
        print(f"[run] {arm} rc={rc} wall={wall:.1f}s peak_rss={peak/(1024*1024):.0f}MB",
              flush=True)

    scores = {}
    for arm in ("avqds", "exchange_q_compiled"):
        score = score_siam_raw_run(
            run_path=OUT / f"{arm}.run.json", seed_path=SEED,
            reference_store_root=REF,
        )
        (OUT / f"{arm}.score.json").write_text(
            json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        scores[arm] = score
        s = score["summary"]
        print(f"[score] {arm} classification={s.get('classification')} "
              f"mean_infidelity={s.get('mean_infidelity'):.6g} "
              f"max_primitive_response_normalized_error="
              f"{s.get('max_primitive_response_normalized_error'):.6g}", flush=True)

    pair = classify_siam_matched_pair(scores["avqds"], scores["exchange_q_compiled"])
    (OUT / "matched_pair.json").write_text(
        json.dumps(pair, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "resources.json").write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[pair]", json.dumps(pair, indent=2, sort_keys=True)[:4000], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
