"""Adaptive-route arms at their marker prefix: E and compiled cost."""
import json, math, sys, time
from pathlib import Path
import numpy as np
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import pipelines.static_adapt.engine_support as es
es.RA_ACCEPTED_REFIT_POWELL_XTOL = 1e-5
es.RA_ACCEPTED_REFIT_POWELL_FTOL = 1e-12
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_request, build_ra_adapt_appendix_pure_hubbard_problem as P,
    RAAdaptAppendixPureHubbardCandidateAdapter as A,
    ra_adapt_appendix_generator_representation as R)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.qiskit_backend_tools import (compile_circuit_for_backend, compiled_gate_stats,
    resolve_backend_targets, safe_circuit_depth)
EX = -2.1414935642714075
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
prob = P(num_sites=3)
pool = {str(c.label): c for c in A(candidate_representation_id=R(3)).executable_pool(prob).candidates}
ref = np.asarray(prob.reference_state.build_state(), dtype=complex).reshape(-1)
nq = int(round(math.log2(ref.size)))
tg, _ = resolve_backend_targets(requested_names=("FakeMarrakesh",), preferred_fake_backends=("FakeMarrakesh",),
    allow_preferred_fallback=False, fallback_mode="single", allow_runtime_lookup=False)
def tup(labels):
    _l, c = build_structural_ansatz_circuit([pool[l].term for l in labels], nq=nq,
        ref_state=ref, structure_theta_value=1.0)
    comp = compile_circuit_for_backend(c, tg[0].backend_obj, seed_transpiler=7,
        optimization_level=1, initial_layout=None)["compiled"]
    st = compiled_gate_stats(comp)
    return int(st["compiled_count_2q"]), int(st["compiled_depth_2q"]), int(safe_circuit_depth(comp))

CASES = [("baseline", 12), ("metric_ablation_gated", 12), ("beam_escalated", 12),
         ("batching", 12), ("batching", 6)]
for arm, K in CASES:
    tag = f"{arm}_k{K}"
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True); t0 = time.time()
    try:
        req = build_ra_adapt_appendix_request(
            arm=arm, num_sites=3, maximum_controller_rounds=K,
            adaptive_phase_shortlists=True, phase0_gradient_screen=True, output_dir=d)
        pl = run_ra_adapt(prob, req).to_dict()
        (d / "result.json").write_text(json.dumps(pl, default=str))
        tr = pl["run"]["accepted_trajectory"]
        led = json.loads(next(d.rglob("estimator_ledger.json")).read_text())["accounting"]
        wl = led.get("winning_lineage") or {}
        s_alg = wl.get("S_alg", led["S_alg"])
        errs = [abs(float(x.get("accepted_energy", x.get("energy"))) - EX) for x in tr]
        first = next((i + 1 for i, e in enumerate(errs) if e <= 1e-6), None)
        t = tup([str(o) for o in tr[-1]["operators"]])
        print(f"  {tag:<28} S_alg={s_alg:>6}  N2q={t[0]:>4} D2q={t[1]:>4} Dc={t[2]:>5}  "
              f"err={errs[-1]:.2e} first<=1e-6={first}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {tag:<28} FAIL {str(e)[:70]}", flush=True)
print("DONE", flush=True)
