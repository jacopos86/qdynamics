"""Theta screen variants with the delete-refit measurement fixed."""
import json, math, sys, time
from pathlib import Path
import numpy as np
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import pipelines.static_adapt.engine_support as es
es.RA_ACCEPTED_REFIT_POWELL_XTOL = 1e-5
es.RA_ACCEPTED_REFIT_POWELL_FTOL = 1e-12
from pipelines.static_adapt.sr_snake.contracts import (
    MetricPruning, SingletonAdmission, BeamOff, SRExecutionPolicy, SRStopPolicy,
    SRMethodPolicy, FreshStart, AppendOnlyInsertion, ExplicitDepthQiskitCompileCost)
from pipelines.static_adapt.ra_adapt.contracts import RAAdaptRequest
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem as P,
    RAAdaptAppendixPureHubbardCandidateAdapter as A,
    ra_adapt_appendix_generator_representation as R, _observation_policy)
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

GRID = [
    ("t1e-3_p1_baseline", dict(protect_steps=1, small_theta_abs=1e-3)),
    ("t1e-2_p1",          dict(protect_steps=1, small_theta_abs=1e-2)),
    ("t1e-1_p1",          dict(protect_steps=1, small_theta_abs=1e-1)),
    ("t1e-2_p3",          dict(protect_steps=3, small_theta_abs=1e-2)),
    ("t1e-1_p3",          dict(protect_steps=3, small_theta_abs=1e-1)),
    ("t1e-2_p1_z0.9",     dict(protect_steps=1, small_theta_abs=1e-2, retained_gain_ratio=0.9)),
    ("t1e-1_p1_z0.9",     dict(protect_steps=1, small_theta_abs=1e-1, retained_gain_ratio=0.9)),
    ("t1e-2_p1_eps1e-12", dict(protect_steps=1, small_theta_abs=1e-2, max_regression=1e-12)),
]
for tag, kw in GRID:
    d = OUT / tag.split()[0]; d.mkdir(parents=True, exist_ok=True); t0 = time.time()
    try:
        req = RAAdaptRequest(
            adapter=A(candidate_representation_id=R(3), adaptive_phase_shortlists=True,
                      phase0_gradient_screen=True),
            method=SRMethodPolicy(admission=SingletonAdmission(), insertion=AppendOnlyInsertion(),
                                  pruning=MetricPruning(**kw), beam=BeamOff(),
                                  compile_cost=ExplicitDepthQiskitCompileCost()),
            execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=15),
                                        resume=FreshStart()),
            observation=_observation_policy(d, horizon=15))
        pl = run_ra_adapt(prob, req).to_dict()
        (d / "result.json").write_text(json.dumps(pl, default=str))
        tr = pl["run"]["accepted_trajectory"]
        ev = pl["scientific_receipts"]["accepted_run_evidence"]["accepted_rounds"]
        acc_rounds = [int(r["controller_round"]) for r in ev
                      if int((r.get("pruning") or {}).get("accepted_deletion_count", 0)) > 0]
        first_nom = next((int(r["controller_round"]) for r in ev
                          if int((r.get("pruning") or {}).get("nomination_count", 0)) > 0), None)
        led = json.loads(next(d.rglob("estimator_ledger.json")).read_text())["accounting"]
        t12 = tup([str(o) for o in tr[11]["operators"]])
        t15 = tup([str(o) for o in tr[-1]["operators"]])
        e12 = abs(float(tr[11].get("accepted_energy", tr[11].get("energy"))) - EX)
        print(f"  {tag:<24} first_nom=r{first_nom} accepts@{acc_rounds} S_alg={led['S_alg']:>6}", flush=True)
        print(f"      k12: {t12[0]}/{t12[1]}/{t12[2]} err={e12:.1e}   k15: {t15[0]}/{t15[1]}/{t15[2]}  ({time.time()-t0:.0f}s)", flush=True)
    except Exception as e:
        print(f"  {tag:<24} FAIL {str(e)[:70]}", flush=True)
print("DONE", flush=True)
