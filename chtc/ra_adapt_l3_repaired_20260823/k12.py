"""k=12 prefix cost: AVQE and RA, both tolerances, one build."""
import json, sys, time
from pathlib import Path
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import pipelines.static_adapt.ra_adapt.append as append_mod
import pipelines.static_adapt.engine_support as es
from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
from pipelines.static_adapt.sr_snake.contracts import (
    SRExecutionPolicy, SRStopPolicy, FreshStart)
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem as P,
    RAAdaptAppendixPureHubbardCandidateAdapter as A,
    ra_adapt_appendix_generator_representation as R, _observation_policy,
    build_ra_adapt_appendix_request)
from pipelines.static_adapt.ra_adapt.engine import run_ra_adapt
EX = -2.1414935642714075
OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
K = 12
prob = P(num_sites=3)

def avqe(tag, xtol, ftol):
    append_mod.AAVQE_COMPARATOR_POWELL_XTOL = xtol
    append_mod.AAVQE_COMPARATOR_POWELL_FTOL = ftol
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    req = AppendAdaptRequest(
        adapter=A(candidate_representation_id=R(3)),
        execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=K),
                                    resume=FreshStart()),
        observation=_observation_policy(d, horizon=K))
    pl = append_mod.run_append_adapt(prob, req).to_dict()
    rp = pl["result_payload"]; cr = rp["compiled_resources"]
    return (rp["estimator_accounting"]["S_alg"], cr.get("compiled_count_2q_total"),
            cr.get("compiled_depth_2q_total"), cr.get("compiled_depth_total"))

def ra(tag, adaptive, xtol, ftol):
    es.RA_ACCEPTED_REFIT_POWELL_XTOL = xtol
    es.RA_ACCEPTED_REFIT_POWELL_FTOL = ftol
    d = OUT / tag; d.mkdir(parents=True, exist_ok=True)
    req = build_ra_adapt_appendix_request(
        arm="baseline", num_sites=3, maximum_controller_rounds=K,
        adaptive_phase_shortlists=adaptive, phase0_gradient_screen=adaptive,
        output_dir=d)
    pl = run_ra_adapt(prob, req).to_dict()
    (d / "result.json").write_text(json.dumps(pl, default=str))
    led = json.loads(next(d.rglob("estimator_ledger.json")).read_text())["accounting"]
    return led["S_alg"]

for tol_tag, xt, ft in (("tight", 1e-5, 1e-12), ("loose", 1e-4, 1e-8)):
    t0 = time.time()
    a = avqe(f"avqe_{tol_tag}_k12", xt, ft)
    print(f"  AVQE   {tol_tag:<6} S_alg={a[0]:>6}  N2q={a[1]} D2q={a[2]} Dc={a[3]}  ({time.time()-t0:.0f}s)", flush=True)
    for adaptive in (False, True):
        t0 = time.time()
        s = ra(f"ra_{'adaptive' if adaptive else 'fixed'}_{tol_tag}_k12", adaptive, xt, ft)
        print(f"  RA {'adaptive' if adaptive else 'fixed   '} {tol_tag:<6} S_alg={s:>6}  ({time.time()-t0:.0f}s)", flush=True)
print("DONE", flush=True)
