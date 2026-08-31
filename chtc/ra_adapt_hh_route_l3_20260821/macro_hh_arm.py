"""One appendix arm on HH's macro semantic-closure algorithm, L=3."""
import dataclasses, json, sys
from pathlib import Path
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v9r", REPO/"chtc/ra_adapt_appendix_demos_20260820_v9/run_local_ra_adapt_appendix_demos_l3_20260820_v9.py")
r = importlib.util.module_from_spec(spec); sys.modules["v9r"]=r; spec.loader.exec_module(r)
r.configure_science_environment()

import pipelines.static_adapt.ra_adapt.semantic_closure_routes as scr
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem as P,
    build_ra_adapt_appendix_request as APPENDIX_REQ)
from pipelines.static_adapt.ra_adapt import run_ra_adapt
from pipelines.static_adapt.sr_snake.contracts import SRMethodPolicy

ARM, OUT, H = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
MACRO = scr.PAPER_I_RA_MACRO_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_V2

# The branch-pricing beam needs the route to declare the three-term compile
# identity; the sanctioned way is the explicit compile-cost policy, which
# carries the same 0.30/0.30/0.25 weights the appendix and the paper use.
base = scr.build_paper_i_ra_all_phase_adaptive_natural_terminal_request(
    insertion_policy="append_only", maximum_controller_rounds=H,
    compile_cost_policy=("explicit_depth_qiskit_cost_v1"
                         if ARM == "beam_escalated" else None))
adapter = type(base.adapter)(route_variant=MACRO)

# take admission / pruning / beam from the appendix arm definition, so the
# only thing that differs from the appendix runs is the selector semantics
appendix = APPENDIX_REQ(arm=ARM, maximum_controller_rounds=H, num_sites=3)
method = dataclasses.replace(
    base.method,
    admission=appendix.method.admission,
    pruning=appendix.method.pruning,
    beam=appendix.method.beam,
)
req = dataclasses.replace(base, adapter=adapter, method=method,
                          observation=appendix.observation)
p = P(num_sites=3)
prot = scr.materialize_paper_i_ra_semantic_protocol(p, req)
print(f"[{ARM}] protocol OK active={prot.active_gradient_policy} "
      f"chart={prot.accepted_refit_coordinate_chart} repr={prot.candidate_representation}", flush=True)
res = run_ra_adapt(p, prot)
payload = {
    "protocol": prot.to_dict(),
    "run": {"final_state": res.run.final_state.to_dict(),
            "accepted_trajectory": [x.to_dict() for x in res.run.accepted_trajectory],
            "stop": res.run.stop.to_dict()},
    "scientific_receipts": dict(res.scientific_receipts),
}
(OUT/"result.json").write_text(json.dumps(payload, indent=2, default=str))
EX = -2.1414935642714075
# This route reports under accepted_round_receipts, not accepted_run_evidence.
rows = [x.to_dict() for x in res.run.accepted_trajectory]
errs = [abs(float(x.get("accepted_energy", x.get("energy"))) - EX) for x in rows
        if x.get("accepted_energy", x.get("energy")) is not None]
first = next((i + 1 for i, e in enumerate(errs) if e <= 1e-6), None)
print(f"[{ARM}] DONE rounds={len(rows)} first<=1e-6={first} "
      f"err_final={errs[-1] if errs else float('nan'):.3e}", flush=True)
