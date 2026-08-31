"""HH-route (macro semantic-closure, forced) baseline under noise, L=3."""
import json, sys
from pathlib import Path
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import importlib.util, dataclasses
spec = importlib.util.spec_from_file_location("v9r", REPO/"chtc/ra_adapt_appendix_demos_20260820_v9/run_local_ra_adapt_appendix_demos_l3_20260820_v9.py")
r = importlib.util.module_from_spec(spec); sys.modules["v9r"]=r; spec.loader.exec_module(r); r.configure_science_environment()

import pipelines.static_adapt.ra_adapt.semantic_closure_routes as scr
from pipelines.static_adapt.ra_adapt.appendix_demos import build_ra_adapt_appendix_pure_hubbard_problem as P
from pipelines.static_adapt.ra_adapt import run_ra_adapt

LEVEL, OUT, H = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
base = scr.build_paper_i_ra_all_phase_adaptive_natural_terminal_request(
    insertion_policy="append_only", maximum_controller_rounds=H)
adapter = type(base.adapter)(
    route_variant=scr.PAPER_I_RA_MACRO_ALL_PHASE_ADAPTIVE_SHORTLIST_FORCED_V2,
    noise_level_id=LEVEL,
)
req = dataclasses.replace(base, adapter=adapter)
p = P(num_sites=3)
prot = scr.materialize_paper_i_ra_semantic_protocol(p, req)
print(f"[{LEVEL}] protocol OK active={prot.active_gradient_policy} "
      f"chart={prot.accepted_refit_coordinate_chart}", flush=True)
res = run_ra_adapt(p, prot)
rows = [x.to_dict() for x in res.run.accepted_trajectory]
(OUT/"result.json").write_text(json.dumps({
    "protocol": prot.to_dict(),
    "run": {"accepted_trajectory": rows, "stop": res.run.stop.to_dict()},
    "scientific_receipts": dict(res.scientific_receipts)}, indent=2, default=str))
EX = -2.1414935642714075
errs=[abs(float(x.get("accepted_energy", x.get("energy")))-EX) for x in rows]
print(f"[{LEVEL}] DONE rounds={len(rows)} err_final={errs[-1]:.3e}", flush=True)
