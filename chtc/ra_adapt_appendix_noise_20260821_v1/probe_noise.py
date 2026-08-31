"""RA baseline under one noise regime, L=3 macro pool."""
import json, sys
from pathlib import Path
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
import importlib.util
spec = importlib.util.spec_from_file_location(
    "v8runner", REPO/"chtc/ra_adapt_appendix_demos_20260820_v8/run_local_ra_adapt_appendix_demos_l3_20260820_v8.py")
runner = importlib.util.module_from_spec(spec); sys.modules["v8runner"]=runner
spec.loader.exec_module(runner); runner.configure_science_environment()

from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem,
    build_ra_adapt_appendix_noise_request,
    ra_adapt_appendix_application_source_contract,
)
from pipelines.static_adapt.ra_adapt import run_ra_adapt

LEVEL, OUT, H = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
request = build_ra_adapt_appendix_noise_request(
    noise_level=LEVEL, maximum_controller_rounds=H, num_sites=3,
    output_dir=OUT/"observations")
source = ra_adapt_appendix_application_source_contract(problem, request)
print(f"[noise:{LEVEL}] noise={source.get("noise")} h={H} "
      f"pool={source['singleton_inventory']['count']}", flush=True)
result = run_ra_adapt(problem, request)
payload = {"protocol": result.protocol.to_dict(),
           "run": {"final_state": result.run.final_state.to_dict(),
                   "accepted_trajectory": [r.to_dict() for r in result.run.accepted_trajectory],
                   "accepted_transitions": [r.to_dict() for r in result.run.accepted_transitions],
                   "stop": result.run.stop.to_dict()},
           "scientific_receipts": dict(result.scientific_receipts)}
(OUT/"result.json").write_text(json.dumps(payload, indent=2, default=str))
(OUT/"source.json").write_text(json.dumps(source, indent=2, default=str))
print(f"[noise:{LEVEL}] DONE rounds={len(payload['run']['accepted_trajectory'])}")
