"""Conventional Append-ADAPT (the repo's `aavqe` comparator route) on the
appendix L=3 pure-Hubbard point, same macro pool and Powell protocol."""
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
    ra_adapt_appendix_generator_representation,
    RAAdaptAppendixPureHubbardCandidateAdapter,
)
from pipelines.static_adapt.ra_adapt import run_append_adapt
from pipelines.static_adapt.ra_adapt.contracts import AppendAdaptRequest
from pipelines.static_adapt.sr_snake.contracts import (
    SRExecutionPolicy, SRStopPolicy,
)

LEVEL = sys.argv[1]            # "none" or low/high/extreme
OUT = Path(sys.argv[2]); H = int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
problem = build_ra_adapt_appendix_pure_hubbard_problem(num_sites=3)
adapter = RAAdaptAppendixPureHubbardCandidateAdapter(
    candidate_representation_id=ra_adapt_appendix_generator_representation(3),
    noise_level_id=None if LEVEL == "none" else LEVEL,
)
request = AppendAdaptRequest(
    adapter=adapter,
    execution=SRExecutionPolicy(stop=SRStopPolicy(maximum_controller_rounds=H)),
)
print(f"[aavqe:{LEVEL}] h={H} representation={adapter.candidate_representation_id}", flush=True)
result = run_append_adapt(problem, request)
payload = result.to_dict()
(OUT/"result.json").write_text(json.dumps(payload, indent=2, default=str))
body = payload["result_payload"]
print(f"[aavqe:{LEVEL}] DONE rounds={body.get('controller_rounds_completed')}")
for row in body["history"]:
    print(f"  r{row['controller_round']:>2} E={float(row['energy_after']):.12f}")
