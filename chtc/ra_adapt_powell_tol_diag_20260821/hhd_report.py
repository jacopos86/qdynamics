"""Powell-tolerance diagnostic report: per-prefix tuples and sequences."""
import json, math, sys
import numpy as np
from pathlib import Path
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
sys.path.insert(0, str(REPO))
from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.qiskit_backend_tools import (
    compile_circuit_for_backend, compiled_gate_stats,
    resolve_backend_targets, safe_circuit_depth)
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem as P,
    RAAdaptAppendixPureHubbardCandidateAdapter as A,
    ra_adapt_appendix_generator_representation as R)

problem = P(num_sites=3)
pool = {str(c.label): c for c in A(candidate_representation_id=R(3)).executable_pool(problem).candidates}
ref = np.asarray(problem.reference_state.build_state(), dtype=complex).reshape(-1)
nq = int(round(math.log2(ref.size)))
targets,_ = resolve_backend_targets(requested_names=("FakeMarrakesh",),
    preferred_fake_backends=("FakeMarrakesh",), allow_preferred_fallback=False,
    fallback_mode="single", allow_runtime_lookup=False)
EX=-2.1414935642714075
ROUNDS={3,6,8,9,12,15}
cache={}
def tup(labels):
    key=tuple(labels)
    if key in cache: return cache[key]
    ops=[pool[l].term for l in labels]
    _l,circ=build_structural_ansatz_circuit(ops,nq=nq,ref_state=ref,structure_theta_value=1.0)
    comp=compile_circuit_for_backend(circ,targets[0].backend_obj,seed_transpiler=7,
        optimization_level=1,initial_layout=None)["compiled"]
    st=compiled_gate_stats(comp)
    out=(int(st["compiled_count_2q"]),int(st["compiled_depth_2q"]),int(safe_circuit_depth(comp)))
    cache[key]=out; return out

for path in sys.argv[1:]:
    d=json.loads((Path(path)/"result.json").read_text())
    tr=d["run"]["accepted_trajectory"]
    errs=[abs(float(x.get("accepted_energy", x.get("energy")))-EX) for x in tr]
    first=next((i+1 for i,e in enumerate(errs) if e<=1e-6),None)
    print(f"\n=== {Path(path).name}  first<=1e-6={first}  err@{len(tr)}={errs[-1]:.2e}")
    prev=[]
    for row in tr:
        n=int(row["controller_round"])
        labels=[str(x) for x in row["operators"]]
        added=[l for l in labels if labels.count(l)>prev.count(l)]
        prev=labels
        mark=""
        if n in ROUNDS:
            t=tup(labels)
            mark=f"  N2q={t[0]:>4} D2q={t[1]:>4} Dc={t[2]:>5}"
        print(f"  r{n:>2} err={errs[n-1]:.2e}{mark}  +{','.join(added) if added else '(none)'}")
