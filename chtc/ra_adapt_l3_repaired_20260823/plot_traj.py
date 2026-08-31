import json, math, sys, importlib.util, glob
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
REPO = Path("/Users/jakestrobel/local_repos/Holstein_test_fullclone_3.worktrees/appendix-demos-20260820")
FIGD = Path("/Users/jakestrobel/local_repos/ADAPT---Paper-I/Paper_I_overleaf_20260818/figures")
sys.path.insert(0, str(REPO))
from pipelines.hardcoded.adapt_circuit_execution import build_structural_ansatz_circuit
from pipelines.qiskit_backend_tools import (compile_circuit_for_backend, compiled_gate_stats,
    resolve_backend_targets, safe_circuit_depth)
from pipelines.static_adapt.ra_adapt.appendix_demos import (
    build_ra_adapt_appendix_pure_hubbard_problem as P,
    RAAdaptAppendixPureHubbardCandidateAdapter as A,
    ra_adapt_appendix_generator_representation as R)
S = Path(sys.argv[1]); OUT = Path(sys.argv[2])
EX = -2.1414935642714075; FLOOR = 1e-16
prob = P(num_sites=3)
pool = {str(c.label): c for c in A(candidate_representation_id=R(3)).executable_pool(prob).candidates}
ref = np.asarray(prob.reference_state.build_state(), dtype=complex).reshape(-1)
nq = int(round(math.log2(ref.size)))
tg, _ = resolve_backend_targets(requested_names=("FakeMarrakesh",), preferred_fake_backends=("FakeMarrakesh",),
    allow_preferred_fallback=False, fallback_mode="single", allow_runtime_lookup=False)
_c = {}
def tup(l):
    k = tuple(l)
    if k in _c: return _c[k]
    _x, c = build_structural_ansatz_circuit([pool[i].term for i in l], nq=nq, ref_state=ref, structure_theta_value=1.0)
    comp = compile_circuit_for_backend(c, tg[0].backend_obj, seed_transpiler=7, optimization_level=1, initial_layout=None)["compiled"]
    st = compiled_gate_stats(comp)
    _c[k] = (int(st["compiled_count_2q"]), int(st["compiled_depth_2q"]), int(safe_circuit_depth(comp)))
    return _c[k]
def load(path):
    tr = json.load(open(glob.glob(path)[0]))["run"]["accepted_trajectory"]
    errs = [max(abs(float(x.get("accepted_energy", x.get("energy"))) - EX), FLOOR) for x in tr]
    n2q = [tup([str(o) for o in x["operators"]])[0] for x in tr]
    return list(range(1, len(tr) + 1)), errs, n2q
CURVES = [
    ("RA-Append",        f"{S}/final_matched/adaptive_baseline/result.json",            "#8B0000", "-"),
    ("RA + beam",        f"{S}/final_matched/adaptive_beam_escalated/result.json",      "#009E73", "-"),
    ("RA + batch",       f"{S}/final_matched/adaptive_batching/result.json",            "#E69F00", "-"),
    ("RA + prune (r9)",  f"{S}/prune_theta2/no_theta_screen/result.json",               "#0072B2", "-"),
    ("RA + prune (r15)", f"{S}/prune_theta2/theta1e-3/result.json",                     "#56B4E9", "--"),
]
spec = importlib.util.spec_from_file_location("st", FIGD / "paper_i_main_plot_style.py")
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st); st.configure()
fig = plt.figure(figsize=(7.10, 3.10))
ax = st.add_axes_grid(fig, nrows=1, ncols=2, left=0.085, right=0.995, bottom=0.42, top=0.86, hgap=0.075)[0]
handles = []
for label, path, color, ls in CURVES:
    try: r, e, n = load(path)
    except Exception: continue
    ax[0].plot(r, e, color=color, linestyle=ls, linewidth=st.PRIMARY_LINEWIDTH, solid_capstyle="round")
    ax[1].plot(r, n, color=color, linestyle=ls, linewidth=st.PRIMARY_LINEWIDTH, solid_capstyle="round")
    handles.append(Line2D([], [], color=color, linestyle=ls, linewidth=st.PRIMARY_LINEWIDTH, label=label))
for a in ax:
    a.axvline(12, color="0.55", linewidth=0.6, linestyle=(0, (3, 2)), zorder=0)
    a.axvspan(9, 11, color="0.85", alpha=0.45, zorder=0)
st.style_log_axis(ax[0], title="(a) Same-cutoff energy error", xlim=(0, 15.4),
    xticks=range(0, 16, 3), ylim=(6e-17, 4.0),
    yticks=[1e0, 1e-4, 1e-8, 1e-12, 1e-16], show_ylabels=True)
ax[0].yaxis.set_minor_formatter(NullFormatter())
ax[1].set_title("(b) Compiled $N_{2q}$", fontsize=st.TITLE_SIZE)
ax[1].set_xlim(0, 15.4); ax[1].set_xticks(range(0, 16, 3))
ax[1].set_xlabel("ADAPT iteration $k$", fontsize=st.AXIS_LABEL_SIZE)
ax[1].set_ylabel("$N_{2q}$", fontsize=st.AXIS_LABEL_SIZE)
ax[1].grid(True, linewidth=st.GRID_LINEWIDTH, alpha=st.GRID_ALPHA)
ax[1].tick_params(labelsize=st.TICK_LABEL_SIZE)
ax[0].set_xlabel("ADAPT iteration $k$", fontsize=st.AXIS_LABEL_SIZE)
st.add_top_legend(fig, handles, ncol=5)
fig.text(0.5, 0.035, "shaded r9-r11: prune advantage window;  dashed line: k=12 marker."
         "  Tight refit 1e-5/1e-12, adaptive shortlists, phase 0, compiled marginals.",
         ha="center", fontsize=5.6, color="0.30")
fig.savefig(OUT, dpi=400)
print("wrote", OUT)
