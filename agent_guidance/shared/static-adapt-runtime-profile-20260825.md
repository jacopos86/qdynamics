# Static ADAPT runtime profile — where the time actually goes (2026-08-25)

Answer to: *"how to speed up our basic static ADAPT — should we use PennyLane,
PyTorch, CUDA, or is it already maximally quick?"*

**Short answer: none of those three.** The statevector numerics are **5.5% of
wall clock**. The other 94.5% is pure-Python Pauli-string bookkeeping, defensive
`deepcopy`, and Qiskit transpilation for cost metrics. PennyLane/PyTorch/CUDA all
attack the 5.5%; Amdahl caps that at **1.06×** even if the linear algebra became
free. The available headroom without any new dependency is **~2×**, and it is all
in code this repo already owns.

## 1. What was measured

`cProfile` over a canonical Paper-I Hubbard--Holstein ADAPT run:

| Setting | Value |
|---|---|
| Problem | `hh`, `L=2`, `t=1.0`, `U=8.0`, `omega0=1.0`, `g_ep=2.0` |
| Boson truncation | `n_ph_max=7`, binary encoding, `blocked`, PBC |
| Register | 4 fermion + 2 sites x 3 boson = **10 qubits**, `dim = 2^10 = 1024` |
| Pool | `hva` (87 available candidates per iteration) |
| Inner optimizer | SPSA, `maxiter=400`, `adapt_reopt_policy="full"` |
| Reached | `depth=30` (continuation `phase3_v1` drove past `max_depth=10`) |
| Entry point | `pipelines.static_adapt.adapt_pipeline._run_hardcoded_adapt_vqe` |

Two runs: **172.7 s** cold cache, **108.6 s** warm (`raw_outputs/cache` populated
by the phase-0 screen). Buckets below are the warm run, which is the steady state.

Profile artifact: `<scratchpad>/adapt.prof`; driver `<scratchpad>/prof_adapt.py`.

## 2. Cost buckets (warm run, 107.8 s profiled, `tottime`)

|  Time |  Share | Bucket |
|------:|-------:|---|
| 31.7 s | 29.4% | builtins/C (`sum`, `sorted`, `len`, `isinstance`, `str.lower/strip/replace`, `dict.get`) |
| 23.3 s | 21.6% | `pipelines/static_adapt/sector_invariants.py` — Pauli-word string algebra |
| 16.0 s | 14.9% | other repo Python |
| 11.5 s | 10.7% | `copy.deepcopy` |
| 11.0 s | 10.2% | Qiskit circuit build + transpile |
| **5.9 s** | **5.5%** | **statevector numerics** (`pauli_actions`, `compiled_ansatz`, `compiled_polynomial`) |
| 2.8 s | 2.6% | numpy misc |
| 2.4 s | 2.3% | JSON encode/decode |
| 2.0 s | 1.8% | hashing / state fingerprints |

The 29.4% "builtins" bucket is not independent: `str.lower` (1.41 s), `str.strip`
(1.26 s), `str.replace` (1.22 s) and `sorted` (2.36 s) are all called *from*
`_canonical_pauli_word`, and `sum` (2.75 s) is the `_pauli_words_commute`
generator. Charged to their callers, the **sector-invariant string algebra is
~32 s, or 30% of the run, in one subsystem.**

### Top functions, warm run

```
ncalls      tottime  cumtime  function
11,225,716   10.477   16.330  sector_invariants.py:146  _canonical_pauli_word
12,202,468    6.959   15.610  copy.py:118               deepcopy
 5,876,641    5.688    5.688  qubitization_module.py:48 PauliTerm.pw2strng
 5,369,198    5.375   27.619  sector_invariants.py:209  _pauli_words_commute
     1,742    4.631    4.631  {vf2_layout_pass_average}          (Qiskit)
20,467,882    3.347    3.347  sector_invariants.py:216  <genexpr>
       687    3.225   38.973  sector_invariants.py:226  audit_generator_sector_contract
   788,740    3.035    3.863  pauli_actions.py:231      apply_compiled_pauli   <-- the physics
     1,762    1.782    2.854  adapt_circuit_execution.py:63 _reference_state_digest
       871    1.692    1.692  {sabre_layout_and_routing}         (Qiskit)
     1,742    1.678    4.549  quantumcircuit.py:4082    QuantumCircuit.depth
```

`audit_generator_sector_contract` is **38.97 s cumulative — 36% of the run — from
687 calls.** It is a *verification* pass, not physics.

The cold run additionally showed `projective_state_fingerprint` at 15.2 s / 26.0 s
cum and `json.encoder.iterencode` at 12.5 s; both collapse once the phase-0 disk
cache is warm, so they are a **cold-start** tax, not a steady-state one.

## 3. Why PennyLane / PyTorch / CUDA are the wrong lever here

The register is `dim = 1024`, i.e. **16 KB of `complex128`** — it lives in L1.

Measured kernel cost (`src/quantum/pauli_actions.py`):

| `nq` | `dim` | `apply_compiled_pauli` | `apply_exp_term` | bare `np.copy` floor |
|---:|---:|---:|---:|---:|
| 10 | 1,024 | 3.86 us | 6.69 us | 0.50 us |
| 14 | 16,384 | 50.16 us | 68.37 us | 4.82 us |
| 20 | 1,048,576 | 7,316.70 us | 11,727.81 us | 1,613.12 us |

- **CUDA**: a kernel launch is ~5--10 us. The *entire* `apply_exp_term` at `nq=10`
  is 6.69 us. A GPU round trip is slower than the whole operation, before any
  host/device transfer. There is also no CUDA on this machine (`darwin`, 8 CPUs,
  16 GB) — MPS at best. GPU only becomes defensible at `nq >~ 20`, and only if the
  entire ADAPT loop lives on-device.
- **PyTorch**: higher per-op dispatch overhead than NumPy on 16 KB arrays. It buys
  autograd — but the inner optimizers in use are SPSA/POWELL/ROTOSOLVE, which are
  gradient-free by construction, and the ADAPT selection gradient is already an
  exact commutator (`gradient_source: "exact_commutator"`), not backprop. Nothing
  to differentiate.
- **PennyLane**: a framework layer *above* the kernel this repo already has. It
  would add dispatch overhead to the 5.5% and would not touch the 94.5%. It also
  imports its own Pauli/operator abstractions, which is precisely the layer whose
  string handling is the current bottleneck.

None are installed (`torch`, `pennylane`, `jax`, `numba`, `cupy`: absent).

## 4. Ranked, evidence-backed speedups

Measured micro-benchmarks, not estimates:

### (1) Memoize `audit_generator_sector_contract` — up to ~36%

`sector_invariants.py:226`. It is a pure function of
`(serialized polynomial, groups, total_qubits, tolerance)`. The pool is fixed
across ADAPT iterations, so the same generators are re-audited every depth. Key it
on `canonical_sha256(serialize_polynomial_terms_exyz(polynomial))` plus the group
signature. **This alone subsumes most of items (2)--(4).**

### (2) `lru_cache` on `_canonical_pauli_word` — measured 7.5x

`sector_invariants.py:146`. Pure `str -> str`, called 11.2M times on a vocabulary
of ~87 ten-character words.

```
current   0.387 us/call
lru_cache 0.052 us/call     -> 7.5x
```

Saves ~9 s (8%) directly, plus most of the `str.lower/strip/replace/sorted`
builtins charged above. One decorator.

### (3) Bitmask `_pauli_words_commute` — measured 12.3x

`sector_invariants.py:209`. Two Pauli words commute iff
`popcount(x1 & z2) + popcount(x2 & z1)` is even. The `(x, z)` masks are already
computed elsewhere in this repo by
`src/quantum/pauli_actions.compile_pauli_action_exyz` (`flip_mask`, `phase_mask`);
`sector_invariants` does not use them and re-derives everything from strings.

```
current (string zip + genexpr)  1.498 us/call
bitmask popcount parity         0.121 us/call   -> 12.3x
agreement over 1,600 word pairs: 0 disagreements
```

Saves ~8 s (7%), including the 20.5M-call `<genexpr>` at line 216.

### (4) Cache `PauliTerm.pw2strng()` — ~5%

`src/quantum/qubitization_module.py:48` rebuilds the word by per-qubit string
concatenation on every call: 5.9M calls, 5.69 s. Cache lazily on the term and
invalidate on mutation (or hoist it out of the `_commutator_l1_norm` double loop,
where the right-hand word is recomputed once per left-hand term).

### (5) Stop deep-copying candidate-cache reads — ~14%

All 12.2M `deepcopy` calls trace to
`pipelines/static_adapt/adapt_candidate_record_cache.py:208
_candidate_record_cache_get` (**15.19 s attributed**). It defensively deep-copies
every cached record on retrieval. Return frozen/immutable records — or copy only
on the write path — and this bucket goes to ~0.

### (6) Cache or elide the Qiskit transpile — ~10%

`vf2_layout_pass_average` 4.63 s + `sabre_layout_and_routing` 1.69 s +
`QuantumCircuit.depth` 1.68 s + `_append_standard_gate` 1.15 s. This is
`transpile(..., optimization_level=1)` computing two-qubit count / depth cost
metrics. Circuits repeat across candidates and depths — memoize by circuit
digest, drop to `optimization_level=0` for cost proxies, or compute the two-qubit
count analytically from the rotation layout.

### (7) Turn on the parallelism that already exists

The profiled run logged `gradient_parallel_backend: "serial"`,
`gradient_parallel_effective_workers: 1`, and
`hardcoded_adapt_phase2_full_feature_parallel ... disabled_reason:
"workers_leq_one"` (that phase 2 stage alone was 10.1 s at one worker). The knob
is `--adapt-parallel-gradient-workers` (`cli_config.py:2736`), with CHTC-aware
CPU detection already in `adapt_worker_limits.py`. 8 cores are available locally.

### (8) Only after all of the above: fuse the statevector kernel

`apply_exp_term` at `nq=10` is 6.69 us against a 0.50 us single-copy floor — ~13x
off, because it makes 5--6 separate passes over the array (int64 gather, `*=
signs`, `*= prefactor`, then `cos*psi - i sin*P psi`), each allocating a
temporary. A fused kernel would recover most of that. **But the whole bucket is
5.5%**, so the best case is ~5% — strictly less than any single item above.

## 5. Expected combined effect

Items (1)--(6) address roughly 45--50% of warm wall clock, i.e. **about 2x**, with
no new dependency and no change to any numerical result. Items (2) and (3) are
small, local, and independently verifiable; (1) and (5) are the large ones.

Item (7) multiplies on top of that for the pool-screening phases.

## 6. When the GPU question becomes real

The lever only inverts if the register grows. At `nq=20` the kernel is 7.3 ms per
Pauli application and the numerics would dominate. That means the honest trigger
is a change to `(L, n_ph_max)` — e.g. `L=4` at `n_ph_max=7` gives
`8 + 4x3 = 20` qubits — not a change of framework at `L=2`, `n_ph_max=7`. At
`dim = 1024` the machine is not the constraint; the bookkeeping is.

## 7. Reproduce

```
python3 <scratchpad>/prof_adapt.py       # cProfile -> adapt.prof, prints top 35
```

Caveat: the profiled configuration is one point in the matrix (HVA pool, SPSA,
`n_ph_max=7`). The bucket *ordering* is expected to be stable across regimes
because `audit_generator_sector_contract` and the candidate-record cache are
pool-size driven, not regime driven — but a POWELL/ROTOSOLVE overlay with full
re-optimization will shift more weight onto the optimizer's repeated state
preparations, i.e. onto item (8). Worth re-profiling once (1)--(5) land.
