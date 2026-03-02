### Optional PAOP-LF tweaks (recommended defaults)

**1) Force `paop_r >= 1` for `paop_lf_full` (match prior `paop_full` behavior).**  
Rationale: `paop_lf_full` is intended to include *extended cloud* terms; allowing `paop_r=0` silently removes nonlocal dressing and can weaken the “LF-full” meaning.

Suggested behavior:
- If `--adapt-pool paop_lf_full` and `--paop-r 0`, internally promote to `paop_r = 1`.

**2) Make `hop2` phonon-identity dropping user-configurable (keep current default ON).**  
Current: `paop_hop2(i,j) = K_ij (p_i - p_j)^2` then drop terms that are identity on all phonon qubits.  
Rationale: dropping prevents `hop2` from degenerating into a pure `K_ij` copy (can distort ADAPT selection), but it may be useful to toggle for ablation.

Suggested CLI flag:
- `--paop-hop2-drop-phonon-identity {0,1}` (default `1`)