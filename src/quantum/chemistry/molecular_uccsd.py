from __future__ import annotations

from src.quantum.vqe_latex_python_pairs import AnsatzTerm, HardcodedUCCSDAnsatz


_MATH_UCCSD_POOL = r"\\mathcal{P}_{\\mathrm{UCCSD}} = \\{\\tau_\\mu - \\tau_\\mu^\\dagger\\}_{\\mu \\in \\mathrm{singles} \\cup \\mathrm{doubles}}"


def build_molecular_uccsd_pool(
    *,
    n_spatial_orbitals: int,
    num_particles: tuple[int, int],
    ordering: str = "blocked",
) -> list[AnsatzTerm]:
    if str(ordering).strip().lower() != "blocked":
        raise ValueError("Chemistry-local UCCSD prototype currently supports ordering='blocked' only.")
    ansatz = HardcodedUCCSDAnsatz(
        dims=int(n_spatial_orbitals),
        num_particles=tuple(int(x) for x in num_particles),
        reps=1,
        repr_mode="JW",
        indexing="blocked",
        include_singles=True,
        include_doubles=True,
    )
    return list(ansatz.base_terms)
