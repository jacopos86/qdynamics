



def build_molecular_electronic_hamiltonian(
    backend_data=None,
    backend="psi4",
    **kwargs,
):
    """
    Build electronic Hamiltonian from backend data.

    backend_data MUST come from:
        - psi4
        - pyscf
        - QE (later)
    """

    if backend_data is None:
        raise ValueError("backend_data is required")

    # ----------------------------------------
    # TODO: extract integrals
    # ----------------------------------------
    h1 = getattr(backend_data, "h1", None)
    h2 = getattr(backend_data, "h2", None)

    # ----------------------------------------
    # TODO: build second-quantized Hamiltonian
    # ----------------------------------------
    H = None

    return ProblemHamiltonianData(
        problem_type="molecular",
        H=H,
        basis=getattr(backend_data, "basis", None),
        backend_data=backend_data,
        metadata={
            "backend": backend,
        },
    )