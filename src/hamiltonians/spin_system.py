


def build_spin_hamiltonian(
    spin_config=None,
    coupling_params=None,
    **kwargs,
):
    """
    Spin Hamiltonian builder (Heisenberg, etc.)
    """

    # ----------------------------------------
    # TODO
    # ----------------------------------------
    H = None

    return ProblemHamiltonianData(
        problem_type="spin",
        H=H,
        metadata={
            "model": "generic_spin",
        },
    )