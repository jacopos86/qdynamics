from __future__ import annotations

import pytest

from pipelines.static_adapt.route_a_child_padding import (
    ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
    RouteAChildPaddingConfig,
)


def test_full_binary_code_space_accepts_h2o_nph1_layout() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
        problem_key="molecular_vibronic_h2o_linear_fd",
        num_sites=6,
        n_ph_max=1,
        boson_encoding="binary",
        total_register_width=15,
    )

    assert config.policy == ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1
    assert config.n_ph_max == 1


def test_full_binary_code_space_accepts_h2o_nph3_layout() -> None:
    config = RouteAChildPaddingConfig(
        policy=ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
        problem_key="molecular_vibronic_h2o_linear_fd",
        num_sites=6,
        n_ph_max=3,
        boson_encoding="binary",
        total_register_width=18,
    )

    assert config.policy == ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1
    assert config.n_ph_max == 3


@pytest.mark.parametrize(
    ("n_ph_max", "encoding"),
    [(2, "binary"), (1, "gray")],
)
def test_full_binary_code_space_rejects_truncated_or_nonbinary_registers(
    n_ph_max: int,
    encoding: str,
) -> None:
    with pytest.raises(ValueError):
        RouteAChildPaddingConfig(
            policy=ROUTE_A_CHILD_PADDING_FULL_BINARY_CODE_SPACE_V1,
            problem_key="molecular_vibronic_h2o_linear_fd",
            num_sites=6,
            n_ph_max=n_ph_max,
            boson_encoding=encoding,
            total_register_width=15,
        )
