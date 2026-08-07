from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR_PATH = REPO_ROOT / "chtc/validate_condor_submit_lifecycle.py"
FACTORIAL_SUBMIT = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "ra_always_factorial48_r50_20260730_v1_chtc_activation/submit.sub"
)
GLOBAL_SINGLETON_SUBMIT = (
    REPO_ROOT
    / "chtc/paper_i_ra_adapt_repair_20260727/"
    "paper_i_ra_adapt_global_singleton_insertion12_r50_20260730_v1_"
    "chtc_activation/submit.sub"
)
PRESERVED_SUBMITTED_DEFECTS = frozenset(
    (FACTORIAL_SUBMIT, GLOBAL_SINGLETON_SUBMIT)
)


def _load_validator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "validate_condor_submit_lifecycle", VALIDATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    return _load_validator()


def test_parse_submit_assignments_preserves_repeated_values(
    validator: ModuleType,
) -> None:
    observed = validator.parse_submit_assignments(
        """
        # comment
        max_materialize = 4
        Leave_In_Queue = False
        leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)
        queue 48
        """
    )

    assert observed == {
        "max_materialize": ("4",),
        "leave_in_queue": (
            "False",
            "(JobStatus == 4) && (ExitCode =!= 0)",
        ),
    }


@pytest.mark.parametrize(
    "text",
    (
        "max_materialize = 4\nleave_in_queue = True\nqueue 48\n",
        "MAX_MATERIALIZE=(1)\nLEAVE_IN_QUEUE=((TRUE))\nqueue 12\n",
    ),
)
def test_bounded_factory_rejects_unconditional_completion_retention(
    text: str,
    validator: ModuleType,
) -> None:
    with pytest.raises(
        validator.SubmitLifecycleError,
        match="successful jobs never free factory slots",
    ):
        validator.validate_submit_lifecycle(text)


@pytest.mark.parametrize(
    "text",
    (
        "queue 12\n",
        "max_materialize = 1\nleave_in_queue = False\nqueue 12\n",
        (
            "max_materialize = 1\n"
            "leave_in_queue = "
            "(JobStatus == 4) && (ExitCode =!= 0)\n"
            "queue 12\n"
        ),
    ),
)
def test_safe_factory_lifecycle_policies_pass(
    text: str,
    validator: ModuleType,
) -> None:
    validator.validate_submit_lifecycle(text)


def test_safe_ordinary_held_exact_proc_release_mode_passes(
    validator: ModuleType,
) -> None:
    validator.validate_submit_lifecycle(
        """
        +HolsteinLifecycleMode = "ordinary_held_exact_proc_release_v1"
        hold = True
        periodic_release = False
        leave_in_queue = (JobStatus == 4) && (ExitCode =!= 0)
        queue execution_id from queue.tsv
        """
    )


@pytest.mark.parametrize(
    ("text", "message"),
    (
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "periodic_release = False\n"
            "queue 12\n",
            "requires exactly one hold=True assignment",
        ),
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "hold = False\n"
            "queue 12\n",
            "requires exactly one hold=True assignment",
        ),
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "hold = True\n"
            "max_materialize = 1\n"
            "queue 12\n",
            "must not use max_materialize or max_idle",
        ),
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "hold = True\n"
            "max_idle = 0\n"
            "queue 12\n",
            "must not use max_materialize or max_idle",
        ),
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "hold = True\n"
            "periodic_release = True\n"
            "queue 12\n",
            "requires periodic_release=False",
        ),
        (
            '+HolsteinLifecycleMode = '
            '"ordinary_held_exact_proc_release_v1"\n'
            "hold = True\n"
            "periodic_release = False\n"
            "leave_in_queue = True\n"
            "queue 12\n",
            "must not retain successful released jobs",
        ),
        (
            '+HolsteinLifecycleMode = "unknown_mode"\n'
            "hold = True\n"
            "queue 12\n",
            "unsupported HolsteinLifecycleMode",
        ),
    ),
)
def test_ordinary_held_mode_rejects_unsafe_lifecycle(
    text: str,
    message: str,
    validator: ModuleType,
) -> None:
    with pytest.raises(validator.SubmitLifecycleError, match=message):
        validator.validate_submit_lifecycle(text)


@pytest.mark.parametrize(
    "text",
    (
        "max_materialize = 0\nqueue 12\n",
        "max_materialize = $(limit)\nqueue 12\n",
        "max_idle = 1\nqueue 12\n",
    ),
)
def test_factory_mode_rejects_unbounded_or_unprovable_materialization(
    text: str,
    validator: ModuleType,
) -> None:
    with pytest.raises(
        validator.SubmitLifecycleError,
        match="positive constant max_materialize",
    ):
        validator.validate_submit_lifecycle(text)


@pytest.mark.parametrize(
    "submit_path",
    (FACTORIAL_SUBMIT, GLOBAL_SINGLETON_SUBMIT),
)
def test_validator_detects_preserved_submitted_factory_defect(
    submit_path: Path,
    validator: ModuleType,
) -> None:
    blockers = validator.factory_lifecycle_blockers(
        submit_path.read_text(encoding="utf-8")
    )

    assert blockers == (
        "positive max_materialize is incompatible with unconditional "
        "leave_in_queue=True because successful jobs never free factory slots",
    )


def test_repository_has_no_other_bounded_factory_lifecycle_defects(
    validator: ModuleType,
) -> None:
    defective_submit_files = frozenset(
        submit_path
        for submit_path in (REPO_ROOT / "chtc").rglob("*.sub")
        if validator.factory_lifecycle_blockers(
            submit_path.read_text(encoding="utf-8")
        )
    )

    assert defective_submit_files == PRESERVED_SUBMITTED_DEFECTS
