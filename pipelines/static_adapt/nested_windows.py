#!/usr/bin/env python3
"""Logical nested-window helpers for static ADAPT admission refits.

The selected inherited old-coordinate window and the post-admission active refit
coordinates are deliberately separate.  Slice A adds these pure helpers only;
pipeline wiring remains deferred.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class NestedWindowError(ValueError):
    """Raised when a nested refit window violates index invariants."""


COMPILE_PROXY_BASIS_OLD_PRE_INHERITED = "old_pre_inherited"
COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST = "optimizer_active_post"
_COMPILE_PROXY_BASIS_ALIASES = {
    "old_pre": COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    "old_pre_inherited": COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    "selection_inherited_old": COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    "selection_inherited_old_pre": COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
    "active_post": COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST,
    "optimizer_active_post": COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST,
    "optimizer_active_refit": COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST,
    "optimizer_active_refit_post": COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST,
}


@dataclass(frozen=True)
class NestedRefitWindow:
    window_version: str
    origin: str
    policy_requested: str
    policy_effective: str
    pre_parameter_count: int
    post_parameter_count: int
    position_id: int
    candidate_post_index: int
    old_pre_indices: tuple[int, ...]
    old_post_indices: tuple[int, ...]
    active_post_indices: tuple[int, ...]
    window_new_post_indices: tuple[int, ...]
    window_age_post_indices: tuple[int, ...]
    periodic_full_refit_triggered: bool = False


@dataclass(frozen=True)
class ActiveDormantNestedWindow:
    """Explicit Route-C active+dormant coordinate window.

    ``context_old_pre_indices`` is ordered as active inherited coordinates first,
    then dormant zero coordinates.  ``optimizer_active_post_indices`` maps that
    full context through the candidate insertion and includes the candidate post
    index, so dormant zero coordinates are present in the future plateau trial
    vector instead of being frozen away.
    """

    window_version: str
    origin: str
    pre_parameter_count: int
    post_parameter_count: int
    position_id: int
    candidate_post_index: int
    active_old_pre_indices: tuple[int, ...]
    dormant_old_pre_indices: tuple[int, ...]
    context_old_pre_indices: tuple[int, ...]
    active_old_post_indices: tuple[int, ...]
    dormant_old_post_indices: tuple[int, ...]
    context_old_post_indices: tuple[int, ...]
    optimizer_active_post_indices: tuple[int, ...]


@dataclass(frozen=True)
class NestedWindowAccounting:
    """Explicit accounting for one nested ADAPT admission window.

    ``selection_inherited_old_*`` is the old-coordinate basis used by reduced
    selector geometry. ``optimizer_active_refit_*`` is the post-insertion basis
    passed to the optimizer. ``compile_proxy_refit_count`` is intentionally
    derived from the named ``compile_proxy_basis`` so compile-cost telemetry does
    not silently conflate those two coordinate systems.
    """

    accounting_version: str
    window_version: str
    compile_proxy_basis: str
    candidate_post_index: int
    old_pre_indices: tuple[int, ...]
    active_post_indices: tuple[int, ...]
    selection_inherited_old_indices: tuple[int, ...]
    optimizer_active_refit_indices: tuple[int, ...]
    selection_inherited_old_count: int
    optimizer_active_refit_count: int
    compile_proxy_refit_count: int


@dataclass(frozen=True)
class NestedBatchWindowAccounting:
    """Explicit accounting for a composed batch nested refit window."""

    accounting_version: str
    window_version: str
    compile_proxy_basis: str
    candidate_post_indices: tuple[int, ...]
    old_pre_indices: tuple[int, ...]
    old_post_indices: tuple[int, ...]
    active_post_indices: tuple[int, ...]
    selection_inherited_old_indices: tuple[int, ...]
    optimizer_active_refit_indices: tuple[int, ...]
    selection_inherited_old_count: int
    optimizer_active_refit_count: int
    compile_proxy_refit_count: int


def _unique_ints(values: Sequence[int] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        item = int(value)
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def _validate_pre_index(index: int, *, pre_parameter_count: int) -> None:
    if int(index) < 0 or int(index) >= int(pre_parameter_count):
        raise NestedWindowError(
            f"Old coordinate index {index!r} is outside pre-insertion range "
            f"[0, {int(pre_parameter_count)})."
        )


def _int_tuple(values: Sequence[Any] | None) -> tuple[int, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes, bytearray)):
        raise NestedWindowError("Index payload must be a sequence of integers, not a string.")
    return tuple(int(x) for x in values)


def _normalize_compile_proxy_basis(basis: str | None) -> str:
    key = str(basis or COMPILE_PROXY_BASIS_OLD_PRE_INHERITED).strip().lower()
    normalized = _COMPILE_PROXY_BASIS_ALIASES.get(key)
    if normalized is None:
        raise NestedWindowError(
            f"Unknown compile_proxy_basis {basis!r}; expected one of "
            f"{sorted(set(_COMPILE_PROXY_BASIS_ALIASES.values()))}."
        )
    return str(normalized)


def map_pre_to_post_index(pre_index: int, position_id: int) -> int:
    """Map an old pre-insertion coordinate through one candidate insertion."""

    i = int(pre_index)
    p = int(position_id)
    if i < 0:
        raise NestedWindowError(f"pre_index must be nonnegative, got {pre_index!r}.")
    if p < 0:
        raise NestedWindowError(f"position_id must be nonnegative, got {position_id!r}.")
    return int(i if i < p else i + 1)


def map_post_to_pre_old_index(post_index: int, position_id: int) -> int | None:
    """Invert ``map_pre_to_post_index`` for old coordinates; candidate maps to None."""

    j = int(post_index)
    p = int(position_id)
    if j < 0:
        raise NestedWindowError(f"post_index must be nonnegative, got {post_index!r}.")
    if p < 0:
        raise NestedWindowError(f"position_id must be nonnegative, got {position_id!r}.")
    if j == p:
        return None
    return int(j if j < p else j - 1)


def _old_pre_indices_for_windowed_policy(
    *,
    pre_parameter_count: int,
    window_size: int,
    window_topk: int,
    window_new_pre_indices: Sequence[int] | None,
    window_age_pre_indices: Sequence[int] | None,
    theta_pre: Sequence[float] | None,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    new_pre = _unique_ints(window_new_pre_indices)
    age_pre_raw = _unique_ints(window_age_pre_indices)
    for idx in new_pre + age_pre_raw:
        _validate_pre_index(idx, pre_parameter_count=int(pre_parameter_count))

    if int(window_topk) > 0 and theta_pre is not None:
        theta = [float(x) for x in theta_pre]
        if len(theta) != int(pre_parameter_count):
            raise NestedWindowError(
                f"theta_pre length {len(theta)} does not match pre_parameter_count={pre_parameter_count}."
            )
        age_pre = tuple(
            sorted(age_pre_raw, key=lambda idx: (-abs(theta[int(idx)]), int(idx)))[: int(window_topk)]
        )
    elif int(window_topk) > 0:
        age_pre = age_pre_raw[: int(window_topk)]
    else:
        age_pre = age_pre_raw

    ordered_old: list[int] = []
    seen: set[int] = set()
    for idx in new_pre + age_pre:
        if int(idx) in seen:
            continue
        seen.add(int(idx))
        ordered_old.append(int(idx))
    if int(window_size) > 0:
        old_budget = max(0, int(window_size) - 1)  # candidate always consumes one active slot
        ordered_old = ordered_old[:old_budget]
    selected = tuple(ordered_old)
    selected_set = set(selected)
    selected_new = tuple(idx for idx in new_pre if int(idx) in selected_set)
    selected_age = tuple(idx for idx in age_pre if int(idx) in selected_set)
    return selected, selected_new, selected_age


def _strict_unique_int_tuple(values: Sequence[Any] | None, *, label: str) -> tuple[int, ...]:
    items = _int_tuple(values)
    if len(set(items)) != len(items):
        raise NestedWindowError(f"{label} must not contain duplicates.")
    return items


def predict_active_dormant_nested_window(
    *,
    pre_parameter_count: int,
    position_id: int,
    active_old_pre_indices: Sequence[int] | None = None,
    dormant_old_pre_indices: Sequence[int] | None = None,
    origin: str = "route_c_plateau_acquisition_v1",
) -> ActiveDormantNestedWindow:
    """Build a Route-C active+dormant post-insertion window.

    Active and dormant inputs are old-pre logical indices.  They must be disjoint
    because dormant records are zero-amplitude logical coordinates, not a second
    label for already-active coordinates.
    """

    pre_n = int(pre_parameter_count)
    if pre_n < 0:
        raise NestedWindowError(f"pre_parameter_count must be nonnegative, got {pre_n!r}.")
    pos = int(position_id)
    if pos < 0 or pos > pre_n:
        raise NestedWindowError(f"position_id={pos!r} is outside insertion range [0, {pre_n}].")
    active_old = _strict_unique_int_tuple(active_old_pre_indices, label="active_old_pre_indices")
    dormant_old = _strict_unique_int_tuple(dormant_old_pre_indices, label="dormant_old_pre_indices")
    for idx in active_old + dormant_old:
        _validate_pre_index(int(idx), pre_parameter_count=pre_n)
    overlap = set(active_old).intersection(dormant_old)
    if overlap:
        raise NestedWindowError(
            f"active and dormant old-pre indices must be disjoint; overlap={sorted(overlap)!r}."
        )
    context_old = tuple(active_old + dormant_old)
    active_post_old = tuple(map_pre_to_post_index(idx, pos) for idx in active_old)
    dormant_post_old = tuple(map_pre_to_post_index(idx, pos) for idx in dormant_old)
    context_post = tuple(active_post_old + dormant_post_old)
    candidate_post = int(pos)
    optimizer_active = tuple(sorted(set(context_post).union({candidate_post})))
    window = ActiveDormantNestedWindow(
        window_version="active_dormant_nested_window_v1",
        origin=str(origin),
        pre_parameter_count=int(pre_n),
        post_parameter_count=int(pre_n + 1),
        position_id=int(pos),
        candidate_post_index=int(candidate_post),
        active_old_pre_indices=tuple(int(x) for x in active_old),
        dormant_old_pre_indices=tuple(int(x) for x in dormant_old),
        context_old_pre_indices=tuple(int(x) for x in context_old),
        active_old_post_indices=tuple(int(x) for x in active_post_old),
        dormant_old_post_indices=tuple(int(x) for x in dormant_post_old),
        context_old_post_indices=tuple(int(x) for x in context_post),
        optimizer_active_post_indices=tuple(int(x) for x in optimizer_active),
    )
    validate_active_dormant_nested_window(window)
    return window


def validate_active_dormant_nested_window(window: ActiveDormantNestedWindow) -> None:
    pre_n = int(window.pre_parameter_count)
    post_n = int(window.post_parameter_count)
    pos = int(window.position_id)
    candidate = int(window.candidate_post_index)
    if post_n != pre_n + 1:
        raise NestedWindowError("post_parameter_count must equal pre_parameter_count + 1.")
    if pos < 0 or pos > pre_n:
        raise NestedWindowError(f"position_id={pos!r} is outside insertion range [0, {pre_n}].")
    if candidate != pos:
        raise NestedWindowError("candidate_post_index must equal position_id.")
    active_old = tuple(int(x) for x in window.active_old_pre_indices)
    dormant_old = tuple(int(x) for x in window.dormant_old_pre_indices)
    context_old = tuple(int(x) for x in window.context_old_pre_indices)
    if len(set(active_old)) != len(active_old):
        raise NestedWindowError("active_old_pre_indices must not contain duplicates.")
    if len(set(dormant_old)) != len(dormant_old):
        raise NestedWindowError("dormant_old_pre_indices must not contain duplicates.")
    if set(active_old).intersection(dormant_old):
        raise NestedWindowError("active and dormant old-pre indices must be disjoint.")
    if context_old != tuple(active_old + dormant_old):
        raise NestedWindowError("context_old_pre_indices must equal active_old_pre_indices plus dormant_old_pre_indices.")
    for idx in context_old:
        _validate_pre_index(int(idx), pre_parameter_count=pre_n)
    expected_active_post = tuple(map_pre_to_post_index(idx, pos) for idx in active_old)
    expected_dormant_post = tuple(map_pre_to_post_index(idx, pos) for idx in dormant_old)
    if tuple(window.active_old_post_indices) != expected_active_post:
        raise NestedWindowError("active_old_post_indices do not match mapped active_old_pre_indices.")
    if tuple(window.dormant_old_post_indices) != expected_dormant_post:
        raise NestedWindowError("dormant_old_post_indices do not match mapped dormant_old_pre_indices.")
    if tuple(window.context_old_post_indices) != tuple(expected_active_post + expected_dormant_post):
        raise NestedWindowError("context_old_post_indices must equal active_old_post_indices plus dormant_old_post_indices.")
    optimizer_active = tuple(int(x) for x in window.optimizer_active_post_indices)
    if len(set(optimizer_active)) != len(optimizer_active):
        raise NestedWindowError("optimizer_active_post_indices must not contain duplicates.")
    if candidate not in set(optimizer_active):
        raise NestedWindowError("optimizer_active_post_indices must include the inserted candidate coordinate.")
    expected_optimizer = tuple(sorted(set(window.context_old_post_indices).union({candidate})))
    if optimizer_active != expected_optimizer:
        raise NestedWindowError("optimizer_active_post_indices must equal mapped active+dormant context plus candidate.")
    for idx in tuple(window.context_old_post_indices) + optimizer_active:
        if int(idx) < 0 or int(idx) >= post_n:
            raise NestedWindowError(f"Post-insertion coordinate {idx!r} is outside [0, {post_n}).")


def serialize_active_dormant_nested_window(window: ActiveDormantNestedWindow) -> dict[str, Any]:
    validate_active_dormant_nested_window(window)
    payload = asdict(window)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = [int(x) for x in value]
    return payload


def active_dormant_nested_window_from_json(payload: Mapping[str, Any]) -> ActiveDormantNestedWindow:
    if not isinstance(payload, Mapping):
        raise NestedWindowError("Active+dormant nested-window payload must be a mapping.")
    window = ActiveDormantNestedWindow(
        window_version=str(payload.get("window_version", "active_dormant_nested_window_v1")),
        origin=str(payload.get("origin", "route_c_plateau_acquisition_v1")),
        pre_parameter_count=int(payload.get("pre_parameter_count", -1)),
        post_parameter_count=int(payload.get("post_parameter_count", -1)),
        position_id=int(payload.get("position_id", -1)),
        candidate_post_index=int(payload.get("candidate_post_index", -1)),
        active_old_pre_indices=_int_tuple(payload.get("active_old_pre_indices")),
        dormant_old_pre_indices=_int_tuple(payload.get("dormant_old_pre_indices")),
        context_old_pre_indices=_int_tuple(payload.get("context_old_pre_indices")),
        active_old_post_indices=_int_tuple(payload.get("active_old_post_indices")),
        dormant_old_post_indices=_int_tuple(payload.get("dormant_old_post_indices")),
        context_old_post_indices=_int_tuple(payload.get("context_old_post_indices")),
        optimizer_active_post_indices=_int_tuple(payload.get("optimizer_active_post_indices")),
    )
    validate_active_dormant_nested_window(window)
    return window


def predict_nested_refit_window(
    *,
    pre_parameter_count: int | None = None,
    theta_pre: Sequence[float] | None = None,
    position_id: int,
    policy: str = "append_only",
    window_size: int = 0,
    window_topk: int = 0,
    window_new_pre_indices: Sequence[int] | None = None,
    window_age_pre_indices: Sequence[int] | None = None,
    periodic_full_refit_triggered: bool = False,
    allowed_old_pre_indices: Sequence[int] | None = None,
) -> NestedRefitWindow:
    """Predict the nested post-admission active refit window.

    ``old_pre_indices`` is the inherited old-coordinate window used for candidate
    context. ``active_post_indices`` is the optimizer coordinate set after the
    candidate is inserted and always includes ``candidate_post_index``.
    """

    if pre_parameter_count is None:
        if theta_pre is None:
            raise NestedWindowError("pre_parameter_count or theta_pre is required.")
        pre_n = len(theta_pre)
    else:
        pre_n = int(pre_parameter_count)
    if pre_n < 0:
        raise NestedWindowError(f"pre_parameter_count must be nonnegative, got {pre_n!r}.")
    pos = int(position_id)
    if pos < 0 or pos > pre_n:
        raise NestedWindowError(f"position_id={pos!r} is outside insertion range [0, {pre_n}].")
    post_n = pre_n + 1
    requested = str(policy)
    effective = "full" if bool(periodic_full_refit_triggered) else requested

    if effective in {"full", "periodic_full"}:
        old_pre = tuple(range(pre_n))
        new_pre = old_pre
        age_pre: tuple[int, ...] = ()
        effective = "full"
    elif effective in {"append_only", "candidate_only"}:
        old_pre = ()
        new_pre = ()
        age_pre = ()
        effective = "append_only"
    elif effective in {"windowed", "nested", "nested_window"}:
        old_pre, new_pre, age_pre = _old_pre_indices_for_windowed_policy(
            pre_parameter_count=pre_n,
            window_size=int(window_size),
            window_topk=int(window_topk),
            window_new_pre_indices=window_new_pre_indices,
            window_age_pre_indices=window_age_pre_indices,
            theta_pre=theta_pre,
        )
        effective = "windowed"
    else:
        raise NestedWindowError(f"Unknown nested refit window policy: {policy!r}.")

    old_post = tuple(map_pre_to_post_index(idx, pos) for idx in old_pre)
    new_post = tuple(map_pre_to_post_index(idx, pos) for idx in new_pre if idx in set(old_pre))
    age_post = tuple(map_pre_to_post_index(idx, pos) for idx in age_pre if idx in set(old_pre))
    candidate_post = int(pos)
    active = tuple(sorted(set(old_post).union({candidate_post})))
    window = NestedRefitWindow(
        window_version="nested_refit_window_v1",
        origin="nested_inherited_v1",
        policy_requested=requested,
        policy_effective=effective,
        pre_parameter_count=int(pre_n),
        post_parameter_count=int(post_n),
        position_id=int(pos),
        candidate_post_index=int(candidate_post),
        old_pre_indices=tuple(int(x) for x in old_pre),
        old_post_indices=tuple(int(x) for x in old_post),
        active_post_indices=tuple(int(x) for x in active),
        window_new_post_indices=tuple(int(x) for x in new_post),
        window_age_post_indices=tuple(int(x) for x in age_post),
        periodic_full_refit_triggered=bool(periodic_full_refit_triggered),
    )
    validate_nested_window(window, allowed_old_pre_indices=allowed_old_pre_indices)
    return window


def validate_nested_window(
    window: NestedRefitWindow,
    *,
    allowed_old_pre_indices: Sequence[int] | None = None,
) -> None:
    """Validate logical nested-window index invariants."""

    pre_n = int(window.pre_parameter_count)
    post_n = int(window.post_parameter_count)
    pos = int(window.position_id)
    candidate = int(window.candidate_post_index)
    if post_n != pre_n + 1:
        raise NestedWindowError(
            f"post_parameter_count must equal pre_parameter_count + 1, got {post_n} and {pre_n}."
        )
    if pos < 0 or pos > pre_n:
        raise NestedWindowError(f"position_id={pos!r} is outside insertion range [0, {pre_n}].")
    if candidate != pos:
        raise NestedWindowError(
            f"candidate_post_index={candidate!r} must equal insertion position_id={pos!r}."
        )
    if candidate not in set(window.active_post_indices):
        raise NestedWindowError("active_post_indices must include the inserted candidate coordinate.")
    if len(set(window.active_post_indices)) != len(window.active_post_indices):
        raise NestedWindowError("active_post_indices must not contain duplicates.")
    if len(set(window.old_pre_indices)) != len(window.old_pre_indices):
        raise NestedWindowError("old_pre_indices must not contain duplicates.")
    for idx in window.active_post_indices + window.old_post_indices + window.window_new_post_indices + window.window_age_post_indices:
        if int(idx) < 0 or int(idx) >= post_n:
            raise NestedWindowError(f"Post-insertion coordinate {idx!r} is outside [0, {post_n}).")
    for idx in window.old_pre_indices:
        _validate_pre_index(int(idx), pre_parameter_count=pre_n)
    expected_old_post = tuple(map_pre_to_post_index(idx, pos) for idx in window.old_pre_indices)
    if tuple(window.old_post_indices) != expected_old_post:
        raise NestedWindowError(
            f"old_post_indices={window.old_post_indices!r} do not match mapped old_pre_indices "
            f"{expected_old_post!r}."
        )
    if candidate in set(window.window_new_post_indices).union(window.window_age_post_indices):
        raise NestedWindowError("window_new/window_age indices must describe inherited old coordinates only.")
    if not set(window.window_new_post_indices).issubset(set(window.old_post_indices)):
        raise NestedWindowError("window_new_post_indices must be a subset of old_post_indices.")
    if not set(window.window_age_post_indices).issubset(set(window.old_post_indices)):
        raise NestedWindowError("window_age_post_indices must be a subset of old_post_indices.")
    expected_active = set(window.old_post_indices).union({candidate})
    if set(window.active_post_indices) != expected_active:
        raise NestedWindowError(
            "active_post_indices must equal old_post_indices plus candidate_post_index."
        )
    if allowed_old_pre_indices is not None:
        allowed = {int(x) for x in allowed_old_pre_indices}
        actual = {int(x) for x in window.old_pre_indices}
        if not actual.issubset(allowed):
            raise NestedWindowError(
                f"old_pre_indices {sorted(actual)!r} are not a subset of allowed context {sorted(allowed)!r}."
            )


def build_nested_window_accounting(
    window: NestedRefitWindow,
    *,
    compile_proxy_basis: str = COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
) -> NestedWindowAccounting:
    """Build explicit count/basis accounting for ``window``.

    The default compile proxy basis is the inherited old-pre window. That matches
    the current static-ADAPT proxy convention: candidate synthesis is counted in
    the new-generator terms, while the refit proxy counts old coordinates whose
    amplitudes may be re-estimated. If a caller wants to model optimizer active
    dimensions directly, it must opt into ``optimizer_active_post`` explicitly.
    """

    validate_nested_window(window)
    basis = _normalize_compile_proxy_basis(compile_proxy_basis)
    old_pre = tuple(int(x) for x in window.old_pre_indices)
    active_post = tuple(int(x) for x in window.active_post_indices)
    if basis == COMPILE_PROXY_BASIS_OLD_PRE_INHERITED:
        compile_count = int(len(old_pre))
    elif basis == COMPILE_PROXY_BASIS_OPTIMIZER_ACTIVE_POST:
        compile_count = int(len(active_post))
    else:  # pragma: no cover - _normalize_compile_proxy_basis guards this.
        raise NestedWindowError(f"Unsupported compile_proxy_basis {basis!r}.")
    accounting = NestedWindowAccounting(
        accounting_version="nested_window_accounting_v1",
        window_version=str(window.window_version),
        compile_proxy_basis=str(basis),
        candidate_post_index=int(window.candidate_post_index),
        old_pre_indices=old_pre,
        active_post_indices=active_post,
        selection_inherited_old_indices=old_pre,
        optimizer_active_refit_indices=active_post,
        selection_inherited_old_count=int(len(old_pre)),
        optimizer_active_refit_count=int(len(active_post)),
        compile_proxy_refit_count=int(compile_count),
    )
    validate_nested_window_accounting(accounting, window=window)
    return accounting


def validate_nested_window_accounting(
    accounting: NestedWindowAccounting,
    *,
    window: NestedRefitWindow | None = None,
) -> None:
    """Validate nested-window accounting counts and optional window agreement."""

    basis = _normalize_compile_proxy_basis(accounting.compile_proxy_basis)
    old_pre = tuple(int(x) for x in accounting.old_pre_indices)
    active_post = tuple(int(x) for x in accounting.active_post_indices)
    selection_old = tuple(int(x) for x in accounting.selection_inherited_old_indices)
    optimizer_active = tuple(int(x) for x in accounting.optimizer_active_refit_indices)
    if basis != str(accounting.compile_proxy_basis):
        raise NestedWindowError(
            f"compile_proxy_basis must be normalized to {basis!r}, got {accounting.compile_proxy_basis!r}."
        )
    if old_pre != selection_old:
        raise NestedWindowError("old_pre_indices must equal selection_inherited_old_indices.")
    if active_post != optimizer_active:
        raise NestedWindowError("active_post_indices must equal optimizer_active_refit_indices.")
    if len(set(old_pre)) != len(old_pre):
        raise NestedWindowError("Accounting old_pre_indices must not contain duplicates.")
    if len(set(active_post)) != len(active_post):
        raise NestedWindowError("Accounting active_post_indices must not contain duplicates.")
    if int(accounting.candidate_post_index) not in set(active_post):
        raise NestedWindowError("Accounting active_post_indices must include candidate_post_index.")
    if int(accounting.selection_inherited_old_count) != len(selection_old):
        raise NestedWindowError("selection_inherited_old_count does not match selection_inherited_old_indices.")
    if int(accounting.optimizer_active_refit_count) != len(optimizer_active):
        raise NestedWindowError("optimizer_active_refit_count does not match optimizer_active_refit_indices.")
    expected_compile_count = (
        len(selection_old)
        if basis == COMPILE_PROXY_BASIS_OLD_PRE_INHERITED
        else len(optimizer_active)
    )
    if int(accounting.compile_proxy_refit_count) != int(expected_compile_count):
        raise NestedWindowError(
            "compile_proxy_refit_count does not match compile_proxy_basis."
        )
    if window is not None:
        validate_nested_window(window)
        if str(accounting.window_version) != str(window.window_version):
            raise NestedWindowError("Accounting window_version does not match nested window.")
        if int(accounting.candidate_post_index) != int(window.candidate_post_index):
            raise NestedWindowError("Accounting candidate_post_index does not match nested window.")
        if old_pre != tuple(int(x) for x in window.old_pre_indices):
            raise NestedWindowError("Accounting old_pre_indices do not match nested window.")
        if active_post != tuple(int(x) for x in window.active_post_indices):
            raise NestedWindowError("Accounting active_post_indices do not match nested window.")


def serialize_nested_window(window: NestedRefitWindow) -> dict[str, Any]:
    """Return a JSON-safe telemetry payload for a nested window."""

    payload = asdict(window)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = [int(x) for x in value]
    return payload


def serialize_nested_window_accounting(accounting: NestedWindowAccounting) -> dict[str, Any]:
    """Return a JSON-safe telemetry payload for nested-window accounting."""

    validate_nested_window_accounting(accounting)
    payload = asdict(accounting)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = [int(x) for x in value]
    return payload


def validate_composed_batch_window_payload(batch_window: Mapping[str, Any]) -> None:
    """Validate final-space invariants for a composed batch nested-window payload."""

    if not isinstance(batch_window, Mapping):
        raise NestedWindowError("Batch nested-window payload must be a mapping.")
    pre_n = int(batch_window.get("pre_parameter_count", -1))
    post_n = int(batch_window.get("post_parameter_count", -1))
    positions = _int_tuple(batch_window.get("positions_in_commit_order"))
    candidate_post = _int_tuple(batch_window.get("candidate_post_indices"))
    old_pre = _int_tuple(batch_window.get("old_pre_indices"))
    old_post = _int_tuple(batch_window.get("old_post_indices"))
    active_post = _int_tuple(batch_window.get("active_post_indices"))
    if pre_n < 0 or post_n < 0:
        raise NestedWindowError("Batch pre/post parameter counts are invalid.")
    if len(positions) != len(candidate_post):
        raise NestedWindowError("Batch positions_in_commit_order length must match candidate_post_indices length.")
    if post_n != pre_n + len(candidate_post):
        raise NestedWindowError(
            "Batch post_parameter_count must equal pre_parameter_count plus candidate count."
        )
    if len(set(candidate_post)) != len(candidate_post):
        raise NestedWindowError("Batch candidate_post_indices must not contain duplicates.")
    if len(set(old_pre)) != len(old_pre):
        raise NestedWindowError("Batch old_pre_indices must not contain duplicates.")
    if len(set(old_post)) != len(old_post):
        raise NestedWindowError("Batch old_post_indices must not contain duplicates.")
    if len(set(active_post)) != len(active_post):
        raise NestedWindowError("Batch active_post_indices must not contain duplicates.")
    for pos in positions:
        if int(pos) < 0 or int(pos) > pre_n:
            raise NestedWindowError(f"Batch insertion position {pos!r} is outside [0, {pre_n}].")
    for idx in old_pre:
        if int(idx) < 0 or int(idx) >= pre_n:
            raise NestedWindowError(f"Batch old_pre index {idx!r} is outside [0, {pre_n}).")
    for idx in old_post + candidate_post + active_post:
        if int(idx) < 0 or int(idx) >= post_n:
            raise NestedWindowError(f"Batch post index {idx!r} is outside [0, {post_n}).")
    expected_old_post = tuple(_map_original_old_index_after_insertions(idx, positions) for idx in old_pre)
    if old_post != expected_old_post:
        raise NestedWindowError(
            f"Batch old_post_indices={old_post!r} do not match mapped old_pre_indices {expected_old_post!r}."
        )
    expected_candidate_post = tuple(_candidate_final_index(i, positions) for i in range(len(positions)))
    if candidate_post != expected_candidate_post:
        raise NestedWindowError(
            "Batch candidate_post_indices do not match positions_in_commit_order in final post space."
        )
    expected_active = tuple(sorted(set(old_post).union(candidate_post)))
    if active_post != expected_active:
        raise NestedWindowError(
            "Batch active_post_indices must equal old_post_indices plus candidate_post_indices."
        )


def build_nested_batch_window_accounting(
    batch_window: Mapping[str, Any],
    *,
    compile_proxy_basis: str = COMPILE_PROXY_BASIS_OLD_PRE_INHERITED,
) -> NestedBatchWindowAccounting:
    """Build explicit count/basis accounting for a composed batch window."""

    if not isinstance(batch_window, Mapping):
        raise NestedWindowError("Batch nested-window payload must be a mapping.")
    basis = _normalize_compile_proxy_basis(compile_proxy_basis)
    validate_composed_batch_window_payload(batch_window)
    candidate_post = _int_tuple(batch_window.get("candidate_post_indices"))
    old_pre = _int_tuple(batch_window.get("old_pre_indices"))
    old_post = _int_tuple(batch_window.get("old_post_indices"))
    active_post = _int_tuple(batch_window.get("active_post_indices"))
    if len(set(candidate_post)) != len(candidate_post):
        raise NestedWindowError("Batch candidate_post_indices must not contain duplicates.")
    if len(set(old_pre)) != len(old_pre):
        raise NestedWindowError("Batch old_pre_indices must not contain duplicates.")
    if len(set(old_post)) != len(old_post):
        raise NestedWindowError("Batch old_post_indices must not contain duplicates.")
    if len(set(active_post)) != len(active_post):
        raise NestedWindowError("Batch active_post_indices must not contain duplicates.")
    if not set(candidate_post).issubset(set(active_post)):
        raise NestedWindowError("Batch active_post_indices must include every candidate_post_index.")
    pre_n = int(batch_window.get("pre_parameter_count", -1))
    post_n = int(batch_window.get("post_parameter_count", -1))
    if pre_n < 0 or post_n < pre_n:
        raise NestedWindowError("Batch pre/post parameter counts are invalid.")
    for idx in old_pre:
        if int(idx) < 0 or int(idx) >= pre_n:
            raise NestedWindowError(f"Batch old_pre index {idx!r} is outside [0, {pre_n}).")
    for idx in old_post + candidate_post + active_post:
        if int(idx) < 0 or int(idx) >= post_n:
            raise NestedWindowError(f"Batch post index {idx!r} is outside [0, {post_n}).")
    compile_count = len(old_pre) if basis == COMPILE_PROXY_BASIS_OLD_PRE_INHERITED else len(active_post)
    accounting = NestedBatchWindowAccounting(
        accounting_version="nested_window_batch_accounting_v1",
        window_version=str(batch_window.get("window_version", "nested_refit_window_batch_v1")),
        compile_proxy_basis=str(basis),
        candidate_post_indices=candidate_post,
        old_pre_indices=old_pre,
        old_post_indices=old_post,
        active_post_indices=active_post,
        selection_inherited_old_indices=old_pre,
        optimizer_active_refit_indices=active_post,
        selection_inherited_old_count=int(len(old_pre)),
        optimizer_active_refit_count=int(len(active_post)),
        compile_proxy_refit_count=int(compile_count),
    )
    validate_nested_batch_window_accounting(accounting, batch_window=batch_window)
    return accounting


def validate_nested_batch_window_accounting(
    accounting: NestedBatchWindowAccounting,
    *,
    batch_window: Mapping[str, Any] | None = None,
) -> None:
    """Validate composed batch nested-window accounting."""

    basis = _normalize_compile_proxy_basis(accounting.compile_proxy_basis)
    if basis != str(accounting.compile_proxy_basis):
        raise NestedWindowError(
            f"compile_proxy_basis must be normalized to {basis!r}, got {accounting.compile_proxy_basis!r}."
        )
    old_pre = tuple(int(x) for x in accounting.old_pre_indices)
    active_post = tuple(int(x) for x in accounting.active_post_indices)
    candidate_post = tuple(int(x) for x in accounting.candidate_post_indices)
    old_post = tuple(int(x) for x in accounting.old_post_indices)
    if old_pre != tuple(int(x) for x in accounting.selection_inherited_old_indices):
        raise NestedWindowError("Batch old_pre_indices must equal selection_inherited_old_indices.")
    if active_post != tuple(int(x) for x in accounting.optimizer_active_refit_indices):
        raise NestedWindowError("Batch active_post_indices must equal optimizer_active_refit_indices.")
    if len(set(candidate_post)) != len(candidate_post):
        raise NestedWindowError("Batch candidate_post_indices must not contain duplicates.")
    if len(set(old_pre)) != len(old_pre):
        raise NestedWindowError("Batch old_pre_indices must not contain duplicates.")
    if len(set(old_post)) != len(old_post):
        raise NestedWindowError("Batch old_post_indices must not contain duplicates.")
    if len(set(active_post)) != len(active_post):
        raise NestedWindowError("Batch active_post_indices must not contain duplicates.")
    if not set(candidate_post).issubset(set(active_post)):
        raise NestedWindowError("Batch active_post_indices must include candidate_post_indices.")
    if int(accounting.selection_inherited_old_count) != len(old_pre):
        raise NestedWindowError("Batch selection_inherited_old_count does not match indices.")
    if int(accounting.optimizer_active_refit_count) != len(active_post):
        raise NestedWindowError("Batch optimizer_active_refit_count does not match indices.")
    expected_compile_count = (
        len(old_pre)
        if basis == COMPILE_PROXY_BASIS_OLD_PRE_INHERITED
        else len(active_post)
    )
    if int(accounting.compile_proxy_refit_count) != int(expected_compile_count):
        raise NestedWindowError("Batch compile_proxy_refit_count does not match compile_proxy_basis.")
    if batch_window is not None:
        validate_composed_batch_window_payload(batch_window)
        if str(accounting.window_version) != str(batch_window.get("window_version", "nested_refit_window_batch_v1")):
            raise NestedWindowError("Batch accounting window_version does not match payload.")
        if candidate_post != _int_tuple(batch_window.get("candidate_post_indices")):
            raise NestedWindowError("Batch accounting candidate_post_indices do not match payload.")
        if old_pre != _int_tuple(batch_window.get("old_pre_indices")):
            raise NestedWindowError("Batch accounting old_pre_indices do not match payload.")
        if old_post != _int_tuple(batch_window.get("old_post_indices")):
            raise NestedWindowError("Batch accounting old_post_indices do not match payload.")
        if active_post != _int_tuple(batch_window.get("active_post_indices")):
            raise NestedWindowError("Batch accounting active_post_indices do not match payload.")


def serialize_nested_batch_window_accounting(
    accounting: NestedBatchWindowAccounting,
) -> dict[str, Any]:
    """Return a JSON-safe telemetry payload for batch nested-window accounting."""

    validate_nested_batch_window_accounting(accounting)
    payload = asdict(accounting)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = [int(x) for x in value]
    return payload


def nested_window_accounting_from_json(payload: Mapping[str, Any]) -> NestedWindowAccounting:
    """Round-trip a JSON payload back into ``NestedWindowAccounting``."""

    if not isinstance(payload, Mapping):
        raise NestedWindowError("Nested-window accounting payload must be a mapping.")
    accounting = NestedWindowAccounting(
        accounting_version=str(payload.get("accounting_version", "nested_window_accounting_v1")),
        window_version=str(payload.get("window_version", "nested_refit_window_v1")),
        compile_proxy_basis=_normalize_compile_proxy_basis(
            str(payload.get("compile_proxy_basis", COMPILE_PROXY_BASIS_OLD_PRE_INHERITED))
        ),
        candidate_post_index=int(payload.get("candidate_post_index", -1)),
        old_pre_indices=_int_tuple(payload.get("old_pre_indices")),
        active_post_indices=_int_tuple(payload.get("active_post_indices")),
        selection_inherited_old_indices=_int_tuple(
            payload.get("selection_inherited_old_indices", payload.get("old_pre_indices"))
        ),
        optimizer_active_refit_indices=_int_tuple(
            payload.get("optimizer_active_refit_indices", payload.get("active_post_indices"))
        ),
        selection_inherited_old_count=int(payload.get("selection_inherited_old_count", -1)),
        optimizer_active_refit_count=int(payload.get("optimizer_active_refit_count", -1)),
        compile_proxy_refit_count=int(payload.get("compile_proxy_refit_count", -1)),
    )
    validate_nested_window_accounting(accounting)
    return accounting


def _map_original_old_index_after_insertions(pre_index: int, positions: Sequence[int]) -> int:
    i = int(pre_index)
    return int(i + sum(1 for pos in positions if int(pos) <= i))


def _candidate_final_index(insert_idx: int, positions: Sequence[int]) -> int:
    p = int(positions[int(insert_idx)])
    before = sum(1 for pos in positions if int(pos) < p)
    same_before = sum(1 for j, pos in enumerate(positions) if j < int(insert_idx) and int(pos) == p)
    return int(p + before + same_before)


def build_composed_batch_window_payload(
    *,
    pre_parameter_count: int,
    positions_in_commit_order: Sequence[int],
    old_pre_indices: Sequence[int],
) -> dict[str, Any]:
    """Build and validate a final-space batch nested-window payload.

    Positions are interpreted in the original pre-batch logical coordinate
    system.  Same-position insertions are ordered by commit order.
    """

    pre_n = int(pre_parameter_count)
    if pre_n < 0:
        raise NestedWindowError("pre_parameter_count must be nonnegative.")
    positions = tuple(int(x) for x in positions_in_commit_order)
    for pos in positions:
        if int(pos) < 0 or int(pos) > pre_n:
            raise NestedWindowError(f"Batch insertion position {pos!r} is outside [0, {pre_n}].")
    old_pre = tuple(int(x) for x in old_pre_indices)
    if len(set(old_pre)) != len(old_pre):
        raise NestedWindowError("Batch old_pre_indices must not contain duplicates.")
    for idx in old_pre:
        if int(idx) < 0 or int(idx) >= pre_n:
            raise NestedWindowError(f"Batch old_pre index {idx!r} is outside [0, {pre_n}).")
    post_n = int(pre_n + len(positions))
    old_post = tuple(_map_original_old_index_after_insertions(idx, positions) for idx in old_pre)
    candidate_post = tuple(_candidate_final_index(i, positions) for i in range(len(positions)))
    active = tuple(sorted(set(old_post).union(candidate_post)))
    payload = {
        "window_version": "nested_refit_window_batch_v1",
        "pre_parameter_count": int(pre_n),
        "post_parameter_count": int(post_n),
        "positions_in_commit_order": [int(x) for x in positions],
        "candidate_post_indices": [int(x) for x in candidate_post],
        "old_pre_indices": [int(x) for x in old_pre],
        "old_post_indices": [int(x) for x in old_post],
        "active_post_indices": [int(x) for x in active],
    }
    validate_composed_batch_window_payload(payload)
    return payload


def compose_batch_nested_windows(
    windows: Sequence[NestedRefitWindow],
    positions_in_commit_order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compose single-record nested windows into one final batch refit payload.

    Positions are interpreted in the original pre-batch logical coordinate system,
    with same-position insertions ordered by commit order.
    """

    if not windows:
        return {
            "window_version": "nested_refit_window_batch_v1",
            "pre_parameter_count": 0,
            "post_parameter_count": 0,
            "positions_in_commit_order": [],
            "candidate_post_indices": [],
            "old_pre_indices": [],
            "old_post_indices": [],
            "active_post_indices": [],
        }
    pre_n = int(windows[0].pre_parameter_count)
    for window in windows:
        if int(window.pre_parameter_count) != pre_n:
            raise NestedWindowError("All batch windows must share one original pre_parameter_count.")
    positions = tuple(
        int(x)
        for x in (
            positions_in_commit_order if positions_in_commit_order is not None else [w.position_id for w in windows]
        )
    )
    if len(positions) != len(windows):
        raise NestedWindowError("positions_in_commit_order length must match windows length.")
    for pos in positions:
        if pos < 0 or pos > pre_n:
            raise NestedWindowError(f"Batch insertion position {pos!r} is outside [0, {pre_n}].")
    old_pre_union = tuple(sorted({int(idx) for window in windows for idx in window.old_pre_indices}))
    return build_composed_batch_window_payload(
        pre_parameter_count=int(pre_n),
        positions_in_commit_order=positions,
        old_pre_indices=old_pre_union,
    )
