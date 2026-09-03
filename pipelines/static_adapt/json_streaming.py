"""Chunked JSON encoding that is byte-identical to :mod:`json` output.

Why this exists (Paper-I B3 tail diagnosis, 2026-09-03): the accepted
checkpoint publisher digests and writes multi-gigabyte projections with
``json.dump``.  ``json.dump`` always takes the pure-Python ``_make_iterencode``
path (it calls ``iterencode`` with ``_one_shot=False``), which is several
times slower than the C encoder that ``json.dumps`` uses, while ``json.dumps``
materializes the whole document as one string.  The generators here emit
bounded-size chunks produced by the C encoder, so the concatenation equals
the ``json`` module's output byte for byte while resident memory stays
bounded by the chunk target.

Identity contract (verified by ``test/test_json_streaming.py``):

* ``"".join(iter_canonical_json(v))`` ==
  ``json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=..., allow_nan=...)``
* ``"".join(iter_pretty_json(v))`` ==
  ``json.dumps(v, sort_keys=True, indent=2, ensure_ascii=..., allow_nan=...)``

Both forms sort keys, as every checkpoint writer in this package does.  Any
value the C encoder rejects (non-string dictionary keys mixed with strings,
NaN with ``allow_nan=False``, unsupported types) raises the same exception
here because container slices are encoded by ``json.dumps`` itself.

Mechanism: a container is split into slices of consecutive children; each
slice is encoded by the C encoder in one call and its surrounding brackets
are stripped, so slices concatenate with the container's item separator
into exactly the one-shot encoding.  Slice length adapts to the measured
output size (target a few megabytes), so homogeneous long lists (ledger
occurrences, entries, Phase-0 populations) are encoded at C speed with a
handful of Python-level steps, while a child that is too large for a slice
of one is recursed into and chunked itself.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Iterator, Mapping

# Adaptive slice control.
MAX_SLICE_ITEMS = 4096
TARGET_SLICE_BYTES = 2 * 1024 * 1024
# A single child whose encoding would exceed this is recursed into rather
# than encoded whole, bounding the largest transient string.
RECURSE_CHILD_BYTES = 8 * TARGET_SLICE_BYTES
# Bounded node-count probe used before recursing into a lone child.
LEAF_NODE_LIMIT = 512

_INDENT = "  "


def _bounded_node_count(value: Any, limit: int) -> int:
    """Count JSON nodes in ``value``, stopping once ``limit`` is exceeded."""

    count = 0
    stack: list[Any] = [value]
    while stack:
        item = stack.pop()
        count += 1
        if count > limit:
            return count
        if isinstance(item, Mapping):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return count


def _is_small(value: Any) -> bool:
    if not isinstance(value, (Mapping, list, tuple)):
        return True
    return _bounded_node_count(value, LEAF_NODE_LIMIT) <= LEAF_NODE_LIMIT


def _all_str_keys(mapping: Mapping[Any, Any]) -> bool:
    return all(isinstance(key, str) for key in mapping)


def _sorted_items(mapping: Mapping[str, Any]) -> list[tuple[str, Any]]:
    # ``json`` sorts ``dict.items()`` tuples; keys are unique so this is a
    # pure key sort.
    return sorted(mapping.items(), key=lambda item: item[0])


def _next_slice_length(current: int, encoded_length: int) -> int:
    if encoded_length > 2 * TARGET_SLICE_BYTES:
        return max(1, current // 4)
    if encoded_length < TARGET_SLICE_BYTES // 4:
        return min(MAX_SLICE_ITEMS, current * 2)
    return current


def _walk(
    value: Any,
    *,
    level: int,
    dumps: Callable[[Any], str],
    pretty: bool,
) -> Iterator[str]:
    """Yield chunks for ``value`` nested at depth ``level``."""

    if not isinstance(value, (Mapping, list, tuple)) or not value:
        # Scalars and empty containers: the C encoder is exact and cheap.
        yield dumps(value)
        return

    is_mapping = isinstance(value, Mapping)
    if is_mapping and not _all_str_keys(value):
        # Non-string keys: let ``json`` apply (or reject) its own coercion
        # rules on the whole mapping so behaviour stays identical.
        yield _reindent(dumps(value), level) if pretty else dumps(value)
        return

    if is_mapping:
        children = _sorted_items(value)

        def encode_slice(chunk: list[tuple[str, Any]]) -> str:
            return dumps(dict(chunk))

        def encode_child(child: tuple[str, Any]) -> Iterator[str]:
            key, item = child
            yield json.dumps(key, ensure_ascii=ensure_ascii_flag)
            yield ": " if pretty else ":"
            yield from _walk(item, level=level + 1, dumps=dumps, pretty=pretty)

        def child_is_small(child: tuple[str, Any]) -> bool:
            return _is_small(child[1])

        open_token, close_token = "{", "}"
    else:
        children = list(value)

        def encode_slice(chunk: list[Any]) -> str:  # type: ignore[misc]
            return dumps(chunk)

        def encode_child(child: Any) -> Iterator[str]:  # type: ignore[misc]
            yield from _walk(child, level=level + 1, dumps=dumps, pretty=pretty)

        def child_is_small(child: Any) -> bool:  # type: ignore[misc]
            return _is_small(child)

        open_token, close_token = "[", "]"

    ensure_ascii_flag = _ensure_ascii_of(dumps)
    child_prefix = _INDENT * (level + 1)
    separator = ",\n" if pretty else ","

    if pretty:
        yield open_token + "\n"
    else:
        yield open_token

    first = True
    index = 0
    total = len(children)
    slice_length = 1
    while index < total:
        # One bounded probe per slice: a large child at the head of the
        # slice is recursed into so its own slices stay bounded, instead of
        # being encoded whole together with its siblings.
        if not child_is_small(children[index]):
            if not first:
                yield separator
            first = False
            if pretty:
                yield child_prefix
            yield from encode_child(children[index])
            index += 1
            slice_length = 1
            continue
        chunk = children[index : index + slice_length]
        encoded = encode_slice(chunk)
        if pretty:
            # ``[\n  a,\n  b\n]`` -> ``  a,\n  b`` at level 1; shift to
            # ``level + 1`` with one C-level replace (JSON strings never
            # contain raw newlines, so every ``\n`` starts a line).
            body = encoded[2:-2]
            if level > 0:
                prefix = _INDENT * level
                body = prefix + body.replace("\n", "\n" + prefix)
        else:
            body = encoded[1:-1]
        if not first:
            yield separator
        first = False
        yield body
        index += len(chunk)
        if len(chunk) == 1 and len(encoded) > RECURSE_CHILD_BYTES:
            slice_length = 1
        else:
            slice_length = _next_slice_length(len(chunk), len(encoded))

    if pretty:
        yield "\n" + _INDENT * level + close_token
    else:
        yield close_token


def _reindent(block: str, level: int) -> str:
    """Prefix every line after the first with ``level`` indent units."""

    if level <= 0 or "\n" not in block:
        return block
    return block.replace("\n", "\n" + _INDENT * level)


def _ensure_ascii_of(dumps: Callable[[Any], str]) -> bool:
    return bool(getattr(dumps, "ensure_ascii", True))


def _make_dumps(
    *,
    ensure_ascii: bool,
    allow_nan: bool,
    pretty: bool,
) -> Callable[[Any], str]:
    if pretty:

        def dumps(item: Any) -> str:
            return json.dumps(
                item,
                sort_keys=True,
                indent=2,
                ensure_ascii=ensure_ascii,
                allow_nan=allow_nan,
            )

    else:

        def dumps(item: Any) -> str:
            return json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=ensure_ascii,
                allow_nan=allow_nan,
            )

    dumps.ensure_ascii = bool(ensure_ascii)  # type: ignore[attr-defined]
    return dumps


def iter_canonical_json(
    value: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = False,
) -> Iterator[str]:
    """Yield chunks equal in concatenation to the compact sorted ``json.dumps``."""

    dumps = _make_dumps(ensure_ascii=ensure_ascii, allow_nan=allow_nan, pretty=False)
    yield from _walk(value, level=0, dumps=dumps, pretty=False)


def iter_pretty_json(
    value: Any,
    *,
    ensure_ascii: bool = True,
    allow_nan: bool = False,
) -> Iterator[str]:
    """Yield chunks equal in concatenation to ``json.dumps(..., indent=2)``."""

    dumps = _make_dumps(ensure_ascii=ensure_ascii, allow_nan=allow_nan, pretty=True)
    yield from _walk(value, level=0, dumps=dumps, pretty=True)


__all__ = [
    "LEAF_NODE_LIMIT",
    "MAX_SLICE_ITEMS",
    "RECURSE_CHILD_BYTES",
    "TARGET_SLICE_BYTES",
    "iter_canonical_json",
    "iter_pretty_json",
]
