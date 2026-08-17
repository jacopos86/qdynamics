#!/usr/bin/env python3
"""Source-locked streaming JSON parser derived from ijson 3.5.1.

Only the synchronous ``parse`` surface of ijson's pure-Python backend is
vendored here.  Keeping this small backend in the control plane avoids an
ambient-site-package dependency on CHTC while preserving bounded-memory
checkpoint inspection.
"""

from __future__ import annotations

import codecs
import decimal
from functools import wraps
from json.decoder import scanstring
import re
from typing import Any, BinaryIO, Generator, Iterable


VENDORED_IJSON_VERSION = "3.5.1"
BACKEND = "python"


class JSONError(Exception):
    """Base streaming-JSON error."""


class IncompleteJSONError(JSONError):
    """Raised when the input ends before a JSON value is complete."""


class UnexpectedSymbol(JSONError):
    def __init__(self, symbol: str, position: int) -> None:
        super().__init__(f"Unexpected symbol {symbol!r} at {position}")


def coroutine(function: Any) -> Any:
    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        generator = function(*args, **kwargs)
        next(generator)
        return generator

    return wrapper


class _SendableList(list[Any]):
    send = list.append


def _chain(sink: Any, *pipeline: tuple[Any, tuple[Any, ...], dict[str, Any]]) -> Any:
    target = sink
    for function, args, kwargs in pipeline:
        target = function(target, *args, **kwargs)
    return target


def _coros2gen(
    source: Iterable[bytes],
    *pipeline: tuple[Any, tuple[Any, ...], dict[str, Any]],
) -> Generator[Any, None, None]:
    events = _SendableList()
    target = _chain(events, *pipeline)
    try:
        for value in source:
            try:
                target.send(value)
            except Exception as exc:
                yield from events
                if isinstance(exc, StopIteration):
                    return
                raise
            yield from events
            del events[:]
    except GeneratorExit:
        try:
            target.close()
        except Exception:
            pass


@coroutine
def _parse_basecoro(target: Any) -> Generator[Any, Any, None]:
    path: list[str | None] = []
    while True:
        event, value = yield
        if event == "map_key":
            prefix = ".".join(str(item) for item in path[:-1])
            path[-1] = value
        elif event == "start_map":
            prefix = ".".join(str(item) for item in path)
            path.append(None)
        elif event == "end_map":
            path.pop()
            prefix = ".".join(str(item) for item in path)
        elif event == "start_array":
            prefix = ".".join(str(item) for item in path)
            path.append("item")
        elif event == "end_array":
            path.pop()
            prefix = ".".join(str(item) for item in path)
        else:
            prefix = ".".join(str(item) for item in path)
        target.send((prefix, event, value))


_LEXEME_RE = re.compile(r"[a-z0-9eE\.\+\-]+|\S")
_UNARY_LEXEMES = set("[]{},")
_EOF = (-1, None)


@coroutine
def _utf8_encoder(target: Any) -> Generator[Any, bytes, None]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    while True:
        try:
            final = False
            data = yield
        except GeneratorExit:
            final = True
            data = b""
        try:
            decoded = decoder.decode(data, final)
        except UnicodeDecodeError as exc:
            try:
                target.close()
            except Exception:
                pass
            raise IncompleteJSONError(str(exc)) from exc
        if decoded:
            target.send(decoded)
        elif not data:
            target.close()
            break


@coroutine
def _lexer(target: Any) -> Generator[Any, str, None]:
    try:
        data = yield
    except GeneratorExit:
        data = ""
    buffer = data
    position = 0
    discarded = 0
    while True:
        match = _LEXEME_RE.search(buffer, position)
        if match:
            lexeme = match.group()
            if lexeme == '"':
                position = match.start()
                start = position + 1
                while True:
                    try:
                        end = buffer.index('"', start)
                        escape = end - 1
                        while escape >= 0 and buffer[escape] == "\\":
                            escape -= 1
                        if (end - escape) % 2 == 0:
                            start = end + 1
                        else:
                            break
                    except ValueError:
                        try:
                            data = yield
                        except GeneratorExit:
                            data = ""
                        if not data:
                            raise IncompleteJSONError("Incomplete string lexeme")
                        buffer += data
                target.send((discarded + position, buffer[position : end + 1]))
                position = end + 1
            else:
                while lexeme not in _UNARY_LEXEMES and match.end() == len(buffer):
                    try:
                        data = yield
                    except GeneratorExit:
                        data = ""
                    if not data:
                        break
                    buffer += data
                    match = _LEXEME_RE.search(buffer, position)
                    assert match is not None
                    lexeme = match.group()
                target.send((discarded + match.start(), lexeme))
                position = match.end()
        else:
            if data:
                try:
                    data = yield
                except GeneratorExit:
                    data = ""
            if not data:
                try:
                    target.send(_EOF)
                except StopIteration:
                    pass
                break
            discarded += len(buffer)
            buffer = data
            position = 0


_PARSE_VALUE = 0
_PARSE_ARRAY_ELEMENT_END = 1
_PARSE_OBJECT_KEY = 2
_PARSE_OBJECT_END = 3
_INFINITY = float("inf")


def _number(symbol: str) -> int | decimal.Decimal:
    if not any(token in symbol for token in (".", "e", "E")):
        return int(symbol)
    return decimal.Decimal(symbol)


@coroutine
def _parse_value(target: Any) -> Generator[Any, tuple[int, str | None], None]:
    stack = [_PARSE_VALUE]
    previous_position: int | None = None
    previous_symbol: str | None = None
    while True:
        if previous_position is None:
            position, symbol = yield
            if (position, symbol) == _EOF:
                if stack:
                    raise IncompleteJSONError("Incomplete JSON content")
                break
        else:
            position, symbol = previous_position, previous_symbol
            previous_position = previous_symbol = None
        if not stack:
            raise JSONError("Additional data found")
        assert symbol is not None
        state = stack[-1]
        if state == _PARSE_VALUE:
            if symbol == "null":
                target.send(("null", None))
                stack.pop()
            elif symbol == "true":
                target.send(("boolean", True))
                stack.pop()
            elif symbol == "false":
                target.send(("boolean", False))
                stack.pop()
            elif symbol[0] == '"':
                target.send(("string", scanstring(symbol, 1)[0]))
                stack.pop()
            elif symbol == "[":
                target.send(("start_array", None))
                position, symbol = yield
                if (position, symbol) == _EOF:
                    raise IncompleteJSONError("Incomplete JSON content")
                if symbol == "]":
                    target.send(("end_array", None))
                    stack.pop()
                else:
                    previous_position, previous_symbol = position, symbol
                    stack.extend((_PARSE_ARRAY_ELEMENT_END, _PARSE_VALUE))
            elif symbol == "{":
                target.send(("start_map", None))
                position, symbol = yield
                if (position, symbol) == _EOF:
                    raise IncompleteJSONError("Incomplete JSON content")
                if symbol == "}":
                    target.send(("end_map", None))
                    stack.pop()
                else:
                    previous_position, previous_symbol = position, symbol
                    stack.append(_PARSE_OBJECT_KEY)
            else:
                if (
                    len(symbol) > 1
                    and symbol[0] == "0"
                    and symbol[1] not in ("e", "E", ".")
                ) or (
                    len(symbol) > 2
                    and symbol[:2] == "-0"
                    and symbol[2] not in ("e", "E", ".")
                ):
                    raise JSONError(f"Invalid JSON number: {symbol}")
                if symbol[0] == "." or symbol[-1] == ".":
                    raise JSONError(f"Invalid JSON number: {symbol}")
                try:
                    value = _number(symbol)
                    if value == _INFINITY:
                        raise JSONError(f"float overflow: {symbol}")
                except Exception as exc:
                    if any(word.startswith(symbol) for word in ("true", "false", "null")):
                        raise IncompleteJSONError("Incomplete JSON content") from exc
                    raise UnexpectedSymbol(symbol, position) from exc
                target.send(("number", value))
                stack.pop()
        elif state == _PARSE_OBJECT_KEY:
            if symbol[0] != '"':
                raise UnexpectedSymbol(symbol, position)
            target.send(("map_key", scanstring(symbol, 1)[0]))
            position, symbol = yield
            if (position, symbol) == _EOF:
                raise IncompleteJSONError("Incomplete JSON content")
            if symbol != ":":
                raise UnexpectedSymbol(str(symbol), position)
            stack[-1] = _PARSE_OBJECT_END
            stack.append(_PARSE_VALUE)
        elif state == _PARSE_OBJECT_END:
            if symbol == ",":
                stack[-1] = _PARSE_OBJECT_KEY
            elif symbol != "}":
                raise UnexpectedSymbol(symbol, position)
            else:
                target.send(("end_map", None))
                stack.pop()
                stack.pop()
        elif state == _PARSE_ARRAY_ELEMENT_END:
            if symbol == ",":
                stack.append(_PARSE_VALUE)
            elif symbol != "]":
                raise UnexpectedSymbol(symbol, position)
            else:
                target.send(("end_array", None))
                stack.pop()
                stack.pop()


def _basic_parse_basecoro(target: Any) -> Any:
    return _utf8_encoder(_lexer(_parse_value(target)))


def _file_source(stream: BinaryIO, buffer_size: int) -> Generator[bytes, None, None]:
    while True:
        block = stream.read(buffer_size)
        if isinstance(block, str):
            block = block.encode("utf-8")
        yield block
        if not block:
            break


def parse(
    stream: BinaryIO, *, buf_size: int = 64 * 1024
) -> Generator[tuple[str, str, Any], None, None]:
    """Yield ``(prefix, event, value)`` using the vendored Python backend."""

    if not hasattr(stream, "read"):
        raise ValueError(f"Unknown source type: {type(stream)!r}")
    return _coros2gen(
        _file_source(stream, buf_size),
        (_parse_basecoro, (), {}),
        (_basic_parse_basecoro, (), {}),
    )
