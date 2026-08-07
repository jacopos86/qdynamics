#!/usr/bin/env python3
"""Stage HTCondor's renamed executable into the authenticated package tree.

HTCondor may place ``transfer_executable`` at a scheduler-owned scratch name
instead of preserving the submit-side relative path.  The Study-1 control
receipt nevertheless authenticates the wrapper at its repository-relative
package path.  This helper makes that layout explicit without transferring the
wrapper twice or overwriting any independently transferred file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence


WRAPPER_NAME = "execute_source_locked_job.sh"


class ExecutableStagingError(RuntimeError):
    """Raised when the transferred executable cannot be staged safely."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stage_transferred_executable(
    *,
    wrapper_source: str | Path,
    package_dir: str | Path,
) -> dict[str, object]:
    """Place ``wrapper_source`` at the sole authenticated wrapper path.

    An already-present destination is accepted only when it is the same file
    as the source (the local/direct-execution layout).  A distinct existing
    path is a collision and fails closed even when its bytes happen to match.
    """

    source = Path(wrapper_source)
    package = Path(package_dir)
    if source.is_symlink() or not source.is_file():
        raise ExecutableStagingError(
            f"Transferred executable is unavailable or symlinked: {source}"
        )
    if package.is_symlink() or not package.is_dir():
        raise ExecutableStagingError(
            f"Authenticated package directory is unavailable: {package}"
        )

    source = source.resolve(strict=True)
    destination = package.resolve(strict=True) / WRAPPER_NAME
    action = "already_at_authenticated_path"
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or not destination.is_file()
            or not os.path.samefile(source, destination)
        ):
            raise ExecutableStagingError(
                "Refusing a duplicate transferred-executable collision at "
                f"{destination}"
            )
    else:
        action = "staged_renamed_transfer_executable"
        temporary = package / (
            f".{WRAPPER_NAME}.stage.{os.getpid()}"
        )
        temporary_created = False
        try:
            with source.open("rb") as input_stream, temporary.open(
                "xb"
            ) as output_stream:
                temporary_created = True
                for block in iter(
                    lambda: input_stream.read(1024 * 1024), b""
                ):
                    output_stream.write(block)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            temporary.chmod(source.stat().st_mode & 0o777)
            # A hard-link publication is atomic and refuses an existing
            # destination; it cannot overwrite a raced-in transferred file.
            os.link(temporary, destination)
        finally:
            if (
                temporary_created
                and temporary.exists()
                and not temporary.is_symlink()
            ):
                temporary.unlink()

    source_sha256 = _sha256_file(source)
    destination_sha256 = _sha256_file(destination)
    if (
        source_sha256 != destination_sha256
        or source.stat().st_size != destination.stat().st_size
    ):
        raise ExecutableStagingError(
            "Staged transferred executable does not match its source bytes."
        )
    return {
        "schema": "paper_i_ra_adapt_transferred_executable_layout_v1",
        "action": action,
        "authenticated_relative_path": WRAPPER_NAME,
        "sha256": destination_sha256,
        "size_bytes": destination.stat().st_size,
        "layout_verified": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper-source", required=True)
    parser.add_argument("--package-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = stage_transferred_executable(
        wrapper_source=args.wrapper_source,
        package_dir=args.package_dir,
    )
    print(
        json.dumps(
            receipt,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
