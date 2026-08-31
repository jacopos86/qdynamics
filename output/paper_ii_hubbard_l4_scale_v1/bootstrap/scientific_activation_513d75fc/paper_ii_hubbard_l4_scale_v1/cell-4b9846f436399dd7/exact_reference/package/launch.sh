#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/exact_reference_action_package.tar.gz"
[[ -f "$PACKAGE" ]] || { echo "missing exact_reference_action_package.tar.gz" >&2; exit 2; }
[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "585583" ]] || { echo "package byte mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "557dc3ccc3134bafd744fa13ce30c7eee32c982f8be65b4bb4648c363737664f" ]] || { echo "package SHA-256 mismatch" >&2; exit 2; }
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_exact_reference_action.sh
