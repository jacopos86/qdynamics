#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/exact_reference_action_package.tar.gz"
[[ -f "$PACKAGE" ]] || { echo "missing exact_reference_action_package.tar.gz" >&2; exit 2; }
[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "585578" ]] || { echo "package byte mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "75b22b2490f068afe41a8c13a6bbbf33b0e60deb976ebf69c4347a40f144a118" ]] || { echo "package SHA-256 mismatch" >&2; exit 2; }
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_exact_reference_action.sh
