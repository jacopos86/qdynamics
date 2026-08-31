#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/exact_reference_action_package.tar.gz"
[[ -f "$PACKAGE" ]] || { echo "missing exact_reference_action_package.tar.gz" >&2; exit 2; }
[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "620455" ]] || { echo "package byte mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "d6bb53d8391cf1a107f6de08c88266ca05081db13a429b36f634e6c1de155388" ]] || { echo "package SHA-256 mismatch" >&2; exit 2; }
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_exact_reference_action.sh
