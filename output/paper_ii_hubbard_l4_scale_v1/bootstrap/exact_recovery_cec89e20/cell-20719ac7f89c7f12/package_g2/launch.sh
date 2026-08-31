#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/exact_reference_action_package.tar.gz"
[[ -f "$PACKAGE" ]] || { echo "missing exact_reference_action_package.tar.gz" >&2; exit 2; }
[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "620467" ]] || { echo "package byte mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "68642ef7a8d699097f45f658c2cbc02a2cbd7eaa29f49ddaafb258b5c0b0270f" ]] || { echo "package SHA-256 mismatch" >&2; exit 2; }
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_exact_reference_action.sh
