#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/seed_action_package.tar.gz"
[[ -f "$PACKAGE" ]] || { echo "missing seed_action_package.tar.gz" >&2; exit 2; }
[[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "555495" ]] || { echo "package byte mismatch" >&2; exit 2; }
[[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "75fd38286f7a4eadc2f3be95a352e4f110b0ac9635e27220788409947e81e794" ]] || { echo "package SHA-256 mismatch" >&2; exit 2; }
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_seed_action.sh
