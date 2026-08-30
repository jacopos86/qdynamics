#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/apm_exchange_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "631254" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "92dd1a2761a27f865c04c538472644ee82cf816974b580b62f1b7f7eb87116b5" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_apm_exchange.sh
