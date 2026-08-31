#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/apm_exchange_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "631196" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "95e896cc117c4376a193afff092655fdb47f564f98e91753d747bdb2c1e00006" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_apm_exchange.sh
