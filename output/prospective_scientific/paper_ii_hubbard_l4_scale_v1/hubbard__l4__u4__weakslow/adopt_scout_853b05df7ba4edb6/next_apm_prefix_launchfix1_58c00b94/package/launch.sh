#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/apm_exchange_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "626921" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "63409104ba15c29c672b032dbd88cfbe91277fec799c428b7e14627271bcf00f" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_apm_exchange.sh
