#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/apm_exchange_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "626873" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "0143b668890ab21dc933426778b846e6b5e4609219eb5f340fc3be99c03cb74f" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_apm_exchange.sh
