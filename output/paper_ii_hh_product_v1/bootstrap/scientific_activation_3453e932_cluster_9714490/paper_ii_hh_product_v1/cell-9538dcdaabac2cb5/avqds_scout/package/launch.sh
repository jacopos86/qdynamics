#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/avqds_prefix_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "92013" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "b9ef0ab748242a5ec287e43b64f151407b82e12301d563c514669d02e83d713c" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_avqds_prefix.sh
