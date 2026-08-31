#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/avqds_prefix_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "93857" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "820e92d9128f53ef7c53f758c95865e6c6ea3a1bc4ad0bab2b4ff2bb6a982e68" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_avqds_prefix.sh
