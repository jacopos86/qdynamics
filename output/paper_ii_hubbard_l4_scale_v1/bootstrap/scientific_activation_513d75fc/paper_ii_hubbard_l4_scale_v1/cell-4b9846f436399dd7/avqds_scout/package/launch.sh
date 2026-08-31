#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/avqds_prefix_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "95820" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "e0fe5a51c9acc6313ec0c1c6f5709ab7f41f0566be88ac5d32aee7f053ea19d0" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_avqds_prefix.sh
