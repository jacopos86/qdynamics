#!/usr/bin/env bash
set -euo pipefail
umask 077
PACKAGE="$PWD/avqds_prefix_action.tar.gz"
[[ -f "$PACKAGE" ]] && [[ "$(wc -c < "$PACKAGE" | tr -d " ")" == "95804" ]] && [[ "$(sha256sum "$PACKAGE" | awk '{print $1}')" == "5d02abff0220148c3e300cb2051cb7a11df8db3cd59150889233d8ed51b1b0c1" ]] || exit 2
mkdir action_package
tar -xzf "$PACKAGE" -C action_package
exec bash action_package/run_avqds_prefix.sh
