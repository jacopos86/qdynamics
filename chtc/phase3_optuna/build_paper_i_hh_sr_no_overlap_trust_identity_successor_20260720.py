#!/usr/bin/env python3
"""Build the immutable v4 no-overlap route-identity successor.

The v3 archive registered the already-defined route in the request and Powell
allowlists, but omitted it from the legacy-controller ablation allowlist used by
the final resolved-route identity gate.  This patch changes only that runtime
registration surface; the route digest and all scientific settings remain
unchanged.
"""

from __future__ import annotations

import sys

import build_paper_i_hh_sr_no_overlap_trust_registration_successor_20260720 as base


PARENT_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v3_chtc"
OUTPUT_ID = "paper_i_hh_sr_snake_no_overlap_trust_all_six_r50_20260720_v4_chtc"
PARENT_BATCH = "paper-i-hh-sr-no-overlap-trust-six-r50-20260720-v3"
OUTPUT_BATCH = "paper-i-hh-sr-no-overlap-trust-six-r50-20260720-v4"
PROFILE = (
    "SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_"
    "NO_OVERLAP_TRUST_V1"
)


def patch_adapt(text: str) -> str:
    old = """            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    new = """            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_V1,
            SR_ROUTE_PROFILE_NO_PRUNE_SYMMETRIC_COST_PROJECTED_PHASE3_NO_OVERLAP_TRUST_V1,
            SR_ROUTE_PROFILE_GUARDED_SINGLETON_POOL_V1,
"""
    start = text.index("registered_controller_ablation = bool(")
    end = text.index("    explicit_ablation = bool(", start)
    block = text[start:end]
    if PROFILE in block:
        raise ValueError("predecessor already contains the controller identity registration")
    if block.count(old) != 1:
        raise ValueError("controller identity registration seam is missing or ambiguous")
    return text[:start] + block.replace(old, new, 1) + text[end:]


def main() -> int:
    base.PARENT_ID = PARENT_ID
    base.OUTPUT_ID = OUTPUT_ID
    base.PARENT_BATCH = PARENT_BATCH
    base.OUTPUT_BATCH = OUTPUT_BATCH
    base.patch_adapt = patch_adapt
    if sys.argv[1:] == ["--finalize-remote-preflight"]:
        base.finalize_remote_preflight()
        return 0
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
