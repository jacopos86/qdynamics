from __future__ import annotations

import signal
from itertools import product

from pipelines.exact_bench.paper_iii_qse_measurement_cost import _qwc_cover


def test_dense_qwc_cover_remains_exact_and_practical() -> None:
    words = ["".join(letters) for letters in product("exyz", repeat=7)]

    def fail_if_quadratic(_signum, _frame) -> None:
        raise AssertionError("dense QWC cover exceeded the two-second regression budget")

    previous_handler = signal.signal(signal.SIGALRM, fail_if_quadratic)
    signal.setitimer(signal.ITIMER_REAL, 2.0)
    try:
        group_count = _qwc_cover(words, nq=7)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, previous_handler)

    assert group_count == 2187
