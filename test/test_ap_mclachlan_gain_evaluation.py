import math

from pipelines.time_dynamics.ap_mclachlan.gain_evaluation import factorized_capture


def test_signed_increments_and_deletion_baseline_are_counted_once():
    # Retained capture 10, singleton increments -2 and +3: combined 11.
    assert factorized_capture(10.0, (8.0, 13.0)) == 11.0
    assert factorized_capture(10.0, ()) == 10.0


def test_singleton_agreement_does_not_lose_precision_in_cancellation():
    assert factorized_capture(1e20, (1.0,)) == 1.0


def test_nonfinite_capture_is_not_silently_clipped_to_zero():
    assert math.isnan(factorized_capture(1.0, (math.nan, 2.0)))
