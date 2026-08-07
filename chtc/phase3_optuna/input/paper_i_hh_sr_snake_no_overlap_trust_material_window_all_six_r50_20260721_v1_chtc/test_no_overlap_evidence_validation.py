import copy

import pytest

from evidence_validation import validate_no_overlap_trust_evidence


POLICY = "source_metric_inverse_sqrt_no_overlap_v1"


def accounting():
    return {
        "status": "not_required_by_policy",
        "performed": False,
        "charged": False,
        "added_query_count": 0,
        "overlap_protocol": "off",
        "trust_update_policy": POLICY,
    }


def normal_receipt():
    return {
        "policy": POLICY,
        "endpoint_overlap_measurement_required": False,
        "endpoint_overlap_measurement_performed": False,
        "endpoint_overlap_query_charge": 0,
        "realized_fs_displacement_exact": None,
        "endpoint_overlap_query_accounting": accounting(),
        "predicted_source_metric_displacement": 0.25,
        "realized_source_metric_displacement": 0.5,
        "displacement_ratio": 2.0,
        "displacement_ratio_metric": (
            "supported_source_gram_parameter_displacement_v1"
        ),
        "trust_radius_update_exponent": -0.5,
        "radius_before": 0.25,
        "radius_after": 0.1767766952966369,
    }


def geometry_receipt():
    return {
        "policy": POLICY,
        "geometry_expansion_active": True,
        "update_reason": (
            "geometry_expansion_no_coordinate_prediction_no_overlap_hold"
        ),
        "endpoint_overlap_measurement_required": False,
        "endpoint_overlap_measurement_performed": False,
        "endpoint_overlap_query_charge": 0,
        "realized_fs_displacement_exact": None,
        "endpoint_overlap_query_accounting": accounting(),
        "radius_before": 0.25,
        "radius_after": 0.25,
    }


def result():
    return {
        "settings": {
            "historical_singleton_trust_region_update_policy": POLICY,
        },
        "adapt_vqe": {
            "history": [
                {"route_a_trust_region_update": normal_receipt()},
                {"route_a_trust_region_update": geometry_receipt()},
            ]
        },
    }


def test_accepts_complete_no_overlap_receipts():
    report = validate_no_overlap_trust_evidence(result=result(), target_round=2)
    assert report["status"] == "pass"
    assert report["source_metric_receipt_count"] == 1
    assert report["geometry_expansion_no_overlap_hold_count"] == 1
    assert report["endpoint_overlap_query_charge"] == 0


@pytest.mark.parametrize("field,value", [
    ("endpoint_overlap_measurement_performed", True),
    ("endpoint_overlap_query_charge", 1),
    ("realized_fs_displacement_exact", 0.1),
])
def test_rejects_overlap_measurement_or_charge(field, value):
    payload = result()
    payload["adapt_vqe"]["history"][0][
        "route_a_trust_region_update"
    ][field] = value
    with pytest.raises(ValueError):
        validate_no_overlap_trust_evidence(result=payload, target_round=2)


def test_rejects_missing_receipt():
    payload = result()
    del payload["adapt_vqe"]["history"][0]["route_a_trust_region_update"]
    with pytest.raises(ValueError):
        validate_no_overlap_trust_evidence(result=payload, target_round=2)
