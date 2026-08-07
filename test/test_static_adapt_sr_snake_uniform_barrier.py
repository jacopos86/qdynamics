from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import math

import pytest

from pipelines.static_adapt.sr_snake_escape_controller import (
    reachable_population_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum import (
    CertificateState,
    EligibilityStateToken,
    EnergyInterval,
    PathActionKey,
    PathOrientation,
    UniformBarrierEvidence,
    canonical_action_receipt_digest,
)
from pipelines.static_adapt.sr_snake_modeled_minimum_runtime import (
    ConfigurationBinding,
    ProviderIdentity,
    ProviderRole,
    SourceBinding,
)
from pipelines.static_adapt.sr_snake_uniform_barrier import (
    METHOD,
    BoundCanonicalPath,
    FrozenRealPauliHamiltonian,
    IncumbentBarrierReference,
    SpectralL1BarrierContext,
    SpectralL1UniformBarrierCertificate,
    StoredRealPauliTerm,
    certify_spectral_l1_uniform_barrier,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _term(word: str, coefficient: object) -> StoredRealPauliTerm:
    return StoredRealPauliTerm(pauli_word=word, coefficient=coefficient)  # type: ignore[arg-type]


def _hamiltonian(
    *terms: tuple[str, object],
    hamiltonian_id: str = "ham",
) -> FrozenRealPauliHamiltonian:
    return FrozenRealPauliHamiltonian(
        hamiltonian_id=hamiltonian_id,
        qubit_count=2,
        terms=tuple(_term(word, coefficient) for word, coefficient in terms),
    )


def _token(*, epoch: str = "epoch-7") -> EligibilityStateToken:
    records = ("record-a", "record-b")
    return EligibilityStateToken(
        working_state_fingerprint="working-X",
        reachable_record_ids=records,
        reachable_population_digest=reachable_population_digest(records),
        comparison_epoch=epoch,
        support_provenance_digest="support-v1",
        trust_provenance_digest="trust-v1",
        trust_radius=0.25,
        stationarity_margin=-1.0e-8,
    )


def _action(*, path_index: int = 3) -> PathActionKey:
    return PathActionKey(
        record_id="record-a",
        record_order=1,
        record_count=2,
        orientation=PathOrientation.POSITIVE,
        radius_index=2,
        path_index=path_index,
    )


def _source(*, label: str = "source") -> SourceBinding:
    return SourceBinding(
        repository_id="holstein",
        revision=f"revision-{label}",
        source_digest=_sha(label),
    )


def _config(*, label: str = "config") -> ConfigurationBinding:
    return ConfigurationBinding(
        config_id=f"config-{label}",
        route_family="sr_snake",
        route_profile="modeled_minimum_stage_b",
        config_digest=_sha(label),
        state_replay_tolerance=1.0e-10,
        state_norm_error_tolerance=1.0e-12,
    )


def _provider(*, label: str = "provider") -> ProviderIdentity:
    return ProviderIdentity(
        role=ProviderRole.UNIFORM_INCUMBENT_BARRIER,
        provider_id="spectral-l1-fallback",
        version="1",
        implementation_digest=_sha(label),
    )


def _context(
    hamiltonian: FrozenRealPauliHamiltonian | None = None,
    *,
    token: EligibilityStateToken | None = None,
    action: PathActionKey | None = None,
    path_id: str = "canonical-path-3",
    descriptor_label: str = "descriptor-3",
    incumbent_error: float = 0.0,
    source: SourceBinding | None = None,
    config: ConfigurationBinding | None = None,
    provider: ProviderIdentity | None = None,
) -> SpectralL1BarrierContext:
    resolved_token = _token() if token is None else token
    resolved_action = _action() if action is None else action
    action_receipt = canonical_action_receipt_digest(
        resolved_action, resolved_token.digest
    )
    path = BoundCanonicalPath(
        path_id=path_id,
        descriptor_digest=_sha(descriptor_label),
        origin_state_fingerprint=resolved_token.working_state_fingerprint,
        eligibility_token_digest=resolved_token.digest,
        action_receipt_digest=action_receipt,
    )
    incumbent = IncumbentBarrierReference(
        snapshot_digest=_sha("incumbent-snapshot"),
        state_id="incumbent-I",
        energy=EnergyInterval(
            state_id="incumbent-I",
            energy_estimate=-1.25,
            energy_error_bound=incumbent_error,
            comparison_epoch=resolved_token.comparison_epoch,
            simultaneous=True,
        ),
    )
    return SpectralL1BarrierContext(
        eligibility_token=resolved_token,
        action_key=resolved_action,
        path=path,
        incumbent=incumbent,
        hamiltonian=(
            _hamiltonian(("xe", 0.25))
            if hamiltonian is None
            else hamiltonian
        ),
        source=_source() if source is None else source,
        config=_config() if config is None else config,
        provider=_provider() if provider is None else provider,
    )


def test_identity_shift_does_not_change_global_barrier() -> None:
    base = _hamiltonian(("ee", 7.0), ("xe", 0.25), ("yz", -0.5))
    shifted = _hamiltonian(("ee", -1.0e100), ("xe", 0.25), ("yz", -0.5))

    base_certificate = certify_spectral_l1_uniform_barrier(_context(base))
    shifted_certificate = certify_spectral_l1_uniform_barrier(_context(shifted))

    assert base_certificate.nonidentity_l1 == Fraction(3, 4)
    assert base_certificate.exact_barrier_upper_bound == Fraction(3, 2)
    assert shifted_certificate.exact_barrier_upper_bound == Fraction(3, 2)
    assert base.semantic_digest != shifted.semantic_digest


def test_duplicate_words_combine_before_absolute_values() -> None:
    hamiltonian = _hamiltonian(
        ("xe", 0.25),
        ("xe", -0.25),
        ("yz", 0.125),
        ("yz", 0.125),
    )

    certificate = certify_spectral_l1_uniform_barrier(_context(hamiltonian))

    assert hamiltonian.combined_terms == (("yz", Fraction(1, 4)),)
    assert certificate.nonidentity_l1 == Fraction(1, 4)
    assert certificate.exact_barrier_upper_bound == Fraction(1, 2)


@pytest.mark.parametrize(
    "hamiltonian",
    [
        _hamiltonian(),
        _hamiltonian(("ee", 123.5)),
        _hamiltonian(("xe", 0.5), ("xe", -0.5), ("ee", -8.0)),
    ],
)
def test_zero_or_identity_only_hamiltonian_has_exact_zero_barrier(
    hamiltonian: FrozenRealPauliHamiltonian,
) -> None:
    certificate = certify_spectral_l1_uniform_barrier(_context(hamiltonian))

    assert certificate.nonidentity_l1 == 0
    assert certificate.exact_barrier_upper_bound == 0
    assert certificate.barrier_upper_bound == 0.0


def test_barrier_conversion_rounds_outward_not_to_nearest_below() -> None:
    # 2 * (binary64(0.1) + binary64(0.4)) is one exact rational step above 1.
    hamiltonian = _hamiltonian(("xe", 0.1), ("yz", 0.4))
    certificate = certify_spectral_l1_uniform_barrier(_context(hamiltonian))

    assert certificate.exact_barrier_upper_bound > 1
    assert float(certificate.exact_barrier_upper_bound) == 1.0
    assert certificate.barrier_upper_bound == math.nextafter(1.0, math.inf)
    assert (
        Fraction.from_float(certificate.barrier_upper_bound)
        >= certificate.exact_barrier_upper_bound
    )


def test_passed_certificate_is_content_addressed_and_projects_existing_evidence() -> None:
    context = _context(
        _hamiltonian(("ee", 9.0), ("xe", 0.2), ("yz", -0.3)),
        incumbent_error=0.1,
    )
    certificate = certify_spectral_l1_uniform_barrier(context)

    restored = SpectralL1UniformBarrierCertificate.from_dict(certificate.to_dict())
    evidence = restored.to_uniform_barrier_evidence(current=context)

    assert restored.content_digest == certificate.content_digest
    assert restored.method == METHOD
    assert isinstance(evidence, UniformBarrierEvidence)
    assert evidence.status is CertificateState.PASSED
    assert evidence.enclosure_id == certificate.content_digest
    assert evidence.witness_id.endswith(certificate.content_digest)
    assert evidence.path_id == context.path.path_id
    assert evidence.action_receipt_digest == canonical_action_receipt_digest(
        context.action_key, context.eligibility_token.digest
    )
    assert evidence.incumbent_energy is context.incumbent.energy
    assert evidence.incumbent_referenced
    assert evidence.simultaneous


def test_certificate_explicitly_disclaims_local_or_fs_certificates() -> None:
    certificate = certify_spectral_l1_uniform_barrier(_context())
    payload = certificate.to_dict()

    assert payload["global_uniformity_certified"] is True
    assert payload["uniformity_scope"] == (
        "all_normalized_states_hence_any_bound_canonical_path"
    )
    assert payload["path_sampling_used"] is False
    assert payload["node_taylor_enclosure_certified"] is False
    assert payload["fs_exclusion_certified"] is False


def test_endpoint_and_interior_details_are_irrelevant_to_numeric_global_bound() -> None:
    hamiltonian = _hamiltonian(("xe", 0.3), ("yz", -0.4))
    first = certify_spectral_l1_uniform_barrier(
        _context(
            hamiltonian,
            path_id="path-with-one-endpoint",
            descriptor_label="piecewise-path-a",
        )
    )
    second = certify_spectral_l1_uniform_barrier(
        _context(
            hamiltonian,
            path_id="path-with-different-interior",
            descriptor_label="piecewise-path-b",
        )
    )

    assert first.exact_barrier_upper_bound == second.exact_barrier_upper_bound
    assert first.barrier_upper_bound == second.barrier_upper_bound
    assert first.content_digest != second.content_digest


def test_stale_config_or_path_context_cannot_emit_evidence() -> None:
    context = _context()
    certificate = certify_spectral_l1_uniform_barrier(context)
    stale_config = _context(config=_config(label="new-config"))
    stale_path = _context(path_id="replacement-path", descriptor_label="new-path")

    with pytest.raises(ValueError, match="stale for the current context"):
        certificate.to_uniform_barrier_evidence(current=stale_config)
    with pytest.raises(ValueError, match="stale for the current context"):
        certificate.to_uniform_barrier_evidence(current=stale_path)


def test_certificate_and_nested_digest_tampering_fail_closed() -> None:
    certificate = certify_spectral_l1_uniform_barrier(_context())

    top_level = certificate.to_dict()
    top_level["content_digest"] = _sha("forged")
    with pytest.raises(ValueError, match="content-addressed round-trip"):
        SpectralL1UniformBarrierCertificate.from_dict(top_level)

    nested = deepcopy(certificate.to_dict())
    nested["context"]["path"]["descriptor_digest"] = _sha("forged-path")
    with pytest.raises(ValueError, match="round-trip"):
        SpectralL1UniformBarrierCertificate.from_dict(nested)

    hamiltonian = certificate.context.hamiltonian.to_dict()
    hamiltonian["semantic_digest"] = _sha("forged-hamiltonian")
    with pytest.raises(ValueError, match="Hamiltonian binding"):
        FrozenRealPauliHamiltonian.from_dict(hamiltonian)


def test_nonreal_and_nonfinite_coefficients_are_rejected() -> None:
    for coefficient in (1.0 + 0.0j, complex(1.0, 2.0), math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite real"):
            _term("xe", coefficient)


def test_unrepresentably_large_finite_bound_fails_closed() -> None:
    hamiltonian = _hamiltonian(
        ("xe", float.fromhex("0x1.fffffffffffffp+1023")),
        ("yz", float.fromhex("0x1.fffffffffffffp+1023")),
    )

    with pytest.raises(ValueError, match="no finite binary64 upper enclosure"):
        certify_spectral_l1_uniform_barrier(_context(hamiltonian))


def test_stale_epoch_and_nonuniform_provider_role_fail_closed() -> None:
    token = _token(epoch="new-epoch")
    incumbent = IncumbentBarrierReference(
        snapshot_digest=_sha("incumbent"),
        state_id="incumbent-I",
        energy=EnergyInterval(
            state_id="incumbent-I",
            energy_estimate=-1.0,
            energy_error_bound=0.0,
            comparison_epoch="old-epoch",
            simultaneous=True,
        ),
    )
    action = _action()
    path = BoundCanonicalPath(
        path_id="path",
        descriptor_digest=_sha("path"),
        origin_state_fingerprint=token.working_state_fingerprint,
        eligibility_token_digest=token.digest,
        action_receipt_digest=canonical_action_receipt_digest(action, token.digest),
    )
    with pytest.raises(ValueError, match="comparison epoch is stale"):
        SpectralL1BarrierContext(
            eligibility_token=token,
            action_key=action,
            path=path,
            incumbent=incumbent,
            hamiltonian=_hamiltonian(("xe", 0.1)),
            source=_source(),
            config=_config(),
            provider=_provider(),
        )

    wrong_provider = replace(
        _provider(),
        role=ProviderRole.CANONICAL_PATH,
    )
    with pytest.raises(ValueError, match="uniform_incumbent_barrier"):
        _context(provider=wrong_provider)


def test_full_action_key_and_receipt_are_not_interchangeable() -> None:
    token = _token()
    first_action = _action(path_index=3)
    second_action = _action(path_index=4)
    stale_path = BoundCanonicalPath(
        path_id="path",
        descriptor_digest=_sha("path"),
        origin_state_fingerprint=token.working_state_fingerprint,
        eligibility_token_digest=token.digest,
        action_receipt_digest=canonical_action_receipt_digest(
            first_action, token.digest
        ),
    )
    incumbent = IncumbentBarrierReference(
        snapshot_digest=_sha("incumbent"),
        state_id="incumbent-I",
        energy=EnergyInterval(
            state_id="incumbent-I",
            energy_estimate=-1.0,
            energy_error_bound=0.0,
            comparison_epoch=token.comparison_epoch,
        ),
    )

    with pytest.raises(ValueError, match="full action key"):
        SpectralL1BarrierContext(
            eligibility_token=token,
            action_key=second_action,
            path=stale_path,
            incumbent=incumbent,
            hamiltonian=_hamiltonian(("xe", 0.1)),
            source=_source(),
            config=_config(),
            provider=_provider(),
        )


def test_failed_status_cannot_be_recast_as_passed_core_evidence() -> None:
    certificate = certify_spectral_l1_uniform_barrier(_context())
    serialized = certificate.to_dict()
    serialized["status"] = CertificateState.FAILED.value

    with pytest.raises(ValueError, match="only passed"):
        SpectralL1UniformBarrierCertificate.from_dict(serialized)

