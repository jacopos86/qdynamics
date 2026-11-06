"""Array backends with honest device reporting for the chemistry bridge.

Two backends share one small interface so a statevector computation can be
written once and executed on CPU (numpy) or on an accelerator (torch on
``cuda``/``mps``/``cpu``):

* ``NumpyBackend`` holds complex vectors as ``np.ndarray``.
* ``TorchBackend`` holds a complex vector as a real tensor of shape ``(2, N)``
  (real part, imaginary part) so that every operation is real arithmetic.
  This avoids relying on complex-dtype kernels that are only partially
  implemented on some devices (notably ``mps``), and it lets ``float32`` and
  ``float64`` be selected explicitly.

Rules that keep the reporting honest:

* A backend is constructed only after a probe tensor has been placed on the
  requested device and the device it actually landed on has been read back.
  A mismatch raises ``BackendUnavailable``; there is no silent fallback.
* ``BackendReport.exercised_device`` is read from a live tensor, never copied
  from the request, and ``is_gpu`` is true only for ``cuda``/``mps``.
* ``float64`` on a device that rejects it (``mps``) is reported as unavailable
  with the device's own error text as the reason.

Precision: ``float32`` arithmetic carries roughly 1e-7 relative error per
operation; callers compare against a ``float64`` reference at tolerances
derived from ``BackendReport.real_dtype`` (see ``parity_tolerances``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


class BackendUnavailable(RuntimeError):
    """The requested backend/device/dtype cannot be exercised on this machine."""


_REAL_DTYPES = {"float32": np.float32, "float64": np.float64}
_COMPLEX_FOR_REAL = {"float32": np.complex64, "float64": np.complex128}
_GPU_DEVICES = {"cuda", "mps"}


@dataclass(frozen=True)
class BackendReport:
    """What was requested, what was exercised, and why if not."""

    spec: str
    library: str
    library_version: str | None
    requested_device: str
    exercised_device: str | None
    real_dtype: str
    complex_dtype: str
    is_gpu: bool
    available: bool
    reason: str | None = None

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def parity_tolerances(real_dtype: str) -> dict[str, float]:
    """Absolute tolerances used when comparing a backend against float64 numpy."""

    if str(real_dtype) == "float64":
        return {"vector_abs": 1.0e-11, "scalar_abs": 1.0e-11, "matrix_abs": 1.0e-10}
    if str(real_dtype) == "float32":
        return {"vector_abs": 5.0e-5, "scalar_abs": 5.0e-5, "matrix_abs": 2.0e-4}
    raise ValueError(f"unknown real dtype {real_dtype!r}")


def parse_backend_spec(spec: str) -> tuple[str, str, str]:
    """``"numpy"``, ``"numpy:float32"``, ``"torch:mps"``, ``"torch:cuda:float64"`` -> (library, device, real_dtype)."""

    parts = [part.strip().lower() for part in str(spec).split(":") if part.strip()]
    if not parts:
        raise ValueError("empty backend spec")
    library = parts[0]
    if library == "numpy":
        device = "cpu"
        rest = parts[1:]
    elif library == "torch":
        if len(parts) < 2:
            raise ValueError("torch backend spec requires a device: torch:<cpu|mps|cuda>[:<float32|float64>]")
        device = parts[1]
        rest = parts[2:]
    else:
        raise ValueError(f"unknown backend library {library!r}")
    if len(rest) > 1:
        raise ValueError(f"malformed backend spec {spec!r}")
    real_dtype = rest[0] if rest else "float64"
    if real_dtype not in _REAL_DTYPES:
        raise ValueError(f"unknown real dtype {real_dtype!r}; use float32 or float64")
    if library == "numpy" and device != "cpu":
        raise ValueError("numpy backend is CPU only")
    if library == "torch" and device not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unknown torch device {device!r}")
    return library, device, real_dtype


class NumpyBackend:
    """CPU backend over ``np.ndarray``; the float64 variant is the parity reference."""

    def __init__(self, real_dtype: str = "float64") -> None:
        if real_dtype not in _REAL_DTYPES:
            raise ValueError(f"unknown real dtype {real_dtype!r}")
        self.real_dtype = str(real_dtype)
        self._complex = _COMPLEX_FOR_REAL[self.real_dtype]
        self._report = BackendReport(
            spec=f"numpy:{self.real_dtype}",
            library="numpy",
            library_version=str(np.__version__),
            requested_device="cpu",
            exercised_device="cpu",
            real_dtype=self.real_dtype,
            complex_dtype=np.dtype(self._complex).name,
            is_gpu=False,
            available=True,
            reason=None,
        )

    def report(self) -> BackendReport:
        return self._report

    def from_complex(self, vector: np.ndarray) -> np.ndarray:
        return np.array(np.asarray(vector).reshape(-1), dtype=self._complex, copy=True)

    def to_complex(self, handle: np.ndarray) -> np.ndarray:
        return np.asarray(handle, dtype=np.complex128).reshape(-1)

    def index_array(self, indices: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(np.asarray(indices, dtype=np.int64))

    def gather(self, handle: np.ndarray, indices: np.ndarray) -> np.ndarray:
        return handle[indices]

    def zeros(self, size: int) -> np.ndarray:
        return np.zeros(int(size), dtype=self._complex)

    def axpy_weight(self, out: np.ndarray, weight: np.ndarray | complex, source: np.ndarray) -> None:
        """``out += weight * source`` where ``weight`` is a scalar or an elementwise array."""

        out += weight * source

    def vdot(self, a: np.ndarray, b: np.ndarray) -> complex:
        return complex(np.vdot(a, b))

    def norm(self, a: np.ndarray) -> float:
        return float(np.linalg.norm(a))

    def synchronize(self) -> None:
        return None


class TorchBackend:
    """torch backend over real ``(2, N)`` tensors with device verification."""

    def __init__(self, device: str, real_dtype: str = "float64") -> None:
        if real_dtype not in _REAL_DTYPES:
            raise ValueError(f"unknown real dtype {real_dtype!r}")
        device = str(device).strip().lower()
        if device not in {"cpu", "mps", "cuda"}:
            raise ValueError(f"unknown torch device {device!r}")
        try:
            import torch  # type: ignore
        except ModuleNotFoundError as exc:
            raise BackendUnavailable("torch is not importable in this environment") from exc
        self.torch = torch
        self.device_name = device
        self.real_dtype = str(real_dtype)
        self._dtype = torch.float32 if real_dtype == "float32" else torch.float64
        if device == "cuda" and not torch.cuda.is_available():
            raise BackendUnavailable("torch.cuda.is_available() is False")
        if device == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise BackendUnavailable("torch.backends.mps.is_available() is False")
        try:
            probe = torch.zeros(4, dtype=self._dtype, device=device)
        except Exception as exc:  # the device refuses this dtype or cannot be used
            raise BackendUnavailable(f"{device} rejected dtype {real_dtype}: {type(exc).__name__}: {exc}") from exc
        exercised = str(probe.device.type)
        if exercised != device:
            raise BackendUnavailable(f"requested device {device!r} but tensor landed on {exercised!r}; refusing silent fallback")
        self.device = probe.device
        # Functional probe: gather + arithmetic + reduction must round-trip.
        rng = np.random.default_rng(7)
        sample = rng.standard_normal(8) + 1j * rng.standard_normal(8)
        perm = np.array([3, 1, 4, 1, 5, 2, 6, 0], dtype=np.int64)
        handle = self.from_complex(sample)
        gathered = self.to_complex(self.gather(handle, self.index_array(perm)))
        expected = sample[perm]
        if not np.allclose(gathered, expected, atol=parity_tolerances(self.real_dtype)["vector_abs"] * 10):
            raise BackendUnavailable(f"functional gather probe failed on {device}")
        dot = self.vdot(handle, handle)
        if abs(dot - np.vdot(sample, sample)) > parity_tolerances(self.real_dtype)["scalar_abs"] * 100:
            raise BackendUnavailable(f"functional reduction probe failed on {device}")
        self._report = BackendReport(
            spec=f"torch:{device}:{self.real_dtype}",
            library="torch",
            library_version=str(getattr(torch, "__version__", None)),
            requested_device=device,
            exercised_device=exercised,
            real_dtype=self.real_dtype,
            complex_dtype=np.dtype(_COMPLEX_FOR_REAL[self.real_dtype]).name,
            is_gpu=exercised in _GPU_DEVICES,
            available=True,
            reason=None,
        )

    def report(self) -> BackendReport:
        return self._report

    def from_complex(self, vector: np.ndarray) -> Any:
        array = np.asarray(vector).reshape(-1)
        pair = np.stack([np.real(array), np.imag(array)]).astype(_REAL_DTYPES[self.real_dtype], copy=False)
        return self.torch.as_tensor(np.ascontiguousarray(pair), dtype=self._dtype, device=self.device)

    def to_complex(self, handle: Any) -> np.ndarray:
        pair = handle.detach().to("cpu").to(self.torch.float64).numpy()
        return np.asarray(pair[0] + 1j * pair[1], dtype=np.complex128).reshape(-1)

    def index_array(self, indices: np.ndarray) -> Any:
        return self.torch.as_tensor(np.ascontiguousarray(np.asarray(indices, dtype=np.int64)), dtype=self.torch.int64, device=self.device)

    def gather(self, handle: Any, indices: Any) -> Any:
        return handle[:, indices]

    def zeros(self, size: int) -> Any:
        return self.torch.zeros((2, int(size)), dtype=self._dtype, device=self.device)

    def axpy_weight(self, out: Any, weight: Any, source: Any) -> None:
        """``out += weight * source`` with complex ``weight`` as a ``(2,)`` or ``(2, N)`` tensor."""

        if isinstance(weight, complex) or np.isscalar(weight):
            w_re = float(np.real(weight))
            w_im = float(np.imag(weight))
            out[0] += w_re * source[0] - w_im * source[1]
            out[1] += w_re * source[1] + w_im * source[0]
            return
        out[0] += weight[0] * source[0] - weight[1] * source[1]
        out[1] += weight[0] * source[1] + weight[1] * source[0]

    def vdot(self, a: Any, b: Any) -> complex:
        re = self.torch.sum(a[0] * b[0] + a[1] * b[1])
        im = self.torch.sum(a[0] * b[1] - a[1] * b[0])
        return complex(float(re.item()), float(im.item()))

    def norm(self, a: Any) -> float:
        return float(self.torch.sqrt(self.torch.sum(a[0] * a[0] + a[1] * a[1])).item())

    def synchronize(self) -> None:
        if self.device_name == "cuda":
            self.torch.cuda.synchronize()
        elif self.device_name == "mps":
            sync = getattr(getattr(self.torch, "mps", None), "synchronize", None)
            if callable(sync):
                sync()


def resolve_backend(spec: str) -> NumpyBackend | TorchBackend:
    """Construct the backend named by ``spec`` or raise ``BackendUnavailable``."""

    library, device, real_dtype = parse_backend_spec(spec)
    if library == "numpy":
        return NumpyBackend(real_dtype=real_dtype)
    return TorchBackend(device=device, real_dtype=real_dtype)


def backend_availability(spec: str) -> BackendReport:
    """Report availability without raising; the reason names the exact obstacle."""

    try:
        library, device, real_dtype = parse_backend_spec(spec)
    except ValueError as exc:
        return BackendReport(spec=str(spec), library="?", library_version=None, requested_device="?", exercised_device=None, real_dtype="?", complex_dtype="?", is_gpu=False, available=False, reason=f"malformed spec: {exc}")
    try:
        backend = resolve_backend(spec)
    except BackendUnavailable as exc:
        version = None
        if library == "torch":
            try:
                import torch  # type: ignore

                version = str(getattr(torch, "__version__", None))
            except ModuleNotFoundError:
                version = None
        return BackendReport(
            spec=f"{library}:{device}:{real_dtype}" if library == "torch" else f"numpy:{real_dtype}",
            library=library,
            library_version=version,
            requested_device=device,
            exercised_device=None,
            real_dtype=real_dtype,
            complex_dtype=np.dtype(_COMPLEX_FOR_REAL[real_dtype]).name,
            is_gpu=False,
            available=False,
            reason=str(exc),
        )
    return backend.report()


DEFAULT_PROBE_SPECS = (
    "numpy:float64",
    "numpy:float32",
    "torch:cpu:float64",
    "torch:cpu:float32",
    "torch:mps:float32",
    "torch:mps:float64",
    "torch:cuda:float64",
    "torch:cuda:float32",
)


def probe_backends(specs: tuple[str, ...] = DEFAULT_PROBE_SPECS) -> list[BackendReport]:
    return [backend_availability(spec) for spec in specs]


__all__ = [
    "BackendReport",
    "BackendUnavailable",
    "DEFAULT_PROBE_SPECS",
    "NumpyBackend",
    "TorchBackend",
    "backend_availability",
    "parity_tolerances",
    "parse_backend_spec",
    "probe_backends",
    "resolve_backend",
]
