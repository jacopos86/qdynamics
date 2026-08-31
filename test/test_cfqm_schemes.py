import unittest
import sys
from pathlib import Path

# Path setup pattern used by existing tests.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.quantum.time_propagation.cfqm_schemes import get_cfqm_scheme


class TestCFQMSchemes(unittest.TestCase):
    def _assert_close(self, lhs: float, rhs: float, tol: float = 1e-12) -> None:
        self.assertLessEqual(abs(float(lhs) - float(rhs)), float(tol))

    def test_cf4_aliases(self) -> None:
        for alias in ("cfqm4", "CF4:2", "cf4"):
            with self.subTest(alias=alias):
                scheme = get_cfqm_scheme(alias)
                self.assertEqual(scheme["name"], "CF4:2")
                self.assertEqual(scheme["order"], 4)

    def test_cf4_invariants(self) -> None:
        scheme = get_cfqm_scheme("cfqm4")
        c = scheme["c"]
        a = scheme["a"]
        s_static = scheme["s_static"]

        self.assertEqual(len(c), 2)
        self.assertEqual(len(a), 2)
        self.assertEqual(len(a[0]), 2)

        for node in c:
            self.assertGreaterEqual(float(node), -1e-12)
            self.assertLessEqual(float(node), 1.0 + 1e-12)

        total = sum(sum(float(x) for x in row) for row in a)
        self._assert_close(total, 1.0)

        expected_rows = [0.5, 0.5]
        for got, want in zip(s_static, expected_rows):
            self._assert_close(got, want)

    def test_cf6_aliases(self) -> None:
        for alias in ("cfqm6", "CF6:5Opt", "cf6:5opt", "cf6"):
            with self.subTest(alias=alias):
                scheme = get_cfqm_scheme(alias)
                self.assertEqual(scheme["name"], "CF6:5Opt")
                self.assertEqual(scheme["order"], 6)

    def test_cf6_invariants(self) -> None:
        scheme = get_cfqm_scheme("cfqm6")
        c = scheme["c"]
        a = scheme["a"]
        s_static = scheme["s_static"]

        self.assertEqual(len(c), 4)
        self.assertEqual(len(a), 5)
        for row in a:
            self.assertEqual(len(row), 4)

        total = sum(sum(float(x) for x in row) for row in a)
        self._assert_close(total, 1.0)

        expected_rows = [
            0.1714,
            0.37496374319946236513,
            -0.09272748639892473026,
            0.37496374319946236513,
            0.1714,
        ]
        for got, want in zip(s_static, expected_rows):
            self._assert_close(got, want)

        expected_cols = [
            0.1739274225687269286865,
            0.3260725774312730713135,
            0.3260725774312730713135,
            0.1739274225687269286865,
        ]
        for j in range(len(expected_cols)):
            got_col = sum(float(row[j]) for row in a)
            self._assert_close(got_col, expected_cols[j])

    def test_unknown_scheme_raises(self) -> None:
        with self.assertRaises(ValueError):
            _ = get_cfqm_scheme("not-a-scheme")


if __name__ == "__main__":
    unittest.main()
