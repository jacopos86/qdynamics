#!/usr/bin/env python3
from __future__ import annotations

import ast
import tarfile
import unittest
from typing import Any

import numpy as np

import build_bundle


def _archive_source(relative: str) -> str:
    with tarfile.open(build_bundle.BUNDLE_DIR / "source_locked.tar.gz", "r:gz") as handle:
        return handle.extractfile(relative).read().decode("utf-8")


def _load_normalizer():
    source = _archive_source("pipelines/static_adapt/selector_query_closure.py")
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_serialized_matrix_payload"
    )
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"Any": Any, "np": np}
    exec(compile(module, "<source-locked-normalizer>", "exec"), namespace)
    return namespace["normalize_serialized_matrix_payload"]


class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        self.assertTrue(build_bundle.verify())

    def test_zero_extent_matrices_restore_receipt_declared_shape(self):
        normalize = _load_normalizer()
        for shape in ((0, 0), (0, 3), (4, 0)):
            with self.subTest(shape=shape):
                restored = normalize([], expected_shape=shape, field_name="G_AA_raw")
                self.assertEqual(restored.shape, shape)
                self.assertEqual(restored.size, 0)
                self.assertFalse(restored.flags.writeable)

    def test_malformed_or_nonfinite_matrix_still_fails_closed(self):
        normalize = _load_normalizer()
        for payload, shape in (([], (1, 1)), ([[1.0]], (0, 0)), ([[np.nan]], (1, 1))):
            with self.subTest(payload=payload, shape=shape):
                with self.assertRaises(ValueError):
                    normalize(payload, expected_shape=shape, field_name="H_AA_raw")

    def test_batch3_receipt_parser_calls_typed_normalizer(self):
        tree = ast.parse(_archive_source("pipelines/static_adapt/adapt_pipeline.py"))
        outer = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "_phase3_batch_appendix_trust_update_inputs"
        )
        finite_matrix = next(
            node
            for node in outer.body
            if isinstance(node, ast.FunctionDef) and node.name == "_finite_matrix"
        )
        calls = {
            node.func.id
            for node in ast.walk(finite_matrix)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertIn("normalize_serialized_matrix_payload", calls)


if __name__ == "__main__":
    unittest.main()
