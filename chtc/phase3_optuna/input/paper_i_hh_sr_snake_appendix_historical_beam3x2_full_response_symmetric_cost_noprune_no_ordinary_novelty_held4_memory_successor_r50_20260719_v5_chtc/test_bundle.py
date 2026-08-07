#!/usr/bin/env python3
import unittest
import build_bundle
class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        build_bundle.module.verify(build_bundle.Path(__file__).resolve().parent)
if __name__ == "__main__": unittest.main()
