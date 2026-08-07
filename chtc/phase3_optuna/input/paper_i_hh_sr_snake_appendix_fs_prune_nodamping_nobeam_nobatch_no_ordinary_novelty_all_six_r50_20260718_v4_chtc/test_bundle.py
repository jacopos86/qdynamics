#!/usr/bin/env python3
import unittest
import build_bundle
class BundleTest(unittest.TestCase):
    def test_immutable_successor(self):
        self.assertTrue(build_bundle.verify())
if __name__ == "__main__": unittest.main()
