"""Run with: python -m unittest discover -s mclachlan_figure -v"""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import generate_mclachlan as figure

NS = {"s": figure.SVG_NS}


class GeometryTests(unittest.TestCase):
    def test_geometry_validation(self):
        figure.validate_geometry()

    def test_parallel_positive_scaling(self):
        vertices = figure.GEOMETRY["black_vertices"]
        origin = figure.GEOMETRY["origin"]
        for a, b, tip, scale in zip(vertices, vertices[1:],
                                    figure.projected_blue_tips(),
                                    figure.GEOMETRY["blue_scales"]):
            black, blue = figure.sub(b, a), figure.sub(tip, origin)
            self.assertAlmostEqual(black[0] * blue[1] - black[1] * blue[0], 0, places=8)
            self.assertGreater(black[0] * blue[0] + black[1] * blue[1], 0)
            self.assertAlmostEqual(figure.norm(blue) / figure.norm(black), scale, places=12)

    def test_guides_reach_both_blue_tips(self):
        for upper, lower in zip(figure.GEOMETRY["upper_blue_tips"],
                                figure.projected_blue_tips()):
            segments = figure.guide_segments(upper, lower)
            self.assertEqual(segments[0][0], upper)
            self.assertLess(figure.norm(figure.sub(segments[-1][1], lower)), 1e-9)

    def test_negative_scale_is_rejected(self):
        with patch.dict(figure.GEOMETRY, blue_scales=(-0.78, 0.86, 0.9)):
            with self.assertRaises(ValueError):
                figure.validate_geometry()

    def test_open_black_chain_is_rejected(self):
        with patch.dict(figure.GEOMETRY, projected_drift_tip=(999.0, 705.0)):
            with self.assertRaises(ValueError):
                figure.validate_geometry()


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svg = figure.build_svg()
        cls.root = ET.fromstring(cls.svg)

    def test_default_starred_velocities_and_drift_labels(self):
        for index in figure.GEOMETRY["indices"]:
            node = self.root.find(f".//s:g[@id='label-velocity-{index}']", NS)
            self.assertIsNotNone(node)
            self.assertIn(r"^\star", node.get("aria-label"))
        for label_id, tex in (
                ("schrodinger-drift", r"-iH|\psi\rangle"),
                ("projected-schrodinger-drift", r"-i\Pi H|\psi\rangle=\bar d")):
            node = self.root.find(f".//s:g[@id='label-{label_id}']", NS)
            self.assertEqual(node.get("aria-label"), tex)

    def test_svg_is_self_contained_vector(self):
        for tag in ("image", "text", "script", "foreignObject"):
            self.assertEqual(self.root.findall(f".//s:{tag}", NS), [])
        ids = [node.get("id") for node in self.root.iter() if "id" in node.attrib]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertNotIn("data:image", self.svg)
        self.assertNotIn("/mnt/data", self.svg)
        self.assertNotIn("file://", self.svg)
        self.assertNotIn("gray-state-guide", self.svg)

    def test_svg_endpoint_coordinates_match_exactly(self):
        for index in figure.GEOMETRY["indices"]:
            guide = self.root.find(f".//s:g[@id='guide-{index}']", NS)
            upper = self.root.find(f".//s:g[@id='upper-blue-arrow-{index}']", NS)
            lower = self.root.find(f".//s:g[@id='lower-blue-arrow-{index}']", NS)
            self.assertEqual(guide.get("data-start"), upper.get("data-tip"))
            self.assertEqual(guide.get("data-end"), lower.get("data-tip"))
            dashes = guide.findall("s:path", NS)
            self.assertEqual(dashes[0].get("d").split()[1], guide.get("data-start"))
            self.assertEqual(dashes[-1].get("d").split()[-1], guide.get("data-end"))
            for arrow in (upper, lower):
                apex = arrow.findall("s:path", NS)[1].get("d").split()[1]
                self.assertEqual(apex, arrow.get("data-tip"))

    def test_svg_regeneration_is_deterministic(self):
        self.assertEqual(self.svg, figure.build_svg())

    def test_unstarred_option_keeps_arrow_geometry(self):
        unstarred = ET.fromstring(figure.build_svg(starred=False))
        self.assertNotIn(r"^\star", ET.tostring(unstarred, encoding="unicode"))
        for group in self.root.findall(".//s:g", NS):
            if "arrow" in group.get("id", ""):
                other = unstarred.find(f".//s:g[@id='{group.get('id')}']", NS)
                self.assertEqual(ET.tostring(group), ET.tostring(other))

    def test_write_svg_and_reject_bad_extension(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "figure.svg"
            actual, png = figure.build(path)
            self.assertEqual(actual.read_text(encoding="utf-8"), self.svg)
            self.assertIsNone(png)
            with self.assertRaises(ValueError):
                figure.build(Path(tmp) / "figure.png")


if __name__ == "__main__":
    unittest.main()
