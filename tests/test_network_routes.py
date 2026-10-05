import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from network_routes import build_boundary_candidates, contains_path


class BoundaryRoutes(unittest.TestCase):
    def test_route_keeps_upstream_and_downstream_junctions(self):
        with tempfile.TemporaryDirectory() as tmp:
            network = Path(tmp) / "network.xml"
            network.write_text('''<net>
              <edge id="entry"><lane length="200"/></edge><edge id="up"><lane length="100"/></edge>
              <edge id="curve"><lane length="30"/></edge><edge id="down"><lane length="100"/></edge>
              <edge id="exit"><lane length="200"/></edge>
              <connection from="entry" to="up" dir="s" tl="upstream"/>
              <connection from="up" to="curve" dir="r"/>
              <connection from="curve" to="down" dir="s"/>
              <connection from="down" to="exit" dir="s" tl="downstream"/>
            </net>''')
            spec = {"id": "right", "path": ["up", "curve", "down"]}
            candidates, metadata = build_boundary_candidates(network, [spec])
            self.assertEqual(candidates[0]["edges"], ["entry", "up", "curve", "down", "exit"])
            self.assertTrue(metadata["movement_coverage"][0]["covered"])
            self.assertFalse(metadata["junctions_truncated"])

    def test_unresolved_and_partial_bypass_do_not_count(self):
        self.assertFalse(contains_path(["entry", "curve"], ["entry", "curve", "exit"]))
        self.assertTrue(contains_path(["a", "entry", "curve", "exit", "b"], ["entry", "curve", "exit"]))
        with tempfile.TemporaryDirectory() as tmp:
            network = Path(tmp) / "network.xml"
            network.write_text('<net><edge id="a"><lane length="10"/></edge></net>')
            candidates, metadata = build_boundary_candidates(network, [{"id": "unknown", "path": None}])
            self.assertEqual(candidates, [])
            self.assertEqual(metadata["movement_coverage"][0]["reason"], "unresolved_movement")


if __name__ == "__main__":
    unittest.main()
