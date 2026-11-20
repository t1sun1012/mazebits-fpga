"""Physical pin coverage for the integrated navigation top."""
from pathlib import Path
import unittest
from test_board_constraints import pin_records, top_ports

ROOT = Path(__file__).resolve().parents[2]

class NavigationConstraintTests(unittest.TestCase):
    def test_all_navigation_ports_use_board_tested_editor_pins(self):
        reference = pin_records((ROOT / "constraints/mazebits_editor.xdc").read_text())
        actual = pin_records((ROOT / "constraints/mazebits_navigation.xdc").read_text())
        self.assertEqual(actual, reference)
        self.assertEqual(len(actual), 39)
        self.assertEqual({name for name, _, _ in actual},
                         top_ports((ROOT / "rtl/mazebits_navigation_top.v").read_text()))
        self.assertEqual(len({pin for _, pin, _ in actual}), len(actual))

    def test_navigation_does_not_relax_board_tested_timing_constraints(self):
        def commands(name):
            return [line for line in (ROOT / "constraints" / name).read_text().splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        self.assertEqual(commands("mazebits_navigation.xdc"), commands("mazebits_editor.xdc"))

if __name__ == "__main__":
    unittest.main()
