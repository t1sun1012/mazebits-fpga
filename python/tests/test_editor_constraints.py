"""Audit the editor/VGA pins and the narrow asynchronous-path exceptions."""
from pathlib import Path
import re
import unittest
from test_board_constraints import pin_records, top_ports

ROOT=Path(__file__).resolve().parents[2]
ACTIVE=ROOT/"constraints/mazebits_editor.xdc"
TOP=ROOT/"rtl/mazebits_editor_top.v"
REFERENCE=ROOT/"constraints/reference/Nexys-4-DDR-Master.xdc"

class EditorConstraintTests(unittest.TestCase):
    def test_all_39_active_pins_match_user_reference(self):
        original={name:(pin,standard) for name,pin,standard in pin_records(REFERENCE.read_text(),reference=True)}
        active=pin_records(ACTIVE.read_text())
        self.assertEqual(len(active),39)
        for name,pin,standard in active:
            with self.subTest(port=name):
                self.assertEqual((pin,standard),original[name])

    def test_complete_port_coverage_and_no_duplicate_pin_assignments(self):
        active=pin_records(ACTIVE.read_text())
        names=[name for name,_,_ in active]
        pins=[pin for _,pin,_ in active]
        self.assertEqual(len(names),len(set(names)))
        self.assertEqual(len(pins),len(set(pins)))
        self.assertEqual(set(names),top_ports(TOP.read_text()))

    def test_only_100_mhz_clock_and_no_multicycle_or_clock_group_exceptions(self):
        text=ACTIVE.read_text()
        clocks=[line for line in text.splitlines() if line.strip().startswith("create_clock ")]
        self.assertEqual(len(clocks),1)
        self.assertIn("-period 10.000",clocks[0])
        self.assertIn("get_ports CLK100MHZ",clocks[0])
        self.assertNotRegex(text,r"(?m)^\s*set_(?:multicycle_path|clock_groups)\b")

    def test_false_paths_only_first_stage_inputs_or_external_outputs(self):
        text=ACTIVE.read_text()
        cuts=[line.strip() for line in text.splitlines() if line.strip().startswith("set_false_path ")]
        self.assertEqual(len(cuts),4)
        for variable in ("buttons_first","switches_first","reset_first"):
            matching=[line for line in cuts if "$"+variable in line]
            self.assertEqual(len(matching),1)
            self.assertIn("-from [get_ports ",matching[0])
            self.assertIn("-filter {REF_PIN_NAME == D}",matching[0])
        self.assertIn("sync_meta_reg",text)
        self.assertIn("mode_pipe0_reg",text)
        self.assertNotIn("sync_buttons_reg",text)
        self.assertNotIn("mode_pipe1_reg",text)
        self.assertEqual(cuts[-1],"set_false_path -to [get_ports {LED[*] VGA_R[*] VGA_G[*] VGA_B[*] VGA_HS VGA_VS}]")

    def test_full_vga_bus_and_configuration_voltage(self):
        active={name:standard for name,_,standard in pin_records(ACTIVE.read_text())}
        for color in "RGB":
            for bit in range(4):
                self.assertEqual(active[f"VGA_{color}[{bit}]"],"LVCMOS33")
        self.assertEqual(active["VGA_HS"],"LVCMOS33")
        self.assertEqual(active["VGA_VS"],"LVCMOS33")
        text=ACTIVE.read_text()
        self.assertIn("set_property CFGBVS VCCO [current_design]",text)
        self.assertIn("set_property CONFIG_VOLTAGE 3.3 [current_design]",text)

if __name__=="__main__":
    unittest.main()
