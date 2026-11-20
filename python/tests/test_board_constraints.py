"""Audit authored pin constraints against the actual user-supplied board file."""
import hashlib
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "constraints/reference/Nexys-4-DDR-Master.xdc"
ACTIVE = ROOT / "constraints/mazebits_smoke.xdc"
TOP = ROOT / "rtl/mazebits_smoke_top.v"
REFERENCE_SHA256 = "36b0dd2332991699f478f430f0f01add968ff375c65c6d76a301e864ffb70572"


def pin_records(text, reference=False):
    records = []
    pattern = re.compile(
        r"set_property\s+-dict\s+\{\s*PACKAGE_PIN\s+(\w+)\s+"
        r"IOSTANDARD\s+(\w+)\s*\}\s+\[get_ports\s+"
        r"(?:\{\s*([^}]+?)\s*\}|(\w+))\s*\]"
    )
    for line in text.splitlines():
        if not reference and line.lstrip().startswith("#"):
            continue
        match = pattern.search(line)
        if match:
            pin, standard, braced, bare = match.groups()
            records.append(((braced or bare).strip(), pin, standard))
    return records


def top_ports(text):
    # This smoke wrapper uses simple ANSI scalar or constant-width wire ports.
    ports = set()
    for msb, lsb, name in re.findall(
        r"\b(?:input|output)\s+wire\s+(?:\[(\d+):(\d+)\]\s+)?(\w+)", text
    ):
        if msb:
            ports.update(f"{name}[{bit}]" for bit in range(int(lsb), int(msb) + 1))
        else:
            ports.add(name)
    return ports


class BoardConstraintTests(unittest.TestCase):
    def test_original_reference_bytes_are_preserved(self):
        self.assertEqual(hashlib.sha256(REFERENCE.read_bytes()).hexdigest(), REFERENCE_SHA256)

    def test_every_active_pin_and_voltage_matches_user_reference(self):
        original = {name: (pin, standard) for name, pin, standard in
                    pin_records(REFERENCE.read_text(), reference=True)}
        active = pin_records(ACTIVE.read_text())
        self.assertTrue(active)
        for name, pin, standard in active:
            with self.subTest(port=name):
                self.assertIn(name, original)
                self.assertEqual((pin, standard), original[name])

    def test_ports_are_complete_without_duplicate_assignments(self):
        active = pin_records(ACTIVE.read_text())
        names = [name for name, _, _ in active]
        pins = [pin for _, pin, _ in active]
        self.assertEqual(len(names), len(set(names)), "Port assigned more than once")
        self.assertEqual(len(pins), len(set(pins)), "Package pin assigned more than once")
        self.assertEqual(set(names), top_ports(TOP.read_text()))

    def test_100_mhz_clock_is_constrained(self):
        text = ACTIVE.read_text()
        self.assertRegex(text, r"create_clock\s+-name\s+sys_clk_pin\s+-period\s+10\.0+\s+\[get_ports\s+CLK100MHZ\]")
        clocks = [line for line in text.splitlines() if line.strip().startswith("create_clock ")]
        self.assertEqual(len(clocks), 1)

    def test_reference_keeps_switches_8_and_9_at_1v8(self):
        original = {name: standard for name, _, standard in
                    pin_records(REFERENCE.read_text(), reference=True)}
        self.assertEqual(original["SW[8]"], "LVCMOS18")
        self.assertEqual(original["SW[9]"], "LVCMOS18")


if __name__ == "__main__":
    unittest.main()
