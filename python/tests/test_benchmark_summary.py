"""Guard against combining benchmark evidence from different model/build versions."""
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from summarize_benchmarks import hierarchy_resources, summarize


class BenchmarkEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.summary = summarize(ROOT)

    def copy_evidence(self, destination):
        for name in self.summary["source_sha256"]:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)

    def test_routed_mixed_bram_counts_and_unrelated_header_lines(self):
        text = "| Date : example\n| cnn4 | module | 37 | 35 | 2 | 0 | 50 | 3 | 1 | 0 |\n"
        resources = hierarchy_resources(text, "cnn4")
        self.assertEqual(resources["bram_36k_equivalents"], 3.5)
        self.assertEqual(resources["allocated_bram_bits_including_parity_capacity"], 129024)
        self.assertEqual(resources["luts"], 37)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            hierarchy_resources(text + text, "cnn4")

    def test_changed_model_manifest_cannot_reuse_old_measurements(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_evidence(root)
            path = root / "model/cnn-w4a4/manifest.json"
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "identity mismatch.*manifest.json"):
                summarize(root)

    def test_changed_hierarchy_report_cannot_reuse_published_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.copy_evidence(root)
            path = root / "results/navigation/utilization_hierarchy.rpt"
            path.write_bytes(path.read_bytes().replace(b"516", b"100"))
            with self.assertRaisesRegex(ValueError, "identity mismatch.*hierarchy report"):
                summarize(root)

    def test_report_distinguishes_payload_working_memory_and_resources(self):
        eight, four = (self.summary["models"][name] for name in ("w8a8", "w4a4"))
        self.assertLess(four["packed_parameter_bytes"], eight["packed_parameter_bytes"])
        self.assertEqual(eight["bias_bytes"], four["bias_bytes"])
        self.assertEqual(eight["logical_activation_bank_bytes"], 2 * four["logical_activation_bank_bytes"])
        self.assertEqual(eight["cycles_per_decision"], four["cycles_per_decision"])
        self.assertFalse(self.summary["dataset"]["test_partition_evaluated"])


if __name__ == "__main__":
    unittest.main()
