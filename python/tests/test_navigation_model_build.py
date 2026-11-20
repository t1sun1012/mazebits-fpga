"""Guard model-specific build identity and preserve the original board image."""
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_navigation_vivado as build
import stage_navigation


class NavigationModelBuildTests(unittest.TestCase):
    def test_model_build_outputs_are_disjoint_and_explicit(self):
        old_dir, old_bit = build.build_paths("original")
        new_dir, new_bit = build.build_paths("structured")
        self.assertNotEqual(old_dir, new_dir)
        self.assertEqual(old_bit, "mazebits_navigation.bit")
        self.assertEqual(new_bit, "mazebits_navigation_structured.bit")
        command = build.build_command("vivado", "bitstream", "structured")
        self.assertEqual(command[command.index("-tclargs") + 1:], ["bitstream", "structured"])
        with self.assertRaises(ValueError): build.build_paths("../original")
        with self.assertRaises(ValueError): build.build_command("vivado", "invalid", "structured")

    def test_staging_identity_and_artifact_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            evidence = stage_navigation.stage(out, model_set="structured")
            build.validate_staging(evidence, "structured", out)
            with self.assertRaisesRegex(ValueError, "model set"):
                build.validate_staging(evidence, "original", out)
            wrong = copy.deepcopy(evidence)
            wrong["source_sha256"]["model/cnn-structured-w8a8/manifest.json"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "provenance"):
                build.validate_staging(wrong, "structured", out)
            with (out / "weights_w4a4.mem").open("ab") as handle:
                handle.write(b"00\n")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                build.validate_staging(evidence, "structured", out)

    def test_failure_cannot_leave_a_plausible_bitstream_or_success_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            bit = "mazebits_navigation_structured.bit"
            (out / bit).write_bytes(b"stale")
            (out / "implementation.json").write_text('{"status":"pass"}')
            untouched = out / "mazebits_navigation.bit"
            untouched.write_bytes(b"original")
            def failed_run(*args, **kwargs):
                self.assertFalse((out / bit).exists())
                self.assertFalse((out / "implementation.json").exists())
                (out / bit).write_bytes(b"incomplete")
                return subprocess.CompletedProcess(args[0], 1, "ERROR: synthetic build failure")
            with patch.object(build, "build_paths", return_value=(out, bit)), patch.object(build.subprocess, "run", side_effect=failed_run), patch.object(sys, "argv", ["runner", "--model-set", "structured", "--stage", "bitstream", "--vivado", "vivado"]):
                with self.assertRaisesRegex(RuntimeError, "build failed"):
                    build.main()
            self.assertFalse((out / bit).exists())
            self.assertFalse((out / "implementation.json").exists())
            self.assertTrue((out / "build_failure.json").is_file())
            self.assertEqual(untouched.read_bytes(), b"original")

    def test_synthesis_evidence_excludes_stale_route_and_gui_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            current = ("utilization_synth.rpt", "timing_synth.rpt")
            stale = ("utilization_route.rpt", "utilization_hierarchy.rpt",
                     "timing_route.rpt", "drc_route.rpt", "clock_interaction.rpt",
                     "gui_compile_order.rpt")
            for name in current:
                (out / name).write_text("current synthesis report")
            for name in stale:
                (out / name).write_text("older report")
            hashes = build.report_hashes(out, "synth")
            self.assertEqual(set(hashes), set(current))
            self.assertEqual(hashes["timing_synth.rpt"], build.sha(out / "timing_synth.rpt"))
            self.assertTrue((out / "gui_compile_order.rpt").exists())
            routed = build.report_hashes(out, "route")
            self.assertEqual(len(routed), 7)
            self.assertNotIn("gui_compile_order.rpt", routed)
            (out / "timing_synth.rpt").unlink()
            with self.assertRaises(FileNotFoundError):
                build.report_hashes(out, "synth")

    def test_hash_inventory_binds_requested_models_and_manifests(self):
        old = build.source_hashes("original")
        new = build.source_hashes("structured")
        for bits in (8,4):
            for name in ("manifest.json", "weights.mem", "biases.mem", "quantization.vh"):
                self.assertIn(f"model/cnn-w{bits}a{bits}/{name}", old)
                self.assertIn(f"model/cnn-structured-w{bits}a{bits}/{name}", new)
                self.assertNotIn(f"model/cnn-w{bits}a{bits}/{name}", new)


if __name__ == "__main__":
    unittest.main()
