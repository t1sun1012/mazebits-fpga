"""Prevent mixed-precision staging errors and verify frozen closed-loop evidence."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
sys.path.insert(0,str(ROOT/"python"))
import stage_navigation
from run_navigation import verify_vector_bundle
from mazebits.examples import EXAMPLES
from mazebits.maze import step

class NavigationStageTests(unittest.TestCase):
    def test_specialization_preserves_every_logic_character(self):
        source=(ROOT/"rtl/mazebits_cnn.v").read_text()
        for bits in (8,4):
            header=(ROOT/f"model/cnn-w{bits}a{bits}/quantization.vh").read_text()
            text=stage_navigation.specialize(source,header,bits)
            restored=text.replace(f"module mazebits_cnn_w{bits}a{bits} #(","module mazebits_cnn #(")
            restored=restored.replace(header.rstrip(),'`include "quantization.vh"')
            self.assertEqual(restored,source)
            self.assertIn(f"localparam integer W_BITS = {bits};",text)
            self.assertIn(f"localparam integer A_BITS = {bits};",text)

    def test_staged_weights_and_biases_are_exact_frozen_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            evidence=stage_navigation.stage(out)
            self.assertEqual(len(evidence["generated_sha256"]),6)
            for bits in (8,4):
                for stem in ("weights","biases"):
                    self.assertEqual((out/f"{stem}_w{bits}a{bits}.mem").read_bytes(),
                                     (ROOT/f"model/cnn-w{bits}a{bits}/{stem}.mem").read_bytes())
            for name,digest in evidence["generated_sha256"].items():
                self.assertEqual(stage_navigation.sha(out/name),digest)

    def test_precision_mismatch_is_rejected(self):
        source=(ROOT/"rtl/mazebits_cnn.v").read_text()
        header=(ROOT/"model/cnn-w8a8/quantization.vh").read_text()
        with self.assertRaisesRegex(ValueError,"does not match"):
            stage_navigation.specialize(source,header,4)
        with self.assertRaisesRegex(ValueError,"4 or 8"):
            stage_navigation.specialize(source,header,16)

    def test_missing_or_duplicate_binding_is_rejected(self):
        source=(ROOT/"rtl/mazebits_cnn.v").read_text()
        header=(ROOT/"model/cnn-w8a8/quantization.vh").read_text()
        for bad in (source.replace('`include "quantization.vh"',''),source+'\n`include "quantization.vh"'):
            with self.assertRaisesRegex(ValueError,"exactly one"):
                stage_navigation.specialize(bad,header,8)

    def test_corrupt_model_cannot_produce_partial_hardware_bundle(self):
        with tempfile.TemporaryDirectory() as temp:
            sandbox=Path(temp)
            (sandbox/"rtl").mkdir()
            shutil.copyfile(ROOT/"rtl/mazebits_cnn.v",sandbox/"rtl/mazebits_cnn.v")
            shutil.copytree(ROOT/"model/cnn-w8a8",sandbox/"model/cnn-w8a8")
            weights=sandbox/"model/cnn-w8a8/weights.mem"
            contents=weights.read_bytes()
            weights.write_bytes((b"0" if contents[0:1] != b"0" else b"1")+contents[1:])
            with patch.object(stage_navigation,"ROOT",sandbox):
                with self.assertRaisesRegex(ValueError,"integrity"):
                    stage_navigation.stage(sandbox/"out")
            self.assertEqual(list((sandbox/"out").iterdir()),[])

    def test_frozen_vectors_are_bound_to_current_reference_and_models(self):
        manifest,hashes=verify_vector_bundle(ROOT/"test_vectors/navigation")
        self.assertEqual(manifest["scalar_checked_decisions"],60)
        self.assertEqual(sum(e["steps"] for e in manifest["episodes"]),60)
        self.assertIn("python/mazebits/scalar_fixed.py",hashes)
        self.assertEqual(len(manifest["episodes"]),8)

    def test_frozen_paths_obey_raw_argmax_and_environment(self):
        manifest=json.loads((ROOT/"test_vectors/navigation/manifest.json").read_text())
        outcome_codes={"success":2,"blocked":3,"loop":4,"timeout":5}
        for episode in manifest["episodes"]:
            maze=EXAMPLES[episode["case"]]
            position=maze.start
            visited={position}
            trail=1 << (position[1]*8+position[0])
            for index,event in enumerate(episode["events"]):
                self.assertEqual(event["action"],max(range(4),key=lambda i:event["scores"][i]))
                moved=step(maze,position,event["action"])
                position=moved.position
                trail|=1 << (position[1]*8+position[0])
                self.assertEqual(event["position"],position[1]*8+position[0])
                self.assertEqual(int(event["trail_hex"],16),trail)
                self.assertEqual(event["steps"],index+1)
                self.assertEqual(event["total_cycles"],(index+1)*3217153)
                expected=(2 if moved.done else 3 if moved.blocked else 4 if position in visited
                          else 5 if index+1==64 else 1)
                self.assertEqual(event["status"],expected)
                visited.add(position)
            self.assertEqual(episode["final_position"],position[1]*8+position[0])
            self.assertEqual(episode["steps"],len(episode["events"]))
            if episode["events"]:
                self.assertEqual(episode["events"][-1]["status"],outcome_codes[episode["outcome"]])
            else:
                self.assertEqual(position,maze.goal)

if __name__=="__main__":
    unittest.main()
