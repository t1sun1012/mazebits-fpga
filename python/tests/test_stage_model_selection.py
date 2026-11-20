"""Keep original and improved FPGA model bundles explicit and internally paired."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import stage_navigation as staging

class ModelSelectionTests(unittest.TestCase):
    def test_both_selections_stage_exact_distinct_model_bytes(self):
        hashes={}
        with tempfile.TemporaryDirectory() as tmp:
            for name in staging.MODEL_SETS:
                out=Path(tmp)/name
                record=staging.stage(out,name)
                self.assertEqual(record["model_set"],name)
                for bits in (8,4):
                    selected=staging.model_directory(bits,name)
                    for kind in ("weights","biases"):
                        self.assertEqual((out/f"{kind}_w{bits}a{bits}.mem").read_bytes(),(selected/f"{kind}.mem").read_bytes())
                    self.assertIn((selected/"manifest.json").relative_to(ROOT).as_posix(),record["source_sha256"])
                hashes[name]=record["source_checkpoint"]["source_weights_sha256"]
        self.assertNotEqual(hashes["original"],hashes["structured"])
    def test_unknown_selection_creates_no_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"output"
            with self.assertRaisesRegex(ValueError,"model_set"):
                staging.stage(out,"typo")
            self.assertFalse(out.exists())
    def make_sandbox(self, root, improved_four=True):
        (root/"rtl").mkdir()
        shutil.copyfile(ROOT/"rtl/mazebits_cnn.v",root/"rtl/mazebits_cnn.v")
        for bits in (8,4):
            name="structured" if bits==8 or improved_four else "original"
            shutil.copytree(staging.model_directory(bits,name),root/f"model/cnn-structured-w{bits}a{bits}")
    def test_mixed_checkpoints_cannot_stage_partial_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_sandbox(root,improved_four=False)
            with patch.object(staging,"ROOT",root):
                with self.assertRaisesRegex(ValueError,"paired models"):
                    staging.stage(root/"out","structured")
            self.assertEqual(list((root/"out").iterdir()),[])
    def test_incompatible_manifest_architecture_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.make_sandbox(root)
            p=root/"model/cnn-structured-w8a8/manifest.json"
            d=json.loads(p.read_text());d["layers"][1]["weight_shape"][0]=32
            p.write_text(json.dumps(d))
            with patch.object(staging,"ROOT",root):
                with self.assertRaisesRegex(ValueError,"architecture"):
                    staging.stage(root/"out","structured")
            self.assertEqual(list((root/"out").iterdir()),[])

if __name__=="__main__":unittest.main()
