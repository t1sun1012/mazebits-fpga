"""Guard model/vector selection and immutable simulator inputs."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import run_rtl


class RTLSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.sandbox = Path(self.temporary.name)
        self.model = self.sandbox / "model/cnn-structured-w8a8"
        self.vectors = self.sandbox / "test_vectors/cnn-structured-w8a8"
        shutil.copytree(ROOT / "model/cnn-structured-w8a8", self.model)
        shutil.copytree(ROOT / "test_vectors/cnn-structured-w8a8", self.vectors)
        self.destination = self.sandbox / "staged"

    def tearDown(self):
        self.temporary.cleanup()

    def write_manifest(self, directory, value):
        (directory / "manifest.json").write_text(json.dumps(value), encoding="utf-8")

    def test_each_selection_stages_exact_source_bytes_and_header(self):
        for model_set in run_rtl.MODEL_SETS:
            for bits in (8,4):
                with self.subTest(model_set=model_set,bits=bits):
                    out = self.sandbox / f"{model_set}-{bits}"
                    # Mapped Windows paths must not be resolved into UNC paths.
                    with patch.object(Path, "resolve", side_effect=AssertionError("must retain mapped drive")):
                        evidence = run_rtl.stage(bits, out, model_set)
                    source, _ = run_rtl.model_paths(bits, model_set)
                    for name in ("weights.mem", "biases.mem", "quantization.vh"):
                        self.assertEqual((out/name).read_bytes(), (source/name).read_bytes())
                    self.assertEqual(evidence["model_set"], model_set)
                    self.assertIn("quantization.vh", evidence["staged_sha256"])
                    self.assertEqual(len(run_rtl.words(out/"expected_acc.mem")), 67584)
                    self.assertEqual(len(run_rtl.words(out/"expected_act.mem")), 65536)
                    run_rtl.check_staged_inputs(evidence, out)
        self.assertIn("cnn-w8a8", str(run_rtl.model_paths(8)[0]))
        for bits, model_set in ((16,"original"),(8,"unknown")):
            with self.assertRaises(ValueError): run_rtl.model_paths(bits,model_set)

    def test_vector_identity_mismatch_is_rejected_before_any_output(self):
        manifest = json.loads((self.vectors/"manifest.json").read_text())
        manifest["source_model"]["weights_sha256"] = "0"*64
        self.write_manifest(self.vectors,manifest)
        with patch.object(run_rtl,"ROOT",self.sandbox):
            with self.assertRaisesRegex(ValueError,"different model"):
                run_rtl.stage(8,self.destination,"structured")
        self.assertFalse(self.destination.exists())

    def test_corrupt_memory_is_rejected_before_any_output(self):
        path = self.model/"weights.mem"
        contents=path.read_bytes()
        path.write_bytes((b"f" if contents[:1] != b"f" else b"0")+contents[1:])
        with patch.object(run_rtl,"ROOT",self.sandbox):
            with self.assertRaisesRegex(ValueError,"hash mismatch"):
                run_rtl.stage(8,self.destination,"structured")
        self.assertFalse(self.destination.exists())

    def test_truncated_trace_is_rejected_even_with_matching_file_hash(self):
        path = self.vectors/"conv0_acc.mem"
        path.write_text("\n".join(path.read_text().splitlines()[:-1])+"\n",encoding="ascii",newline="\n")
        manifest=json.loads((self.vectors/"manifest.json").read_text())
        manifest["files"][path.name]["sha256"]=run_rtl.sha(path)
        manifest["files"][path.name]["bytes"]=path.stat().st_size
        self.write_manifest(self.vectors,manifest)
        with patch.object(run_rtl,"ROOT",self.sandbox):
            with self.assertRaisesRegex(ValueError,"trace length"):
                run_rtl.stage(8,self.destination,"structured")
        self.assertFalse(self.destination.exists())

    def test_edit_after_manifest_validation_is_rejected_before_writing(self):
        original_read = run_rtl.read_manifest
        def mutate_after_validation(directory, captured=None):
            manifest = original_read(directory, captured)
            if directory == self.vectors:
                path = directory / "actions.mem"
                contents = path.read_bytes()
                # Preserve word count and syntax. The old runner could record
                # this changed content in a late snapshot after validating it.
                path.write_bytes((b"0" if contents[:1] != b"0" else b"1") + contents[1:])
            return manifest
        with patch.object(run_rtl, "ROOT", self.sandbox), patch.object(run_rtl, "read_manifest", side_effect=mutate_after_validation):
            with self.assertRaisesRegex(RuntimeError, "bundle changed during staging"):
                run_rtl.stage(8, self.destination, "structured")
        self.assertFalse(self.destination.exists())

    def test_staged_header_and_memory_tampering_are_detected(self):
        with patch.object(run_rtl,"ROOT",self.sandbox):
            for name in ("quantization.vh","weights.mem","expected_acc.mem"):
                with self.subTest(name=name):
                    evidence=run_rtl.stage(8,self.destination,"structured")
                    path=self.destination/name
                    path.write_bytes(path.read_bytes()+b"\n")
                    with self.assertRaisesRegex(RuntimeError,"staged memories"):
                        run_rtl.check_staged_inputs(evidence,self.destination)

    def test_staged_write_mutation_cannot_be_recorded_as_valid_input(self):
        original_write = Path.write_bytes
        def mutate_written_memory(path, contents):
            result = original_write(path, contents)
            if path == self.destination / "weights.mem":
                original_write(path, contents + b"\n")
            return result
        with patch.object(run_rtl, "ROOT", self.sandbox), patch.object(Path, "write_bytes", mutate_written_memory):
            with self.assertRaisesRegex(RuntimeError, "staged memories"):
                run_rtl.stage(8, self.destination, "structured")

    def test_source_bundle_change_after_staging_is_detected(self):
        with patch.object(run_rtl,"ROOT",self.sandbox):
            evidence=run_rtl.stage(8,self.destination,"structured")
            path=self.vectors/"actions.mem"
            path.write_bytes(path.read_bytes()+b"\n")
            with self.assertRaisesRegex(RuntimeError,"reference bundle changed"):
                run_rtl.check_staged_inputs(evidence,self.destination)

    def test_precision_mismatch_and_unsafe_filename_are_rejected(self):
        manifest=json.loads((self.vectors/"manifest.json").read_text())
        manifest["bits"]=4
        self.write_manifest(self.vectors,manifest)
        with patch.object(run_rtl,"ROOT",self.sandbox):
            with self.assertRaisesRegex(ValueError,"precision"):
                run_rtl.stage(8,self.destination,"structured")
        for name in ("../outside.mem", "..\\outside.mem"):
            manifest["files"]={name:{"sha256":"0"*64,"bytes":0}}
            self.write_manifest(self.vectors,manifest)
            with self.assertRaisesRegex(ValueError,"unsafe"):
                run_rtl.read_manifest(self.vectors)


if __name__=="__main__": unittest.main()
