"""Verify improved navigation fixtures, scalar coverage, and model-set binding."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))
from run_navigation import verify_vector_bundle
from export_structured_navigation_vectors import MAZES, export
from mazebits.maze import step


class StructuredNavigationTests(unittest.TestCase):
    def setUp(self):
        (ROOT / "runs").mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / "runs")
        self.copy = Path(self.temporary.name) / "vectors"
        shutil.copytree(ROOT / "test_vectors/navigation-structured", self.copy)

    def tearDown(self):
        self.temporary.cleanup()

    def manifest(self):
        return json.loads((self.copy / "manifest.json").read_text())

    def save(self, manifest):
        (self.copy / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def test_both_model_sets_keep_complete_scalar_coverage(self):
        old, _ = verify_vector_bundle(ROOT / "test_vectors/navigation")
        new, hashes = verify_vector_bundle(self.copy, "structured")
        self.assertEqual(old["case_count"], 4)
        self.assertEqual(old["scalar_checked_decisions"], 60)
        self.assertEqual(new["case_count"], 7)
        self.assertEqual(new["scalar_checked_decisions"], 157)
        self.assertEqual(len(new["episodes"]), 14)
        self.assertEqual(sum(len(e["events"]) for e in new["episodes"]), 157)
        self.assertEqual({case["name"] for case in new["cases"]}, set(MAZES))
        self.assertIn("model/cnn-structured-w4a4/manifest.json", hashes)

    def test_wrong_model_set_or_model_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "contract or model set"):
            verify_vector_bundle(self.copy, "original")
        manifest = self.manifest()
        manifest["source_sha256"]["model/cnn-structured-w8a8/manifest.json"] = hashlib.sha256(
            (ROOT / "model/cnn-w8a8/manifest.json").read_bytes()).hexdigest()
        self.save(manifest)
        with self.assertRaisesRegex(ValueError, "different model set"):
            verify_vector_bundle(self.copy, "structured")

    def test_missing_episode_or_scalar_check_is_rejected(self):
        original = self.manifest()
        manifest = json.loads(json.dumps(original))
        manifest["episodes"].pop()
        self.save(manifest)
        with self.assertRaisesRegex(ValueError, "episode coverage"):
            verify_vector_bundle(self.copy, "structured")
        original["scalar_checked_decisions"] -= 1
        self.save(original)
        with self.assertRaisesRegex(ValueError, "scalar oracle"):
            verify_vector_bundle(self.copy, "structured")

    def test_corrupt_and_wrong_length_words_are_rejected(self):
        path = self.copy / "events_w8a8.mem"
        path.write_bytes(path.read_bytes() + b"bad\n")
        with self.assertRaisesRegex(ValueError, "corruption"):
            verify_vector_bundle(self.copy, "structured")
        manifest = self.manifest()
        manifest["files"][path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
        self.save(manifest)
        with self.assertRaisesRegex(ValueError, "vector words"):
            verify_vector_bundle(self.copy, "structured")

    def test_reference_paths_and_terminals_are_unrepaired_model_actions(self):
        manifest, _ = verify_vector_bundle(self.copy, "structured")
        for episode in manifest["episodes"]:
            maze = MAZES[episode["case"]]
            position, visited = maze.start, {maze.start}
            trail = 1 << (position[1]*8+position[0])
            for index, event in enumerate(episode["events"]):
                action = max(range(4), key=lambda a: event["scores"][a])
                self.assertEqual(event["action"], action)
                moved = step(maze, position, action)
                position = moved.position
                trail |= 1 << (position[1]*8+position[0])
                self.assertEqual(event["position"], position[1]*8+position[0])
                self.assertEqual(int(event["trail_hex"],16), trail)
                self.assertEqual(event["steps"], index+1)
                self.assertEqual(event["total_cycles"], (index+1)*3217153)
                status = 2 if moved.done else 3 if moved.blocked else 4 if position in visited else 5 if index+1==64 else 1
                self.assertEqual(event["status"], status)
                visited.add(position)
            self.assertEqual(episode["final_position"], position[1]*8+position[0])
            self.assertEqual(episode["steps"], len(episode["events"]))
            self.assertEqual(episode["outcome"], "success" if position==maze.goal else {3:"blocked",4:"loop",5:"timeout"}[episode["events"][-1]["status"]])

    def test_export_refuses_to_overwrite_frozen_evidence(self):
        before = (self.copy / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            export(self.copy)
        self.assertEqual(before, (self.copy / "manifest.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
