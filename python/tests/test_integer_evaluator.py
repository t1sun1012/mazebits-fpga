"""Protect integer evaluation from mismatched sources and changed evidence."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate_integer_bundle as runner
from mazebits.dataset import generate_dataset
from mazebits.model_io import sha256
ROOT = Path(__file__).absolute().parents[2]

class IntegerEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.args = (ROOT/'model/cnn-w8a8', ROOT/'model/cnn-fp32', self.root/'data', self.root/'output')
    def tearDown(self):
        self.temporary.cleanup()
    def test_existing_output_is_rejected_before_loading_models(self):
        self.args[-1].mkdir()
        (self.args[-1]/'keep.txt').write_text('keep')
        with patch.object(runner, 'load_quantized', side_effect=AssertionError('must not load')):
            with self.assertRaises(FileExistsError): runner.evaluate_bundle(*self.args)
    def test_unrelated_float_source_is_rejected_before_dataset_read(self):
        with patch.object(runner, 'load_model', return_value=(None, {'weights_sha256': 'unrelated'})), patch.object(runner, 'load_split', side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError, 'source checkpoint'): runner.evaluate_bundle(*self.args)
    def test_changed_evaluation_source_is_rejected_before_publishing(self):
        generate_dataset(self.args[2], train_layouts=1,val_layouts=1,test_layouts=1,
                         goals_per_layout=1,episodes_per_goal=2)
        watched = Path(runner.__file__).absolute()
        calls = 0
        def changed_hash(path):
            nonlocal calls
            if Path(path).absolute() == watched:
                calls += 1
                if calls > 1: return '0'*64
            return sha256(path)
        with patch.object(runner, 'sha256', side_effect=changed_hash), patch.object(runner, 'float_scores', return_value=np.zeros((1,4))), patch.object(runner, 'evaluate_integer', return_value=({}, {'summary':{},'episodes':[]}, np.zeros((1,4),dtype=np.int32))):
            with self.assertRaisesRegex(RuntimeError, 'source changed'): runner.evaluate_bundle(*self.args)
        self.assertFalse(self.args[-1].exists())

if __name__ == '__main__': unittest.main()
