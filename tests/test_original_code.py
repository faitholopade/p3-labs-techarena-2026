"""Regression checks on original competition functions; no model training."""
import importlib.util
from pathlib import Path
import sys
import unittest
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import split_data, binary_labels
spec = importlib.util.spec_from_file_location('v142_replay', ROOT / 'experiments/v142/replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


class OriginalCodeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(142)
        self.model = dict(catalog=rng.normal(size=(30, 4)), mean=np.zeros(4),
                          std=np.ones(4), weight=rng.normal(size=(1, 4)))
        self.inputs = dict(items=rng.integers(0, 30, size=(12, 10)), context=np.arange(12) % 8)
        self.base = rng.uniform(.001, .05, size=(12, 10))

    def test_v142_preserves_event_mass(self):
        values, delta = replay.predict(self.model, self.inputs, self.base.ravel())
        assert_allclose(values.reshape(-1, 10).sum(1), self.base.sum(1), atol=1e-14)
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue((values >= 0).all())
        self.assertTrue((abs(delta) <= .2).all())

    def test_v142_zero_weights_preserve_baseline(self):
        self.model['weight'][:] = 0
        values, _ = replay.predict(self.model, self.inputs, self.base.ravel())
        assert_allclose(values, self.base.ravel(), atol=1e-14)

    def test_v142_candidate_permutation(self):
        perm = np.array([3, 1, 8, 0, 9, 4, 6, 2, 7, 5])
        p, _ = replay.predict(self.model, self.inputs, self.base.ravel())
        inputs = dict(self.inputs, items=self.inputs['items'][:, perm])
        q, _ = replay.predict(self.model, inputs, self.base[:, perm].ravel())
        assert_allclose(q.reshape(-1, 10), p.reshape(-1, 10)[:, perm], atol=1e-14)

    def test_v142_independent_probability_transfer(self):
        p, delta = replay.predict(self.model, self.inputs, self.base.ravel())
        weighted = self.base * np.exp(.5 * delta)
        expected = weighted / weighted.sum(1, keepdims=True) * self.base.sum(1, keepdims=True)
        assert_allclose(p, expected.ravel(), atol=1e-14)

    def test_split_keeps_first_post_cutoff_event_per_user(self):
        data = dict(timestamp=np.array([8, 14, 11, 9, 12, 13]), user=np.array([1, 1, 2, 2, 1, 2]))
        fit, validation = split_data(data, 10)
        assert_array_equal(fit, [0, 3])
        assert_array_equal(validation, [2, 4])

    def test_no_click_target_produces_no_positive_candidate(self):
        labels = binary_labels(dict(target=np.array([0, 9, 10])))
        assert_array_equal(labels.sum(1), [1, 1, 0])
        self.assertEqual(labels[0, 0], 1)
        self.assertEqual(labels[1, 9], 1)


if __name__ == '__main__':
    unittest.main()
