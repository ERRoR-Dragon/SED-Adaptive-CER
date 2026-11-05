"""Regression checks for budget labels, coarse features, and ADE scoring."""

import unittest

import numpy as np
import torch

from sed_adaptive_cer.metrics import confusion_matrix, mean_iou, pixel_counts
from sed_adaptive_cer.policy import BUDGETS, fit_head, oracle_index, scene_features


class PolicyTests(unittest.TestCase):
    def test_oracle_respects_tolerance_and_nonmonotonic_scores(self):
        self.assertEqual(oracle_index([0.30, 0.497, 0.45, 0.50, 0.50], 0.005), 1)
        self.assertEqual(oracle_index([0.30, 0.40, 0.45, 0.49, 0.50], 0.005), 4)

    def test_void_pixels_excluded_from_confusion(self):
        target = np.array([[0, 1], [255, 1]])
        prediction = np.array([[0, 0], [1, 1]])
        counts = confusion_matrix(target, prediction, 2)
        np.testing.assert_array_equal(counts, [[1, 0], [1, 1]])
        self.assertEqual(mean_iou(pixel_counts(counts)), 0.5)

    def test_scene_features_are_order_invariant_and_finite(self):
        logits = torch.randn(1, 9, 4, 4, generator=torch.Generator().manual_seed(7))
        torch.testing.assert_close(scene_features(logits), scene_features(logits[:, [4, 2, 8, 7, 3, 0, 1, 5, 6]]))
        self.assertTrue(torch.isfinite(scene_features(logits[:, :1])).all())
        with self.assertRaises(ValueError):
            scene_features(logits.repeat(2, 1, 1, 1))

    def test_fit_and_runtime_dispatch(self):
        torch.set_num_threads(2)
        train_features = torch.zeros(6, 10)
        labels = torch.ones(6, dtype=torch.long)
        model, stats = fit_head(train_features, labels, train_features[:2], labels[:2], seed=42)
        self.assertEqual(stats["dev_accuracy"], 1.0)
        self.assertFalse(model.active.any())
        self.assertIn(model.budget(torch.randn(1, 9, 4, 4)), BUDGETS)


if __name__ == "__main__":
    unittest.main()
