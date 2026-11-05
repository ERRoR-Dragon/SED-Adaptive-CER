"""Tests for restarting a long run and scoring cached class counts."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sed_adaptive_cer.experiment import resume_rows
from sed_adaptive_cer.metrics import confusion_matrix, mean_iou, pixel_counts


class RunTests(unittest.TestCase):
    def test_resume_reuses_committed_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "validation.jsonl"
            path.write_bytes(b'{"id":"a","score":1}\n{"id":"b"')
            processed = []

            def calculate(image_id):
                processed.append(image_id)
                return {"score": 2}

            rows = resume_rows(path, ["a", "b"], calculate, "Test")
            self.assertEqual(processed, ["b"])
            self.assertEqual([row["score"] for row in rows], [1, 2])
            self.assertEqual(rows, [json.loads(line) for line in path.read_text().splitlines()])
            with self.assertRaises(ValueError):
                resume_rows(path, ["b", "a"], calculate, "Test")

    def test_class_counts_add_across_images(self):
        first = confusion_matrix(np.array([[0, 0, 1]]), np.array([[0, 0, 0]]), 2)
        second = confusion_matrix(np.array([[1, 255]]), np.array([[1, 0]]), 2)
        aggregated = pixel_counts(first) + pixel_counts(second)
        np.testing.assert_array_equal(aggregated, pixel_counts(first + second))
        self.assertAlmostEqual(mean_iou(aggregated), (2 / 3 + 1 / 2) / 2)
        self.assertIsNone(mean_iou(np.zeros((3, 2))))


if __name__ == "__main__":
    unittest.main()
