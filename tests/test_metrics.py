from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.metrics import class_only_confusion_arrays


class ClassOnlyConfusionMatrixTests(unittest.TestCase):
    def test_removes_background_but_preserves_original_normalization(self) -> None:
        matrix = np.array(
            [
                [8, 1, 2],
                [1, 6, 3],
                [1, 3, 0],
            ],
            dtype=float,
        )

        raw, normalized = class_only_confusion_arrays(matrix, class_count=2)

        np.testing.assert_array_equal(raw, np.array([[8, 1], [1, 6]], dtype=float))
        np.testing.assert_allclose(normalized, np.array([[0.8, 0.1], [0.1, 0.6]]))

    def test_rejects_matrix_smaller_than_class_count(self) -> None:
        with self.assertRaises(ValueError):
            class_only_confusion_arrays(np.zeros((1, 1)), class_count=2)


if __name__ == "__main__":
    unittest.main()
