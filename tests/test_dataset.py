from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.dataset import validate_dataset


class DatasetValidationTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        for split in ("train", "test"):
            (root / "images" / split).mkdir(parents=True)
            (root / "labels" / split).mkdir(parents=True)

    def test_valid_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            self._make_dataset(dataset)
            for split in ("train", "test"):
                (dataset / "images" / split / "sample.jpg").write_bytes(b"fake")
                (dataset / "labels" / split / "sample.txt").write_text(
                    "0 0.5 0.5 0.2 0.2\n1 0.3 0.4 0.1 0.1\n",
                    encoding="utf-8",
                )

            report = validate_dataset(dataset, class_count=2)

            self.assertFalse(report.errors)
            self.assertEqual(report.splits[0].objects, 2)
            self.assertEqual(report.splits[0].class_counts[0], 1)
            self.assertEqual(report.splits[0].class_counts[1], 1)

    def test_invalid_class_and_coordinate_are_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            self._make_dataset(dataset)
            for split in ("train", "test"):
                (dataset / "images" / split / "sample.png").write_bytes(b"fake")
                (dataset / "labels" / split / "sample.txt").write_text(
                    "2 1.2 0.5 0.2 0.2\n",
                    encoding="utf-8",
                )

            report = validate_dataset(dataset, class_count=2)

            self.assertTrue(report.errors)

    def test_image_without_label_is_background_warning(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            self._make_dataset(dataset)
            for split in ("train", "test"):
                (dataset / "images" / split / "background.jpg").write_bytes(b"fake")

            report = validate_dataset(dataset, class_count=2)

            self.assertFalse(report.errors)
            self.assertTrue(report.warnings)
            self.assertEqual(report.splits[0].background_images, 1)


if __name__ == "__main__":
    unittest.main()
