from __future__ import annotations

import sys
import os
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import ProjectConfig, write_ultralytics_adapter, load_config, resolve_project_path


class UltralyticsAdapterTests(unittest.TestCase):
    def test_paths_do_not_depend_on_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                config = load_config("config/project.yaml")
                self.assertEqual(config.root, ROOT)
                self.assertEqual(resolve_project_path("yolo26n.pt"), ROOT / "yolo26n.pt")
                self.assertEqual(resolve_project_path(ROOT / "weights"), ROOT / "weights")
            finally:
                os.chdir(previous)

    def test_adapter_uses_only_physical_train_and_test_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = ProjectConfig(
                root=root,
                source=root / "project.yaml",
                name="test",
                dataset_dir=root / "dataset",
                runs_dir=root / "runs",
                exports_dir=root / "exports",
                classes=("drone", "bird"),
                training={},
                inference={},
                export={},
            )

            adapter_path = write_ultralytics_adapter(config)
            payload = yaml.safe_load(adapter_path.read_text(encoding="utf-8"))

            self.assertEqual(adapter_path.parent, config.runs_dir / ".internal")
            self.assertEqual(payload["train"], "images/train")
            self.assertEqual(payload["test"], "images/test")
            self.assertEqual(payload["val"], "images/test")
            self.assertFalse((config.dataset_dir / "data.yaml").exists())


if __name__ == "__main__":
    unittest.main()
