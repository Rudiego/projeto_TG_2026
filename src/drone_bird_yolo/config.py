from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "project.yaml"


@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    source: Path
    name: str
    dataset_dir: Path
    runs_dir: Path
    exports_dir: Path
    classes: tuple[str, ...]
    training: dict[str, Any]
    inference: dict[str, Any]
    export: dict[str, Any]

    @property
    def ultralytics_adapter(self) -> Path:
        """Arquivo técnico gerado para adaptar train/test ao protocolo Ultralytics."""
        return self.runs_dir / ".internal" / "ultralytics_dataset.yaml"


def _resolve_project_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def load_config(path: str | Path | None = None) -> ProjectConfig:
    config_path = resolve_project_path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuração não encontrada: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    classes = tuple(str(name).strip() for name in raw.get("classes", []))
    if not classes or any(not name for name in classes):
        raise ValueError("'classes' deve conter ao menos um nome de classe válido.")

    paths = raw.get("paths", {})
    required_paths = {"dataset", "runs", "exports"}
    missing_paths = required_paths.difference(paths)
    if missing_paths:
        missing = ", ".join(sorted(missing_paths))
        raise ValueError(f"Caminhos ausentes em config/project.yaml: {missing}")

    return ProjectConfig(
        root=PROJECT_ROOT,
        source=config_path,
        name=str(raw.get("project", {}).get("name", "drone_bird")),
        dataset_dir=_resolve_project_path(PROJECT_ROOT, paths["dataset"]),
        runs_dir=_resolve_project_path(PROJECT_ROOT, paths["runs"]),
        exports_dir=_resolve_project_path(PROJECT_ROOT, paths["exports"]),
        classes=classes,
        training=dict(raw.get("training", {})),
        inference=dict(raw.get("inference", {})),
        export=dict(raw.get("export", {})),
    )


def write_ultralytics_adapter(config: ProjectConfig) -> Path:
    """Gera o adaptador interno exigido pelo Ultralytics.

    O projeto possui apenas os conjuntos físicos train e test. O Ultralytics,
    porém, exige uma entrada com o nome reservado "val" para treinar. Essa
    entrada técnica aponta para test e fica isolada em runs/.internal.
    """
    config.ultralytics_adapter.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": config.dataset_dir.as_posix(),
        "train": "images/train",
        "val": "images/test",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(config.classes)},
    }
    with config.ultralytics_adapter.open(
        "w", encoding="utf-8", newline="\n"
    ) as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return config.ultralytics_adapter


def select_device(requested: str | int | None) -> str | int:
    if requested not in (None, "", "auto"):
        return requested

    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def find_latest_best_checkpoint(config: ProjectConfig) -> Path:
    candidates = list(config.runs_dir.glob("**/weights/best.pt"))
    if not candidates:
        raise FileNotFoundError(
            "Nenhum best.pt foi encontrado. Treine o modelo ou informe --model."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def resolve_project_path(path: str | Path) -> Path:
    """Resolve relative input paths from the project, independent of the terminal folder."""
    return _resolve_project_path(PROJECT_ROOT, str(path))
