from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass
class SplitReport:
    split: str
    images: int = 0
    labels: int = 0
    background_images: int = 0
    objects: int = 0
    class_counts: Counter[int] = field(default_factory=Counter)


@dataclass(frozen=True)
class DatasetIssue:
    level: str
    message: str


@dataclass
class DatasetReport:
    splits: list[SplitReport] = field(default_factory=list)
    issues: list[DatasetIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[DatasetIssue]:
        return [issue for issue in self.issues if issue.level == "ERRO"]

    @property
    def warnings(self) -> list[DatasetIssue]:
        return [issue for issue in self.issues if issue.level == "AVISO"]


def _relative_key(path: Path, base: Path) -> str:
    return path.relative_to(base).with_suffix("").as_posix().casefold()


def _validate_label(
    label_path: Path,
    class_count: int,
    report: DatasetReport,
    split_report: SplitReport,
) -> None:
    try:
        lines = label_path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        report.issues.append(
            DatasetIssue("ERRO", f"Não foi possível ler {label_path}: {exc}")
        )
        return

    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            report.issues.append(
                DatasetIssue(
                    "ERRO",
                    f"{label_path}:{line_number} deve ter 5 valores, recebeu {len(fields)}.",
                )
            )
            continue
        try:
            class_raw, x_center, y_center, width, height = map(float, fields)
        except ValueError:
            report.issues.append(
                DatasetIssue("ERRO", f"{label_path}:{line_number} contém valor não numérico.")
            )
            continue

        class_id = int(class_raw)
        if class_raw != class_id or not 0 <= class_id < class_count:
            report.issues.append(
                DatasetIssue(
                    "ERRO",
                    f"{label_path}:{line_number} usa classe inválida {class_raw}.",
                )
            )
            continue

        coordinates = (x_center, y_center, width, height)
        if not all(0.0 <= value <= 1.0 for value in coordinates):
            report.issues.append(
                DatasetIssue(
                    "ERRO",
                    f"{label_path}:{line_number} possui coordenada fora do intervalo [0, 1].",
                )
            )
            continue
        if width <= 0.0 or height <= 0.0:
            report.issues.append(
                DatasetIssue(
                    "ERRO",
                    f"{label_path}:{line_number} possui largura ou altura igual a zero.",
                )
            )
            continue

        split_report.objects += 1
        split_report.class_counts[class_id] += 1


def validate_dataset(
    dataset_dir: Path,
    class_count: int,
    required_splits: tuple[str, ...] = ("train", "test"),
) -> DatasetReport:
    report = DatasetReport()

    for split in required_splits:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split
        split_report = SplitReport(split=split)
        report.splits.append(split_report)

        if not images_dir.is_dir():
            level = "ERRO" if split in required_splits else "AVISO"
            report.issues.append(DatasetIssue(level, f"Pasta ausente: {images_dir}"))
            continue
        if not labels_dir.is_dir():
            level = "ERRO" if split in required_splits else "AVISO"
            report.issues.append(DatasetIssue(level, f"Pasta ausente: {labels_dir}"))
            continue

        images = sorted(
            path for path in images_dir.rglob("*") if path.suffix.casefold() in IMAGE_SUFFIXES
        )
        labels = sorted(labels_dir.rglob("*.txt"))
        split_report.images = len(images)
        split_report.labels = len(labels)

        if split in required_splits and not images:
            report.issues.append(
                DatasetIssue("ERRO", f"Nenhuma imagem encontrada em {images_dir}")
            )

        image_map: dict[str, Path] = {}
        for image_path in images:
            key = _relative_key(image_path, images_dir)
            if key in image_map:
                report.issues.append(
                    DatasetIssue(
                        "ERRO",
                        f"Imagens com o mesmo nome relativo: {image_map[key]} e {image_path}",
                    )
                )
            image_map[key] = image_path

        label_map = {_relative_key(path, labels_dir): path for path in labels}
        orphan_labels = sorted(set(label_map).difference(image_map))
        for key in orphan_labels:
            report.issues.append(
                DatasetIssue("ERRO", f"Label sem imagem correspondente: {label_map[key]}")
            )

        background_keys = set(image_map).difference(label_map)
        split_report.background_images = len(background_keys)
        if background_keys:
            report.issues.append(
                DatasetIssue(
                    "AVISO",
                    f"{split}: {len(background_keys)} imagem(ns) sem label; serão tratadas como fundo.",
                )
            )

        for key in sorted(set(label_map).intersection(image_map)):
            _validate_label(label_map[key], class_count, report, split_report)

    return report


def print_report(report: DatasetReport, class_names: tuple[str, ...]) -> None:
    print("\nResumo do dataset")
    print("-" * 72)
    for split in report.splits:
        counts = ", ".join(
            f"{class_names[index]}={split.class_counts.get(index, 0)}"
            for index in range(len(class_names))
        )
        print(
            f"{split.split:>5}: imagens={split.images:<6} labels={split.labels:<6} "
            f"fundo={split.background_images:<6} objetos={split.objects:<6} [{counts}]"
        )

    if report.issues:
        print("\nDiagnóstico")
        print("-" * 72)
        for issue in report.issues:
            print(f"[{issue.level}] {issue.message}")
    else:
        print("\nDataset válido, sem avisos.")
