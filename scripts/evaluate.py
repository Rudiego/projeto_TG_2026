from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import (
    find_latest_best_checkpoint,
    load_config,
    resolve_project_path,
    select_device,
    write_ultralytics_adapter,
)
from drone_bird_yolo.dataset import print_report, validate_dataset
from drone_bird_yolo.metrics import save_class_only_confusion_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Avalia um detector YOLO corretamente.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--device", help="auto, cpu ou índice CUDA, como 0.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = validate_dataset(config.dataset_dir, len(config.classes))
    print_report(report, config.classes)
    if report.errors:
        print("\nAvaliação cancelada: o dataset contém erros.")
        return 1

    test_report = next(item for item in report.splits if item.split == "test")
    if test_report.images == 0:
        print("\nAvaliação cancelada: a divisão test está vazia.")
        return 1

    adapter_path = write_ultralytics_adapter(config)
    model_path = resolve_project_path(args.model) if args.model else find_latest_best_checkpoint(config)
    device = select_device(args.device or config.training.get("device", "auto"))
    image_size = args.imgsz or int(config.training.get("image_size", 640))
    batch = args.batch or int(config.training.get("batch", 4))
    confidence = (
        args.conf
        if args.conf is not None
        else float(
            config.training.get(
                "validation_confidence",
                config.inference.get("confidence", 0.45),
            )
        )
    )
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("A confiança deve estar entre 0 e 1.")

    from ultralytics import YOLO

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(adapter_path),
        split="test",
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=int(config.training.get("workers", 2)),
        conf=confidence,
        plots=True,
        project=str(config.runs_dir / "evaluation"),
        name=f"{model_path.stem}_test",
    )
    save_class_only_confusion_matrices(metrics, config.classes, metrics.save_dir)

    print("\nMétricas de detecção")
    print("-" * 72)
    print(f"Confiança : {confidence:.2f}")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}")
    print(f"Precisão : {metrics.box.mp:.4f}")
    print(f"Recall   : {metrics.box.mr:.4f}")
    for index, class_map in enumerate(metrics.box.maps):
        class_name = config.classes[index] if index < len(config.classes) else str(index)
        print(f"mAP50-95 {class_name:>8}: {class_map:.4f}")

    print(f"\nGráficos e relatório: {metrics.save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
