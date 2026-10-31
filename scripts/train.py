from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import (
    load_config,
    resolve_project_path,
    select_device,
    write_ultralytics_adapter,
)
from drone_bird_yolo.dataset import print_report, validate_dataset
from drone_bird_yolo.metrics import save_class_only_confusion_matrices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina o detector drone/bird localmente.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", help="Checkpoint inicial, por exemplo yolo26n.pt.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--device", help="auto, cpu ou índice CUDA, como 0.")
    parser.add_argument("--resume", type=Path, help="Checkpoint last.pt a retomar.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    training = config.training

    report = validate_dataset(config.dataset_dir, len(config.classes))
    print_report(report, config.classes)
    if report.errors:
        print("\nTreinamento cancelado: o dataset contém erros.")
        return 1

    adapter_path = write_ultralytics_adapter(config)
    config.runs_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    if args.resume:
        resume_path = resolve_project_path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Checkpoint não encontrado: {resume_path}")
        print(f"\nRetomando treinamento: {resume_path}")
        model = YOLO(str(resume_path))
        if not model.ckpt or model.ckpt.get("epoch", -1) < 0 or model.ckpt.get("optimizer") is None:
            raise ValueError("Este checkpoint encerrou o treino e não pode ser retomado. Use --model para iniciar um novo treino com esses pesos.")
        results = model.train(
            resume=True,
            data=str(adapter_path),
            save_dir=str(resume_path.parent.parent),
            device=select_device(args.device or training.get("device", "auto")),
        )
        save_class_only_confusion_matrices(results, config.classes, results.save_dir)
        return 0

    device = select_device(args.device or training.get("device", "auto"))
    model_name = str(resolve_project_path(args.model or str(training.get("model", "yolo26n.pt"))))
    if not Path(model_name).is_file():
        raise FileNotFoundError(f"Modelo inicial não encontrado: {model_name}")
    epochs = args.epochs or int(training.get("epochs", 100))
    image_size = args.imgsz or int(training.get("image_size", 640))
    batch = args.batch or int(training.get("batch", 4))
    patience = (
        args.patience
        if args.patience is not None
        else int(training.get("patience", 0))
    )
    confidence = float(
        training.get(
            "validation_confidence",
            config.inference.get("confidence", 0.45),
        )
    )
    if patience < 0:
        raise ValueError("A paciência deve ser um número inteiro maior ou igual a zero.")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("A confiança deve estar entre 0 e 1.")

    print("\nConfiguração do treinamento")
    print("-" * 72)
    print(f"Modelo inicial : {model_name}")
    print(f"Dispositivo    : {device}")
    print(f"Épocas         : {epochs}")
    print(f"Imagem         : {image_size} px")
    print(f"Batch          : {batch}")
    print(f"Paciência      : {'desativada' if patience == 0 else patience}")
    print(f"Conf. validação: {confidence:.2f}")

    model = YOLO(model_name)
    results = model.train(
        data=str(adapter_path),
        epochs=epochs,
        imgsz=image_size,
        batch=batch,
        device=device,
        workers=int(training.get("workers", 2)),
        patience=patience,
        conf=confidence,
        amp=bool(training.get("amp", True)),
        cache=training.get("cache", False),
        seed=int(training.get("seed", 42)),
        close_mosaic=int(training.get("close_mosaic", 10)),
        optimizer=str(training.get("optimizer", "auto")),
        degrees=float(training.get("degrees", 0.0)),
        translate=float(training.get("translate", 0.1)),
        scale=float(training.get("scale", 0.5)),
        fliplr=float(training.get("fliplr", 0.5)),
        mosaic=float(training.get("mosaic", 1.0)),
        mixup=float(training.get("mixup", 0.0)),
        cos_lr=bool(training.get("cos_lr", True)),
        deterministic=True,
        pretrained=True,
        plots=True,
        project=str(config.runs_dir / "detect"),
        name=f"train_{config.name}",
    )

    save_dir = Path(results.save_dir)
    save_class_only_confusion_matrices(results, config.classes, save_dir)
    print(f"\nTreinamento concluído: {save_dir}")
    print(f"Melhor checkpoint: {save_dir / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
