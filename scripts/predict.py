from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import find_latest_best_checkpoint, resolve_project_path, load_config, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa uma predição local.")
    parser.add_argument("source", help="Imagem, vídeo, pasta ou índice de webcam.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", help="auto, cpu ou índice CUDA, como 0.")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_project_path(args.model) if args.model else find_latest_best_checkpoint(config)
    source: str | int = int(args.source) if args.source.isdigit() else (args.source if "://" in args.source else str(resolve_project_path(args.source)))

    from ultralytics import YOLO

    results = YOLO(str(model_path)).predict(
        source=source,
        conf=(
            args.conf
            if args.conf is not None
            else float(config.inference.get("confidence", 0.45))
        ),
        iou=float(config.inference.get("iou", 0.70)),
        imgsz=args.imgsz or int(config.inference.get("image_size", 640)),
        device=select_device(args.device or config.training.get("device", "auto")),
        save=True,
        show=args.show,
        project=str(config.runs_dir / "predict"),
        name="prediction",
    )
    if results:
        print(f"Resultado salvo em: {results[0].save_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
