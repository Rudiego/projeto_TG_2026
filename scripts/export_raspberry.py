from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import find_latest_best_checkpoint, resolve_project_path, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta o modelo para o Raspberry Pi.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--format", choices=("ncnn", "onnx", "openvino"))
    parser.add_argument("--imgsz", type=int)
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Exporta NCNN em FP16. Compare precisão e velocidade antes de usar.",
    )
    return parser.parse_args()


def _unique_destination(directory: Path, name: str) -> Path:
    candidate = directory / name
    if not candidate.exists():
        return candidate
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"{candidate.stem}_{stamp}{candidate.suffix}"


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model_path = resolve_project_path(args.model) if args.model else find_latest_best_checkpoint(config)
    export_format = args.format or str(config.export.get("format", "ncnn"))
    image_size = args.imgsz or int(config.export.get("image_size", 640))
    fp16 = args.fp16 or bool(config.export.get("fp16", False))

    if fp16 and export_format != "ncnn":
        raise ValueError("A opção --fp16 deste projeto está preparada apenas para NCNN.")

    from ultralytics import YOLO

    export_args: dict[str, object] = {
        "format": export_format,
        "imgsz": image_size,
        "batch": 1,
        "device": "cpu",
    }
    if fp16:
        export_args["quantize"] = 16

    print(f"Exportando {model_path} para {export_format}...")
    exported_path = Path(YOLO(str(model_path)).export(**export_args)).resolve()
    config.exports_dir.mkdir(parents=True, exist_ok=True)
    destination = _unique_destination(config.exports_dir, exported_path.name)

    if exported_path.is_dir():
        shutil.copytree(exported_path, destination)
    else:
        shutil.copy2(exported_path, destination)

    print(f"\nModelo pronto para copiar ao Raspberry: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

