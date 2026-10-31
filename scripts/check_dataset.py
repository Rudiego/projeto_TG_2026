from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from drone_bird_yolo.config import load_config, write_ultralytics_adapter
from drone_bird_yolo.dataset import print_report, validate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Valida imagens e bounding boxes YOLO.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Também falha quando existem imagens sem arquivo de label.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    report = validate_dataset(config.dataset_dir, len(config.classes))
    print_report(report, config.classes)

    strict_failure = args.strict and bool(report.warnings)
    if report.errors or strict_failure:
        print("\nValidação reprovada. Corrija os itens acima antes do treinamento.")
        return 1

    yaml_path = write_ultralytics_adapter(config)
    print(f"\nAdaptador interno do Ultralytics gerado em: {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
