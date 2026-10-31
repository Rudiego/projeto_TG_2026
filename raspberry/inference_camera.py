from __future__ import annotations

import argparse
import time
from collections import deque

import cv2
from picamera2 import Picamera2
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detecção drone/bird na câmera do Raspberry Pi.")
    parser.add_argument("--model", required=True, help="Pasta do modelo NCNN exportado.")
    parser.add_argument("--conf", type=float, default=0.45)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--process-every",
        type=int,
        default=1,
        help="Processa um quadro a cada N capturas para reduzir uso de CPU.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Executa sem abrir janela e sem salvar imagens.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.process_every < 1:
        raise ValueError("--process-every deve ser maior ou igual a 1.")

    model = YOLO(args.model)
    camera = Picamera2()
    camera.preview_configuration.main.size = (args.width, args.height)
    camera.preview_configuration.main.format = "RGB888"
    camera.preview_configuration.align()
    camera.configure("preview")
    camera.start()

    frame_index = 0
    fps_samples: deque[float] = deque(maxlen=30)
    print("Detecção iniciada. Pressione Ctrl+C para encerrar.")

    try:
        while True:
            frame = camera.capture_array()
            frame_index += 1
            if frame_index % args.process_every:
                continue

            started = time.perf_counter()
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device="cpu",
                verbose=False,
            )[0]
            elapsed = time.perf_counter() - started
            fps_samples.append(1.0 / elapsed if elapsed else 0.0)
            fps = sum(fps_samples) / len(fps_samples)

            detections = len(result.boxes) if result.boxes is not None else 0
            print(f"\rFPS={fps:5.2f} | detecções={detections:2d}", end="", flush=True)

            if not args.headless:
                annotated = result.plot()
                cv2.putText(
                    annotated,
                    f"FPS: {fps:.1f}",
                    (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Drone/Bird", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        print("\nEncerrando câmera.")
        camera.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
