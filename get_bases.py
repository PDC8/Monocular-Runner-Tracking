#!/usr/bin/env python3
"""
Build an empty-field background from a video, then detect home/first/second/third bases.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["home", "first", "second", "third"]
COLORS = {
    "home": (0, 0, 255),
    "first": (0, 255, 0),
    "second": (255, 0, 0),
    "third": (0, 255, 255),
}


def parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent
    models_dir = this_dir / "models"
    parser = argparse.ArgumentParser(
        description="Average video frames into a background image and detect base locations with YOLO."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for background and metadata.")
    parser.add_argument(
        "--base-model",
        type=Path,
        default=models_dir / "base_best.pt",
        help="YOLO weights for base detection.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference image size.")
    parser.add_argument("--device", type=str, default="0", help="Inference device for YOLO.")
    parser.add_argument(
        "--frame-step",
        type=int,
        default=1,
        help="Use every Nth frame when averaging (1 means all frames).",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1280,
        help="Resize frames to this max width before averaging (0 disables resize).",
    )
    return parser.parse_args()


def maybe_resize(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width > 0 and frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        new_w = max_width
        new_h = int(round(frame.shape[0] * scale))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def compute_average_background(video_path: Path, frame_step: int, max_width: int) -> tuple[np.ndarray, int, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = 0
    used_count = 0
    acc: np.ndarray | None = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        if frame_step > 1 and (frame_count % frame_step) != 0:
            frame_count += 1
            continue

        frame = maybe_resize(frame, max_width=max_width)
        frame_f64 = frame.astype(np.float64)
        if acc is None:
            acc = np.zeros_like(frame_f64)
        acc += frame_f64
        used_count += 1
        frame_count += 1

    cap.release()

    if acc is None or used_count == 0:
        raise RuntimeError(f"No frames were usable for averaging: {video_path}")

    bg = np.clip(acc / float(used_count), 0, 255).astype(np.uint8)
    return bg, used_count, fps


def draw_box(img: np.ndarray, xyxy: list[float], label: str, color: tuple[int, int, int], thickness: int = 2) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    y_text = max(0, y1 - 8)
    cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text + 4), (0, 0, 0), -1)
    cv2.putText(img, label, (x1 + 3, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)


def detect_bases(
    background_bgr: np.ndarray,
    model_path: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> tuple[dict[str, list[float]], dict[str, dict[str, float | list[float]]], np.ndarray]:
    model = YOLO(str(model_path))
    result = model.predict(source=background_bgr, conf=conf, imgsz=imgsz, device=device, verbose=False)[0]

    if result.boxes is None or len(result.boxes) == 0:
        raise RuntimeError("Base detector returned no detections.")

    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    clss = result.boxes.cls.cpu().numpy().astype(int)

    best: dict[int, dict[str, float | list[float]]] = {}
    for i in range(len(confs)):
        cls_id = int(clss[i])
        if cls_id < 0 or cls_id >= len(CLASS_NAMES):
            continue
        if cls_id not in best or float(confs[i]) > float(best[cls_id]["conf"]):
            best[cls_id] = {
                "conf": float(confs[i]),
                "box_xyxy": [float(v) for v in boxes_xyxy[i].tolist()],
            }

    missing = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if i not in best]
    if missing:
        raise RuntimeError(f"Missing base detections for: {missing}")

    debug_img = background_bgr.copy()
    base_points: dict[str, list[float]] = {}
    detections: dict[str, dict[str, float | list[float]]] = {}

    for cls_id, class_name in enumerate(CLASS_NAMES):
        rec = best[cls_id]
        box = [float(v) for v in rec["box_xyxy"]]
        x1, y1, x2, y2 = box
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        base_points[class_name] = [cx, cy]
        detections[class_name] = {
            "conf": float(rec["conf"]),
            "box_xyxy": box,
            "center_xy": [cx, cy],
        }

        color = COLORS[class_name]
        draw_box(debug_img, box, f"{class_name.upper()} {float(rec['conf']):.2f}", color, thickness=3)
        cv2.circle(debug_img, (int(round(cx)), int(round(cy))), 5, color, -1)

    return base_points, detections, debug_img


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.base_model.exists():
        raise FileNotFoundError(f"Base model not found: {args.base_model}")

    bg, used_frames, fps = compute_average_background(
        video_path=video_path,
        frame_step=max(1, int(args.frame_step)),
        max_width=int(args.max_width),
    )

    bg_path = args.output_dir / f"{video_path.stem}_bg.jpg"
    cv2.imwrite(str(bg_path), bg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    base_points, detections, debug_img = detect_bases(
        background_bgr=bg,
        model_path=args.base_model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    debug_path = args.output_dir / f"{video_path.stem}_bases_dbg.jpg"
    cv2.imwrite(str(debug_path), debug_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    meta = {
        "video_path": str(video_path),
        "video_fps": fps,
        "background_image": str(bg_path),
        "debug_image": str(debug_path),
        "base_detector_weights": str(args.base_model),
        "frame_step": int(args.frame_step),
        "frames_used_for_average": used_frames,
        "base_points_px": base_points,
        "detections": detections,
        "infield_polygon_order": ["home", "first", "second", "third"],
    }

    out_meta = args.output_dir / "run_meta_bases.json"
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Base detection complete.")
    print(f"Background: {bg_path}")
    print(f"Debug:      {debug_path}")
    print(f"Meta:       {out_meta}")


if __name__ == "__main__":
    main()
