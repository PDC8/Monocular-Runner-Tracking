#!/usr/bin/env python3
"""
Extract frames, run YOLO + ByteTrack player tracking, then run SAM ground-contact refinement.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def default_inference_device() -> str:
    if torch.cuda.is_available():
        return "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    this_dir = Path(__file__).resolve().parent
    models_dir = this_dir / "models"

    parser = argparse.ArgumentParser(
        description="Run player detection/tracking/contact pipeline from a video."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument(
        "--ground-contact-script",
        "--detect-script",
        dest="ground_contact_script",
        type=Path,
        default=this_dir / "get_ground_contact.py",
        help="Path to get_ground_contact.py",
    )
    parser.add_argument(
        "--yolo-weights",
        type=Path,
        default=models_dir / "player_detect_best.pt",
        help="YOLO weights for player detection (default: models/player_detect_best.pt).",
    )
    parser.add_argument(
        "--sam-weights",
        type=str,
        default=str(models_dir / "sam2.1_l.pt"),
        help="SAM weights path or model name (default: Thesis_Final/models/sam2.1_l.pt).",
    )
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config.")
    parser.add_argument(
        "--person-class",
        type=int,
        default=4,
        help="YOLO class for players (default: 4 for player_detect_best.pt).",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="YOLO confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold.")
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size.")
    parser.add_argument(
        "--contact-bottom-frac",
        type=float,
        default=0.08,
        help="Bottom fraction of SAM mask pixels used for robust contact-point estimation.",
    )
    parser.add_argument(
        "--contact-mode",
        type=str,
        default="hybrid",
        choices=["sam", "bbox_bottom", "hybrid"],
        help=(
            "Ground-contact mode passed to get_ground_contact.py: "
            "sam | bbox_bottom | hybrid (default: hybrid)."
        ),
    )
    parser.add_argument(
        "--contact-smooth-method",
        type=str,
        default="sliding_median",
        choices=["none", "sliding_median"],
        help=(
            "Per-track smoothing for SAM contact trajectories passed to get_ground_contact.py: "
            "none | sliding_median."
        ),
    )
    parser.add_argument(
        "--contact-smooth-window",
        type=int,
        default=5,
        help="Sliding-median window size (samples) for contact trajectory smoothing.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=default_inference_device(),
        help="Inference device (default: auto-detect CUDA, then MPS, else CPU).",
    )
    parser.add_argument("--fps", type=float, default=0.0, help="Override output video FPS (0 uses source video FPS).")
    parser.add_argument("--max-frames", type=int, default=0, help="Max extracted/processed frames (0=all).")
    parser.add_argument("--no-video", action="store_true", help="Disable overlay video from tracking stage.")
    return parser.parse_args()


def extract_frames(video_path: Path, frames_dir: Path, max_frames: int) -> tuple[int, float]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old in frames_dir.glob("frame_*.png"):
        old.unlink()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    idx = 0
    while True:
        if max_frames > 0 and idx >= max_frames:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        out = frames_dir / f"frame_{idx:06d}.png"
        cv2.imwrite(str(out), frame)
        idx += 1

    cap.release()
    if idx == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    return idx, fps


def write_rows_csv(rows: list[dict[str, Any]], out_csv: Path) -> None:
    if not rows:
        out_csv.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_frame_entries(frames_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for i, p in enumerate(sorted(frames_dir.glob("frame_*.png"))):
        stem = p.stem
        tail = stem.split("_")[-1]
        frame_idx = int(tail) if tail.isdigit() else i
        entries.append(
            {
                "frame_index": frame_idx,
                "frame_id": stem,
                "frame_path": p,
            }
        )
    return entries


def run_detection_tracking(
    args: argparse.Namespace,
    frame_entries: list[dict[str, Any]],
    tracks_csv_out: Path,
    track_summary_out: Path,
) -> tuple[Path, Path]:
    t0 = time.time()
    yolo = YOLO(str(args.yolo_weights))

    rows: list[dict[str, Any]] = []
    by_track: dict[int, dict[str, Any]] = defaultdict(
        lambda: {
            "detections": 0,
            "first_frame_index": None,
            "last_frame_index": None,
            "mean_conf_sum": 0.0,
        }
    )

    for k, entry in enumerate(frame_entries):
        frame_index = int(entry["frame_index"])
        frame_id = str(entry["frame_id"])
        frame_path: Path = entry["frame_path"]

        frame = cv2.imread(str(frame_path))
        if frame is None:
            print(f"[WARN] Could not read frame: {frame_path}")
            continue

        yolo_result = yolo.track(
            source=frame,
            persist=True,
            tracker=args.tracker,
            classes=[args.person_class],
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        boxes = yolo_result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confs = boxes.conf.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        if boxes.id is not None:
            track_ids = boxes.id.detach().cpu().numpy().astype(int)
        else:
            track_ids = np.full(len(xyxy), -1, dtype=int)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = [float(v) for v in box]
            det_conf = float(confs[i])
            det_cls = int(classes[i])
            track_id = int(track_ids[i])

            row = {
                "frame_index": frame_index,
                "frame_id": frame_id,
                "image_path": str(frame_path),
                "track_id": track_id,
                "det_class_id": det_cls,
                "det_conf": det_conf,
                "box_x1": x1,
                "box_y1": y1,
                "box_x2": x2,
                "box_y2": y2,
                "box_w": x2 - x1,
                "box_h": y2 - y1,
                "bbox_bottom_center_x": 0.5 * (x1 + x2),
                "bbox_bottom_center_y": y2,
            }
            rows.append(row)

            if track_id >= 0:
                s = by_track[track_id]
                s["detections"] += 1
                s["mean_conf_sum"] += det_conf
                if s["first_frame_index"] is None:
                    s["first_frame_index"] = frame_index
                s["last_frame_index"] = frame_index

        if (k + 1) % 50 == 0:
            print(f"Tracking processed {k + 1}/{len(frame_entries)} frames...")

    write_rows_csv(rows, tracks_csv_out)

    summary_rows: list[dict[str, Any]] = []
    for track_id in sorted(by_track):
        s = by_track[track_id]
        det_n = int(s["detections"])
        summary_rows.append(
            {
                "track_id": track_id,
                "detections": det_n,
                "first_frame_index": s["first_frame_index"],
                "last_frame_index": s["last_frame_index"],
                "mean_conf": (s["mean_conf_sum"] / det_n) if det_n else float("nan"),
            }
        )
    write_rows_csv(summary_rows, track_summary_out)
    print(
        f"Detection/tracking complete in {time.time() - t0:.1f}s "
        f"({len(rows)} rows, {len(by_track)} tracks)."
    )
    return tracks_csv_out, track_summary_out


def run_ground_contact(
    args: argparse.Namespace,
    frames_dir: Path,
    tracks_dir: Path,
    fps_value: float,
    tracks_csv: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(args.ground_contact_script),
        "--tracks-csv",
        str(tracks_csv),
        "--frames-dir",
        str(frames_dir),
        "--max-frames",
        str(args.max_frames),
        "--sam-weights",
        str(args.sam_weights),
        "--contact-bottom-frac",
        str(args.contact_bottom_frac),
        "--contact-mode",
        str(args.contact_mode),
        "--contact-smooth-method",
        str(args.contact_smooth_method),
        "--contact-smooth-window",
        str(args.contact_smooth_window),
        "--device",
        str(args.device),
        "--fps",
        str(fps_value),
        "--output-dir",
        str(tracks_dir),
    ]
    if args.no_video:
        cmd.append("--no-video")
    subprocess.run(cmd, check=True)
    return cmd


def main() -> None:
    args = parse_args()

    video_path = args.video.resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not args.ground_contact_script.exists():
        raise FileNotFoundError(f"ground-contact script not found: {args.ground_contact_script}")
    if not args.yolo_weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {args.yolo_weights}")
    sam_weights_raw = str(args.sam_weights).strip()
    if not sam_weights_raw:
        raise ValueError("--sam-weights cannot be empty.")
    sam_path_hint = Path(sam_weights_raw)
    if ("/" in sam_weights_raw or "\\" in sam_weights_raw) and (not sam_path_hint.exists()):
        raise FileNotFoundError(f"SAM weights path not found: {sam_weights_raw}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = args.output_dir / "frames"
    tracks_dir = args.output_dir / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)

    frame_count, source_fps = extract_frames(video_path=video_path, frames_dir=frames_dir, max_frames=args.max_frames)
    fps_value = args.fps if args.fps > 0 else (source_fps if source_fps > 0 else 30.0)

    frame_entries = build_frame_entries(frames_dir=frames_dir)
    raw_tracks_csv = tracks_dir / ".tracks_detect_track.tmp.csv"
    raw_track_summary = tracks_dir / ".track_summary_detect_track.tmp.csv"
    raw_tracks_csv, raw_track_summary = run_detection_tracking(
        args=args,
        frame_entries=frame_entries,
        tracks_csv_out=raw_tracks_csv,
        track_summary_out=raw_track_summary,
    )
    cmd = run_ground_contact(
        args=args,
        frames_dir=frames_dir,
        tracks_dir=tracks_dir,
        fps_value=fps_value,
        tracks_csv=raw_tracks_csv,
    )

    out_csv = tracks_dir / "tracks_with_sam_contact.csv"
    out_jsonl = tracks_dir / "tracks_with_sam_contact.jsonl"
    out_meta = tracks_dir / "run_meta.json"
    out_summary = tracks_dir / "track_summary.csv"

    if not out_csv.exists():
        raise RuntimeError(f"Player pipeline did not produce expected CSV: {out_csv}")

    # Keep stage outputs stable while using temporary intermediate files.
    for tmp_path in (raw_tracks_csv, raw_track_summary):
        if tmp_path.exists():
            tmp_path.unlink()

    meta = {
        "video_path": str(video_path),
        "source_video_fps": source_fps,
        "pipeline_fps": fps_value,
        "frames_dir": str(frames_dir),
        "frames_extracted": frame_count,
        "detect_script": str(args.ground_contact_script),
        "detect_command": cmd,
        "outputs": {
            "tracks_csv": str(out_csv),
            "tracks_jsonl": str(out_jsonl),
            "track_summary_csv": str(out_summary),
            "detect_run_meta": str(out_meta),
        },
    }

    stage_meta = args.output_dir / "run_meta_players.json"
    stage_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Player detection/tracking complete.")
    print(f"Frames dir: {frames_dir}")
    print(f"Tracks:     {out_csv}")
    print(f"Meta:       {stage_meta}")


if __name__ == "__main__":
    main()
