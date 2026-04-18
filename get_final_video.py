#!/usr/bin/env python3
"""
Combine velocity overlay and bird's-eye contact videos side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render velocity and bird's-eye videos side by side."
    )
    parser.add_argument(
        "--velocity-video",
        "--left-video",
        dest="left_video",
        type=Path,
        required=True,
        help="Input velocity overlay video (e.g., tracks_velocity_overlay.mp4).",
    )
    parser.add_argument(
        "--birdseye-video",
        "--right-video",
        dest="right_video",
        type=Path,
        required=True,
        help="Input bird's-eye contact video (e.g., birdseye_contacts_velocity.mp4).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for final combined video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=0.0,
        help="Output FPS (0 uses input video FPS when available).",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="velocity_birdseye_side_by_side.mp4",
        help="Output video filename.",
    )
    parser.add_argument(
        "--gap-px",
        type=int,
        default=8,
        help="Gap in pixels between left and right videos.",
    )
    parser.add_argument(
        "--title-height-px",
        type=int,
        default=36,
        help="Top title bar height in pixels.",
    )
    parser.add_argument(
        "--left-title",
        type=str,
        default="Velocity Overlay",
        help="Title rendered above left video.",
    )
    parser.add_argument(
        "--right-title",
        type=str,
        default="Bird's-eye Contact",
        help="Title rendered above right video.",
    )
    parser.add_argument(
        "--no-titles",
        action="store_true",
        help="Disable pane titles.",
    )
    return parser.parse_args()


def open_video(path: Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    return cap


def resize_keep_aspect(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(float(target_w) / float(w), float(target_h) / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def draw_title_bar(
    frame: np.ndarray,
    left_w: int,
    right_w: int,
    gap_px: int,
    bar_h: int,
    left_title: str,
    right_title: str,
) -> None:
    cv2.rectangle(frame, (0, 0), (frame.shape[1] - 1, bar_h - 1), (20, 20, 20), -1)
    cv2.putText(
        frame,
        left_title,
        (12, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    right_x = left_w + gap_px + 12
    cv2.putText(
        frame,
        right_title,
        (right_x, int(bar_h * 0.7)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (235, 235, 235),
        2,
        cv2.LINE_AA,
    )
    mid_x = left_w + gap_px // 2
    cv2.line(frame, (mid_x, 0), (mid_x, frame.shape[0] - 1), (55, 55, 55), 1, cv2.LINE_AA)


def main() -> None:
    args = parse_args()

    left_path = args.left_video.resolve()
    right_path = args.right_video.resolve()
    if not left_path.exists():
        raise FileNotFoundError(f"Velocity video not found: {left_path}")
    if not right_path.exists():
        raise FileNotFoundError(f"Bird's-eye video not found: {right_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_video = args.output_dir / args.output_name
    out_meta = args.output_dir / "run_meta_final_video.json"

    left_cap = open_video(left_path)
    right_cap = open_video(right_path)

    left_w = int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    left_h = int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    right_w = int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    right_h = int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    left_fps = float(left_cap.get(cv2.CAP_PROP_FPS) or 0.0)
    right_fps = float(right_cap.get(cv2.CAP_PROP_FPS) or 0.0)

    if left_w <= 0 or left_h <= 0:
        raise RuntimeError(f"Invalid velocity video dimensions: {left_w}x{left_h}")
    if right_w <= 0 or right_h <= 0:
        raise RuntimeError(f"Invalid bird's-eye video dimensions: {right_w}x{right_h}")

    fps_out = (
        float(args.fps)
        if args.fps > 0
        else (left_fps if left_fps > 0 else (right_fps if right_fps > 0 else 30.0))
    )

    pane_h = max(left_h, right_h)
    left_pane_w = max(1, int(round(left_w * (float(pane_h) / float(left_h)))))
    right_pane_w = max(1, int(round(right_w * (float(pane_h) / float(right_h)))))
    gap_px = max(0, int(args.gap_px))
    title_h = 0 if args.no_titles else max(0, int(args.title_height_px))

    out_w = left_pane_w + gap_px + right_pane_w
    out_h = pane_h + title_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_video), fourcc, fps_out, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open output video writer: {out_video}")

    frames_written = 0
    try:
        while True:
            ok_left, frame_left = left_cap.read()
            ok_right, frame_right = right_cap.read()

            if not ok_left and not ok_right:
                break

            if not ok_left or frame_left is None:
                frame_left = np.zeros((left_h, left_w, 3), dtype=np.uint8)
            if not ok_right or frame_right is None:
                frame_right = np.zeros((right_h, right_w, 3), dtype=np.uint8)

            left_canvas = resize_keep_aspect(frame_left, target_h=pane_h, target_w=left_pane_w)
            right_canvas = resize_keep_aspect(frame_right, target_h=pane_h, target_w=right_pane_w)

            frame_out = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            y0 = title_h
            frame_out[y0 : y0 + pane_h, 0:left_pane_w] = left_canvas
            x_right = left_pane_w + gap_px
            frame_out[y0 : y0 + pane_h, x_right : x_right + right_pane_w] = right_canvas

            if title_h > 0:
                draw_title_bar(
                    frame=frame_out,
                    left_w=left_pane_w,
                    right_w=right_pane_w,
                    gap_px=gap_px,
                    bar_h=title_h,
                    left_title=args.left_title,
                    right_title=args.right_title,
                )

            writer.write(frame_out)
            frames_written += 1
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()

    meta = {
        "left_video": str(left_path),
        "right_video": str(right_path),
        "output_video": str(out_video),
        "output_fps": fps_out,
        "output_width": out_w,
        "output_height": out_h,
        "frames_written": frames_written,
        "titles_enabled": not args.no_titles,
    }
    out_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Final side-by-side video complete.")
    print(f"Frames: {frames_written}")
    print(f"Video:  {out_video}")
    print(f"Meta:   {out_meta}")


if __name__ == "__main__":
    main()
