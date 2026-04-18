# Thesis_Final Pipeline

Run these commands from the project root (`Thesis_Final`).

## Quick Start

```bash
# If you are currently one level above the repo:
cd Thesis_Final

./venv/bin/python main \
  --video videos/234595382.mp4 \
  --work-dir runs/234595382_main_test_20260415 \
  --contact-mode sam
```

## Default Paths Used By `main`

When you do not pass model-path flags, `main` uses:

- `--base-model`: `models/base_best.pt`
- `--player-yolo-weights`: `models/player_detect_yolov8l_best.pt`
- `--sam-weights`: `models/sam2.1_l.pt`
- `--player-class`: `4` (matches `Player` in `player_detect_yolov8l_best.pt`)

Main output goes under `--work-dir` in fixed stage folders:

- `01_bases/`
- `02_players/`
- `03_infield_cone/`
- `04_velocity/`
- `05_final_video/`

## Contact Point Modes

Use `--contact-mode` to choose how ground contact is defined:

1. `sam` (default): SAM robust contact `(x, y)`.
2. `bbox_bottom`: detection-box bottom center.
3. `hybrid`: box-center `x` + SAM bottom `y`.

Examples:

```bash
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_sam --contact-mode sam
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_bbox_bottom --contact-mode bbox_bottom
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_hybrid --contact-mode hybrid
```

## Optional Flags

```bash
# Disable stage videos for faster test runs
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_no_video --contact-mode sam --no-video
```
