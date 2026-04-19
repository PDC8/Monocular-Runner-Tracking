# Thesis_Final Pipeline

Run these commands from the project root.

## Model Setup

Model weights are not stored in regular Git history. After cloning the repo, download them from the GitHub release assets:

```bash
./download_models.sh
```

The script places these files under `models/`:

- `models/base_best.pt`
- `models/player_detect_best.pt`
- `models/sam2.1_l.pt`

By default, `download_models.sh` fetches release assets from `PDC8/Monocular_Runner_Tracking` tag `models-v1`. If you publish the assets under a different repo or tag, override the defaults:

```bash
GITHUB_REPO=OWNER/REPO MODEL_RELEASE_TAG=models-v2 ./download_models.sh
```

Each download is verified with a SHA-256 checksum before the file is accepted.

## Quick Start

```bash
# If you are currently one level above the repo:
cd Monocular_Runner_Tracking

./download_models.sh
./venv/bin/python main \
  --video videos/234595382.mp4 \
  --work-dir runs/234595382_main_test_20260415
```

## Default Paths Used By `main`

When you do not pass model-path flags, `main` uses:

- `--base-model`: `models/base_best.pt`
- `--player-yolo-weights`: `models/player_detect_best.pt`
- `--sam-weights`: `models/sam2.1_l.pt`
- `--player-class`: `4` (matches `Player` in `player_detect_best.pt`)

Main output goes under `--work-dir` in fixed stage folders:

- `01_bases/`
- `02_players/`
- `03_infield_cone/`
- `04_velocity/`
- `05_final_video/`

## Contact Point Modes

Use `--contact-mode` to choose how ground contact is defined:

1. `sam`: SAM robust contact `(x, y)`.
2. `bbox_bottom`: detection-box bottom center.
3. `hybrid` (default): box-center `x` + SAM bottom `y`.

Examples:

```bash
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_sam --contact-mode sam
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_bbox_bottom --contact-mode bbox_bottom
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_hybrid --contact-mode hybrid
```

## Optional Flags

```bash
# Disable stage videos for faster test runs
./venv/bin/python main --video videos/234595382.mp4 --work-dir runs/run_no_video --no-video
```
