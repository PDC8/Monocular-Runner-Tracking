# Runner Tracking And Translation To 2D Field Coordinates From Wide Broadcast Video

Run these commands from the project root.

## Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

## Install Dependencies
```bash
pip install -r requirements.txt
```

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

## Running The Pipeline

Minimum command:

```bash
./venv/bin/python main \
  --video path/to/input_video.mp4
```

Common optional overrides:

```bash
./venv/bin/python main \
  --video path/to/input_video.mp4 \
  --work-dir runs/my_run \
  --contact-mode hybrid \
  --no-video
```

## Flags

- Required: `--video`
- Optional: `--work-dir`
  Default: `runs/<video_stem>/`
- Optional: `--contact-mode`
  Default: `hybrid`
  Choices: `sam`, `bbox_bottom`, `hybrid`
- Optional: `--no-video`
  Default: off
  Add this flag to skip stage overlay videos.
- Optional: `--max-frames`
  Default: `0` (`0` means process all frames)
- Optional: `--fps`
  Default: `0` (`0` keeps the source video FPS)
- Optional: `--device`
  Default: auto-detect `CUDA`, then `MPS`, else `CPU`


## Output Layout

If you do not pass `--work-dir`, the pipeline writes to `runs/<video_stem>/`.

Inside the run directory, `main` creates these fixed stage folders:

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
./venv/bin/python main --video path/to/input_video.mp4 --work-dir runs/contact_sam --contact-mode sam
./venv/bin/python main --video path/to/input_video.mp4 --work-dir runs/contact_bbox_bottom --contact-mode bbox_bottom
./venv/bin/python main --video path/to/input_video.mp4 --work-dir runs/contact_hybrid --contact-mode hybrid
```
