# ImageHook

Light gameplay frame recorder for building person-detection datasets for Arena Breakout Infinite (`UAGame.exe`).
It captures the monitor via DXGI while you play and saves only frames worth labeling. It skips interface screens
(inventory, map, loading, death and extraction screens), near-duplicates and long still scenes, and records more
often when the scene changes or a person is detected.

It is one part of a three-project system:

| Project | Role |
|---|---|
| **ImageHook** (this repository) | records gameplay frames into session folders |
| **ImageClassfication** | labeling app: turns sessions into a YOLO dataset |
| **fin_snip_helper** (`screendet`) | real-time detection, training, model comparison |

ImageHook grew out of `fin_snip_helper/tools/collect_frames.py` and keeps its output format, so sessions can be
labeled by either labeling tool.

## Output format

```
<out>/session_YYYYmmdd_HHMMSS/
    frames/HHMMSS_mmm.jpg     time of day with milliseconds; a session may cross midnight
    capture_log.csv           one row per saved frame: name, reason, novelty, ui score, detector confidence
    capture.json              settings, saved/skipped counts by reason, recorded seconds, resource peaks
```

Frames are saved at the native monitor resolution (2560x1440 on a 1440p monitor) as JPEG quality 95, about
1–1.5 MB per gameplay frame. `--height 1080` scales them down like the old collector did. The output root is
`./captures` by default; set it with `--out` or the `IMAGEHOOK_OUT` environment variable.

## Setup

Windows 10/11, Python 3.11+.

```
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
```

The optional detector (`--detector`) also needs `requirements-detector.txt` and a CUDA build of PyTorch
(RTX 50xx needs CUDA 12.8+). Install PyTorch separately, globally or in the venv. `--system-site-packages` lets the
venv reuse a global install.

## Recording

```
.venv\Scripts\python -m imagehook record
.venv\Scripts\python -m imagehook record --detector path\to\weights.pt     # optional, see below
```

| Key | Action |
|---|---|
| `F8` | start / stop recording (a session folder is created on the first start) |
| `Mouse 5` (front side button) | **burst**: keep the unsaved frames of the last 10 s and every new frame for the next 5 s |
| `F9` | quit |

Press burst whenever you have just seen people, including after the fight. The last 10 s are kept in memory, so you
do not need to react at the right moment. Keys are configurable (`--toggle-key`, `--burst-key`, `--quit-key`; names
like `f7`, `mouse4`, `x`, `numpad1`, `0x2D`). Keys are read globally and still reach the game, so choose one that the
game does not use. Short beeps confirm the keys. Three low beeps mean that recording stopped on its own because of a
limit.

Frames are taken only while the game window is in the foreground (`--any-window` turns this off). Recording stops
when a session reaches 20 GB (`--max-session-gb`; the next start opens a new session) or the disk has less than
20 GB free (`--min-free-gb`). A status line with the counts is printed every minute.

## How frames are chosen

The recorder evaluates a frame every 0.5 s. Each evaluation computes two small features in a few milliseconds:

* **Interface check**: an interface screen has no body-status silhouette icon in the bottom-left HUD corner. That
  area is compared with a template (`imagehook/assets/hud_icon.png`, the median icon of about 9,400 recorded
  frames). If the similarity is below 0.3 (`--ui-threshold`), the frame is skipped (`ui`).
* **Novelty**: a 320x180 grayscale thumbnail is compared with the last *saved* frame. HUD areas that change by
  themselves (timer, FPS overlay, chat, weapon panel, stamina bar) are ignored.
  `novelty = max(mean difference / 15, largest share of changed pixels in a 20x20 cell / 0.6)`. The local term catches
  something moving in an otherwise still view.

Save rules, in this order:

1. Interface screen: skip (`ui`).
2. Novelty below 0.15: skip as a duplicate (`duplicate`). Duplicates are never saved, however long you wait.
3. Burst active: save (`burst`).
4. Less than 0.5 s since the last saved frame: wait.
5. Detector hit (only with `--detector`): save at most once per second while a person is seen, and for 2 s after
   (`detection`).
6. `novelty × seconds since the last save ≥ base interval` (1 s, or 2 s while the detector sees nobody): save
   (`change`). A camera turn is saved after 0.5 s, an average change after about 1 s, and a small change only after
   several seconds.
7. 10 s since the last save: save (`periodic`). This covers almost static views, for example holding an angle.

All thresholds are command line options (`python -m imagehook record --help`).

### Optional detector

`--detector weights.pt` loads a YOLO model with PyTorch (never TensorRT) at startup. It runs on each evaluated
frame that is not an interface screen, at `imgsz` 1280 with FP16. The detector **never drops frames**: a person
with confidence ≥ 0.2 (`--det-conf`) makes the recorder save more often, and while it sees nobody the base interval
becomes 2 s (`--idle-interval`). Boxes taller than 15% of the frame (`--det-max-height`) are ignored, because close
targets matter least. Detections on the HUD body icon are ignored too.

## Resource use

The game crashed once when other software used too much of the machine, so the recorder is kept light: no torch in
the default mode, below-normal CPU priority, single-threaded OpenCV, and a bounded writer queue. Hard limits:
4 GB RAM (`--max-ram-gb`) and 2 GB VRAM (`--max-vram-gb`). Above a limit the detector is turned off first, then
recording stops.

Measured on Windows 11, RTX 5070 Ti, 2560x1440, worst case of saving every 0.5 s:

| Mode | RAM (RSS) | VRAM | CPU (share of the whole CPU) |
|---|---|---|---|
| default | 0.12 GB | 0 | 0.4% avg, 4.7% peak |
| `--detector` (YOLO26s, 1280) | 1.96 GB | about 0.65 GB (PyTorch reserve 0.15 GB + CUDA context) | 0.8% avg, 6.5% peak; about 24 ms GPU per frame, 2 frames/s |

The burst buffer holds up to 20 full frames (about 220 MB at 1440p) on top of these numbers.

## Checking the filters offline

`replay` runs recorded frames or a video through the same filters without the game:

```
.venv\Scripts\python -m imagehook replay path\to\session_YYYYmmdd_HHMMSS
.venv\Scripts\python -m imagehook replay path\to\session --det-labels         # emulate the detector with labels_auto/
.venv\Scripts\python -m imagehook replay path\to\clip.mp4 --tick 0.5
.venv\Scripts\python -m imagehook replay path\to\session --write thinned      # copy the kept frames to a new session
```

If the session is labeled (`review.json` with `pos`/`neg`/`skip`, plus `labels/`), the report shows the kept frames
per status. It also checks that every frame with people still has a kept frame with people within ±2 s
(`pos_covered`; `small_pos_covered` counts only people under 40 px at 1080p).

Results on two sessions recorded with the old collector (9,400 frames at 1 s intervals, already without static
frames). The 0.5 s rule cannot be replayed on 1 s data:

| Mode | Kept | People frames kept | Interface/unusable (`skip`) kept | People covered (±2 s) | Small people covered |
|---|---|---|---|---|---|
| default | 72% | 90% | 26% | 99.7–100% | 100% |
| detector (emulated from `labels_auto`) | 61% | 84% | 22% | 98.8–100% | 99.3–100% |

Of the 1,895 frames rejected as interface screens, the labeled ones with people are menu, loading and lobby screens
with character models.

`template` rebuilds the HUD icon template from sessions that are mostly gameplay. Use it if the HUD scale or layout
changes:

```
.venv\Scripts\python -m imagehook template path\to\session1 path\to\session2
```

## Tests

```
.venv\Scripts\python -m pytest
```

The tests use synthetic frames: interface and duplicate decisions, the novelty rules, the detector and burst paths,
session naming (including midnight), and replay. They need neither the game nor a GPU.

## Limits

* DXGI capture needs windowed or borderless mode; exclusive fullscreen may return black frames.
* The interface check assumes the default HUD layout at 16:9. Use `template` after HUD changes and
  `--no-ui-filter` to switch the check off.
* Novelty is measured on a 320x180 thumbnail. A far person who barely moves in a still view changes only a few
  pixels, so such a view is saved by the 10 s `periodic` rule, by the detector, or by the burst key.
