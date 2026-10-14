# ImageHook

Light gameplay frame recorder for building person-detection datasets for Arena Breakout Infinite (`UAGame.exe`).
It captures the monitor via DXGI while you play and saves only frames worth labeling. It skips interface screens
(inventory, map, loading, death and extraction screens), near-duplicates and long still scenes, and records more
often when the scene changes or a person is detected. It can also record short native-resolution video clips, so
that models can be compared on the same real footage (`screendet run --source clip.mp4`).

It is one part of a three-project system:

| Project | Role |
|---|---|
| **ImageHook** (this repository) | records gameplay frames into session folders |
| **ImageClassfication** | labeling app: turns sessions into a YOLO dataset |
| **fin_snip_helper** (`screendet`) | real-time detection, training, model comparison |

ImageHook grew out of `fin_snip_helper/tools/collect_frames.py` and keeps its output format, so sessions can be
labeled by either labeling tool.

**Contents:** [Quick start](#quick-start) · [Keys](#keys) · [Settings](#settings) ·
[What the console shows](#what-the-console-shows) · [Looking at sessions](#looking-at-sessions-imagehook-info) ·
[Output format](#output-format) · [How frames are chosen](#how-frames-are-chosen) ·
[Optional detector](#optional-detector) · [Video clips](#video-clips) · [Resource use](#resource-use) ·
[Checking the filters offline](#checking-the-filters-offline-imagehook-replay) · [Tests](#tests) · [Limits](#limits)

## Quick start

Windows 10/11 with an NVIDIA GPU, Python 3.11+.

```
python -m venv .venv --system-site-packages
.venv\Scripts\python -m pip install -r requirements.txt -r requirements-dev.txt
.venv\Scripts\python -m imagehook record
```

Start the game in windowed or borderless mode, press `F8` to start recording and `F9` to quit. When a session is
closed, a summary is printed; `imagehook info` shows it again later. All commands:

| Command | What it does |
|---|---|
| `imagehook record` | records gameplay frames (and clips with `F7`) |
| `imagehook info [SESSION]` | a table of all sessions, or the details of one |
| `imagehook replay SESSION` | runs a recorded session or a video through the filters, without the game |
| `imagehook clip` | records one video clip |
| `imagehook template SESSION...` | rebuilds the HUD icon template after HUD changes |
| `imagehook config` | shows the settings in effect; `--init` creates `imagehook.toml` |

`imagehook` stands for `.venv\Scripts\python -m imagehook`. Every command has `--help`, which shows the defaults
already including the settings file.

Optional parts:

* the detector (`--detector`) needs `requirements-detector.txt` and a CUDA build of PyTorch (RTX 50xx needs
  CUDA 12.8+), installed globally or in the venv. `--system-site-packages` lets the venv reuse a global install;
* video clips need ffmpeg 6 or newer with the `ddagrab` filter and the `h264_nvenc` encoder (for example a "gpl"
  Windows build), on `PATH`, in `IMAGEHOOK_FFMPEG`, or passed with `--ffmpeg`. Without ffmpeg, frame recording
  still works.

## Keys

| Key | Action |
|---|---|
| `F8` | start / stop recording (a session folder is created on the first start) |
| `Mouse 5` (front side button) | **burst**: keep the unsaved frames of the last 10 s and every new frame for the next 5 s |
| `F7` | start / stop a video clip (stops by itself after 120 s); works with or without frame recording |
| `F9` | quit |

Press burst whenever you have just seen people, including after the fight. The last 10 s are kept in memory, so you
do not need to react at the right moment. Keys are read globally and still reach the game, so choose ones that the
game does not use. Key names: `f1`..`f24`, `a`..`z`, `0`..`9`, `numpad0`..`numpad9`, `mouse3`/`mouse4`/`mouse5`,
`space`, `tab`, `insert`, `home`, ... or a virtual-key code like `0x2D`.

Beeps confirm the keys: high = started, middle = stopped, short high = burst, two short = clip started/saved, three
low = recording stopped on its own because of a limit (or could not start).

Frames are taken only while the game window is in the foreground (`any_window` turns this off). Recording stops
when a session reaches 20 GB (`max_session_gb`; the next start opens a new session) or the disk has less than
20 GB free (`min_free_gb`).

## Settings

Every option can be set on the command line or in a settings file, `imagehook.toml`. Precedence:
**built-in defaults < `imagehook.toml` < command line.**

```
.venv\Scripts\python -m imagehook config --init      # creates ./imagehook.toml with every option commented out
.venv\Scripts\python -m imagehook config             # prints the settings in effect, marks the ones from the file
```

The file is looked up in this order: `--config PATH`, `$IMAGEHOOK_CONFIG`, `./imagehook.toml`, then the project
folder. Keys are the long option names (`ui_threshold` = `--ui-threshold`); the sections only group them. Relative
paths are relative to the file. Unknown keys and wrong types stop the program with a message that names the key
(`ui_treshold: unknown option; did you mean 'ui_threshold'?`). `imagehook.toml` is personal and not committed;
[`imagehook.example.toml`](imagehook.example.toml) documents every option with its default.

Example:

```toml
[output]
out = "D:/captures"

[game]
game_processes = ["uagame.exe"]

[keys]
burst_key = "mouse4"

[filters]
ui_threshold = 0.25
```

| Section | Options |
|---|---|
| `[output]` | `out`, `output` (monitor), `height`, `format`, `quality`, `max_session_gb`, `min_free_gb` |
| `[game]` | `game_processes`, `any_window`, `no_ui_filter`, `template`, HUD layout `icon_zone` and `change_ignore` |
| `[keys]` | `toggle_key`, `burst_key`, `clip_key`, `quit_key`, `burst_back`, `burst_forward`, `no_beep` |
| `[filters]` | `ui_threshold`, `dup_novelty`, `min_interval`, `base_interval`, `max_interval`, `global_ref`, `local_ref` |
| `[detector]` | `detector`, `det_imgsz`, `det_conf`, `det_interval`, `det_hold`, `idle_interval`, `det_max_height` |
| `[clips]` | `ffmpeg`, `clip_seconds`, `clip_fps`, `clip_cq`, `delay` |
| `[limits]` | `target_ram_gb`, `max_ram_gb`, `target_vram_gb`, `max_vram_gb`, `target_gpu_pct`, `max_gpu_pct` |
| `[run]` | `autostart`, `duration` |

The **HUD layout** describes where the game's HUD is, as normalized `[x1, y1, x2, y2]` boxes of a 16:9 frame:
`icon_zone` is the body-status icon in the bottom-left corner (only gameplay frames have it), `change_ignore` are
the areas that change on their own (FPS/ping overlay, chat, weapon panel, stamina bar). Change them if the HUD
layout or scale changes, then rebuild the template with `imagehook template`. The settings file of `record` also
applies to `replay` and `template`, so offline checks use the same filters; only `record` uses `detector`.

## What the console shows

At startup `record` prints the settings in effect:

```
ImageHook 0.2.0, settings: imagehook.toml + command line (see them all: imagehook config)
  output     D:\captures  (412 GB free)
  monitor    output 0, 2560x1440; frames saved at native resolution, JPG q95
  game       only while uagame.exe is the foreground window
  filters    interface screens skipped below HUD icon similarity 0.3 (bundled template); duplicates below novelty 0.15
  intervals  a check every 0.5 s; saved after 0.5 s for a big change, 1 s for an average one, 10 s for a still view
  detector   off (--detector weights.pt saves more often while people are seen)
  clips      F7: 30 FPS H.264, cq 19, stops after 120 s (ffmpeg.exe)
  limits     RAM 2/4 GB, VRAM 2/4 GB, own GPU 20/30% (target/hard); session 20 GB; stop below 20 GB free
  keys       F8 start/stop | MOUSE5 burst (last 10 s + next 5 s) | F7 clip | F9 quit
```

While recording, a status line is printed every minute:

```
REC 1:02:05 | saved 2,512 (+41/min): change 2,330, burst 181, first 1 | skipped of 7,450 checks: ui 34.7%, duplicate 2.5%, waiting 27.6% | outside the game 0:01:20 | 1.82 GB | RAM 0.31 GB, VRAM 0.00 GB, own GPU 0%
```

`REC` is the recorded time (pauses with `F8` excluded), `+N/min` the save rate since the previous line, the skipped
shares are of all checked frames, "outside the game" is the time another window was in front. Notes are printed
when a resource goes above its target, and the detector state changes are logged.

## Looking at sessions (`imagehook info`)

```
.venv\Scripts\python -m imagehook info                      # all sessions in the output folder
.venv\Scripts\python -m imagehook info path\to\session      # one session in detail
.venv\Scripts\python -m imagehook info path\to\session --legend   # ... and what each reason means
```

```
3 session(s) in D:\captures:
  session                  recorded  frames    GB  interface          labels
  session_20261008_044703         -   2,260     -          -  2,260 reviewed
  session_20261011_220514   2:13:18   5,507  3.82      34.7%  4,022 reviewed

session_20261011_220514   (imagehook 0.1.0)
  time       started 2026-10-11 22:05:14, ended 2026-10-12 01:04:00, recorded 2:13:18
  saved      5,507 frames, 3.82 GB (0.71 MB per frame), 41 per recorded minute
             first 1 (0.0%), change 5,123 (93.0%), burst 383 (7.0%)
  checked    15,635 frames (one every 0.5 s while recording in the game)
  skipped    ui 5,421 (34.7%), duplicate 385 (2.5%), waiting 4,322 (27.6%)
             not checked: not_game 160, no_new_frame 95
  gaps       between saved frames: median 1.01 s, 90% within 1.51 s, longest 44.23 s; 14 pause(s) over 60 s
  resources  RAM peak 0.86 GB, CPU avg 0.15% (peak 3.64%), VRAM peak 0.00 GB, own GPU avg 0.2% (peak 12%)
  detector   not used
  labels     pos 736 (13.4%), neg 2,682 (48.7%), skip 287 (5.2%), unlabeled 1,485 (27.0%), todo 317 (5.8%)
```

The same summary is printed when a session is closed. `labels` comes from the labeling app's `review.json`.
Sessions of the old collector (no `capture.json`) show only frames, gaps and labels. `--json` prints the numbers
for scripts.

## Output format

```
<out>/session_YYYYmmdd_HHMMSS/
    frames/HHMMSS_mmm.jpg     time of day with milliseconds; a session may cross midnight
    capture_log.csv           one row per saved frame: name, reason, novelty, ui score, detector confidence
    capture.json              settings, saved/skipped counts by reason, recorded seconds, resource peaks
```

Frames are saved at the native monitor resolution (2560x1440 on a 1440p monitor) as JPEG quality 95, about
0.7–1.5 MB per gameplay frame (0.71 MB on average in a real 2 h session). `height = 1080` scales them down like the
old collector did. The output root is `./captures` by default (`out` or the `IMAGEHOOK_OUT` environment variable).

Video clips go to `<out>/clips/clip_YYYYmmdd_HHMMSS.mp4`: H.264, native resolution, 30 FPS.

## How frames are chosen

The recorder checks a frame every 0.5 s. Each check computes two small features in a few milliseconds:

* **Interface check**: an interface screen has no body-status silhouette icon in the bottom-left HUD corner. That
  area (`icon_zone`) is compared with a template (`imagehook/assets/hud_icon.png`, the median icon of about 9,400
  recorded frames). If the similarity is below 0.3 (`ui_threshold`), the frame is skipped (`ui`).
* **Novelty**: a 320x180 grayscale thumbnail is compared with the last *saved* frame. HUD areas that change by
  themselves (`change_ignore`) are ignored.
  `novelty = max(mean difference / 15, largest share of changed pixels in a 20x20 cell / 0.6)` (`global_ref`,
  `local_ref`). The local term catches something moving in an otherwise still view.

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

Frames that wait (`waiting`) are kept in memory for 10 s, so a burst can still save them.

In the first live session (2 h 13 min, 2026-10-11) 35% of the checked frames were interface screens, 28% waited,
2.5% were duplicates; the median gap between saved frames was 1.0 s and 90% of the gaps were within 1.5 s.

## Optional detector

`--detector weights.pt` (or `detector` in the settings) loads a YOLO model with PyTorch (never TensorRT) at
startup. It runs on each checked frame that is not an interface screen, at native resolution (`imgsz` 2560 at
1440p, like `screendet`; `det_imgsz`) with FP16. If the tool's own GPU load goes above 20%, the detector runs only
on every 2nd, 4th or 8th checked frame. If the load still stays above 30%, the detector is turned off. The detector
**never drops frames**: a person with confidence ≥ 0.2 (`det_conf`) makes the recorder save more often, and while
it sees nobody the base interval becomes 2 s (`idle_interval`). Boxes taller than 15% of the frame
(`det_max_height`) are ignored, because close targets matter least. Detections on the HUD body icon are ignored too.

## Video clips

Press `F7` during `record`, or run it standalone:

```
.venv\Scripts\python -m imagehook clip --clip-seconds 60 --delay 5     # 5 s to switch to the game
```

ffmpeg captures and encodes on the GPU: `ddagrab` (DXGI duplication into D3D11 memory) feeds `h264_nvenc`, the
dedicated video encoder. No frame passes through Python, and the CPU load stays negligible. Clips are H.264 Main
4:2:0, which OpenCV reads directly. Pass a clip to `screendet run --source clip.mp4`, or check the filters on it with
`imagehook replay clip.mp4`. Clip size depends on the scene: about 6 Mbit/s for a mostly still desktop, noticeably
more for gameplay at the default quality (`clip_cq = 19`). Plan for a few hundred MB per minute.

## Resource use

The game crashed once when other software used too much of the machine, so the recorder is kept light: no torch in
the default mode, below-normal CPU priority (the ffmpeg child too), single-threaded OpenCV, and a bounded writer
queue.

| Resource | Target (a note is printed above it) | Hard limit |
|---|---|---|
| RAM (tool + ffmpeg) | 2 GB (`target_ram_gb`) | 4 GB (`max_ram_gb`): detector off, then recording stops |
| VRAM | 2 GB (`target_vram_gb`) | 4 GB (`max_vram_gb`): detector off |
| Own GPU load | 20% (`target_gpu_pct`): the detector is throttled | 30% sustained (`max_gpu_pct`): detector off |

The tool's own GPU load comes from NVML per-process samples (`nvmlDeviceGetProcessUtilization` for its own pid and
the ffmpeg child), because the game shares the device. The status line and `capture.json` report it.

Measured on Windows 11, RTX 5070 Ti, 2560x1440:

| Mode | RAM (RSS) | VRAM | Own GPU (SM) | CPU (share of the whole CPU) |
|---|---|---|---|---|
| default, live 2 h session with the game | 0.86 GB peak | 0 | 0.2% avg, 12% peak | 0.15% avg, 3.6% peak |
| default, synthetic worst case (a save every 0.5 s) | 0.12 GB | 0 | 0 | 0.4% avg, 4.7% peak |
| `--detector` (YOLO26s, native 2560)\* | 1.92 GB | about 1.0 GB (PyTorch reserve + estimated 0.5 GB CUDA context) | 2.1% avg, 14% peak; 62 ms per frame, 2 frames/s | 1.3% avg, 7% peak |
| video clip (ffmpeg ddagrab + NVENC, 1440p30) | 0.32 GB including Python | 0 | 0.4% avg, 6% peak | 0.25% avg |

\* measured while another process kept the GPU about 70% busy, so the detector times are pessimistic.

The burst buffer holds up to 20 full frames (about 220 MB at 1440p) on top of these numbers.

## Checking the filters offline (`imagehook replay`)

`replay` runs recorded frames or a video through the same filters without the game:

```
.venv\Scripts\python -m imagehook replay path\to\session_YYYYmmdd_HHMMSS
.venv\Scripts\python -m imagehook replay path\to\session --det-labels         # emulate the detector with labels_auto/
.venv\Scripts\python -m imagehook replay path\to\session --ui-threshold 0.25  # try another threshold
.venv\Scripts\python -m imagehook replay path\to\clip.mp4 --tick 0.5
.venv\Scripts\python -m imagehook replay path\to\session --write thinned      # copy the kept frames to a new session
```

```
session_20261011_220514: 5,507 frames checked, 3,473 kept (63.1%)
  reasons    waiting 2,027 (36.8%), detection 1,857 (33.7%), change 1,615 (29.3%), duplicate 7 (0.1%), first 1 (0.0%)
  labels     neg 1,476 of 2,682 kept (55.0%) | pos 518 of 736 kept (70.4%) | skip 192 of 287 kept (66.9%) | ...
  people     frames with people that have a kept frame with people within 2 s: 98.8%; small people (under 40 px at 1080p, 217 frames): 99.1%
```

If the session is labeled (`review.json` with `pos`/`neg`/`skip`, plus `labels/`), the report shows the kept frames
per status. It also checks that every frame with people still has a kept frame with people within ±2 s
(`small` counts only people under 40 px at 1080p). `--json` prints the report as JSON.

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

The tests use synthetic frames: interface and duplicate decisions, the novelty rules, the HUD layout, the detector
and burst paths, the GPU-load governor, clip command and names, session naming (including midnight), replay, the
settings file (precedence, errors, the example file listing every option) and the reports. They need neither the
game nor a GPU. `IMAGEHOOK_TEST_CLIP=1` adds a 2 s real screen clip (needs ffmpeg and NVENC).

## Limits

* DXGI capture needs windowed or borderless mode; exclusive fullscreen may return black frames.
* The interface check assumes a 16:9 frame and the HUD layout in the settings (the default HUD by default). Use
  `template` after HUD changes and `no_ui_filter` to switch the check off.
* Novelty is measured on a 320x180 thumbnail. A far person who barely moves in a still view changes only a few
  pixels, so such a view is saved by the 10 s `periodic` rule, by the detector, or by the burst key.
