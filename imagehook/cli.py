"""ImageHook: records gameplay frames worth labeling (skips interface screens, duplicates and still scenes).

Commands: record (live capture), info (what a session contains), replay (check the filters offline), clip (one
video clip), template (rebuild the HUD icon template), config (show or create the settings file).
Settings: imagehook.toml (see imagehook.example.toml); the command line overrides it."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from . import config
from .filters import CHANGE_IGNORE, DEFAULT_TEMPLATE, ICON_ZONE, HudLayout, PolicyConfig

PROJECT = config.PROJECT
DEFAULT_OUT = Path(os.environ.get("IMAGEHOOK_OUT", PROJECT / "captures"))


class HelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """Shows the default (the settings file included) unless it is empty or a plain off switch."""

    def _get_help_string(self, action: argparse.Action) -> str | None:
        if action.default is None or action.default is False:
            return action.help
        return super()._get_help_string(action)


def add_policy_args(ap: argparse.ArgumentParser) -> None:
    d = PolicyConfig()
    g = ap.add_argument_group("filters (a frame is saved when novelty x seconds since the last save >= base interval)")
    g.add_argument("--ui-threshold", type=float, default=d.ui_threshold, help="HUD icon similarity below this = interface screen, skipped")
    g.add_argument("--no-ui-filter", action="store_true", help="keep interface screens too")
    g.add_argument("--template", type=Path, help=f"HUD icon template PNG; not set = the bundled {DEFAULT_TEMPLATE.name}")
    g.add_argument("--dup-novelty", type=float, default=d.dup_novelty, help="novelty below this = duplicate of the last saved frame, never saved")
    g.add_argument("--min-interval", type=float, default=d.min_interval, help="s: a frame is checked this often, and no two saved frames are closer")
    g.add_argument("--base-interval", type=float, default=d.base_interval, help="s: an average change (novelty 1) is saved about this often")
    g.add_argument("--max-interval", type=float, default=d.max_interval, help="s: a nearly still view is still saved this often")
    g.add_argument("--global-ref", type=float, default=d.global_ref, help="mean grayscale change (0..255) that counts as novelty 1")
    g.add_argument("--local-ref", type=float, default=d.local_ref, help="share of changed pixels in one thumbnail cell that counts as novelty 1")
    g = ap.add_argument_group("detector decisions")
    g.add_argument("--det-conf", type=float, default=d.det_conf, help="detector confidence that counts as a person")
    g.add_argument("--det-interval", type=float, default=d.det_interval, help="s between saved frames while a person is seen")
    g.add_argument("--det-hold", type=float, default=d.det_hold, help="s a detection keeps the faster rate")
    g.add_argument("--idle-interval", type=float, default=d.idle_interval, help="base interval while the detector sees nobody")
    g.add_argument("--det-max-height", type=float, default=0.15, help="ignore detections taller than this share of the frame (close people)")


def add_layout_args(ap: argparse.ArgumentParser) -> None:
    g = ap.add_argument_group("HUD layout (normalized 0..1 boxes of a 16:9 frame; usually set in imagehook.toml)")
    g.add_argument("--icon-zone", type=float, nargs=4, default=list(ICON_ZONE), metavar=("X1", "Y1", "X2", "Y2"),
                   help="HUD body-status icon used by the interface check")
    g.add_argument("--change-ignore", type=config.boxes, default=CHANGE_IGNORE, metavar="JSON",
                   help="areas that change on their own, a JSON list of [x1, y1, x2, y2] boxes")


def policy_from(a: argparse.Namespace) -> PolicyConfig:
    return PolicyConfig(min_interval=a.min_interval, base_interval=a.base_interval, max_interval=a.max_interval,
                        idle_interval=a.idle_interval, global_ref=a.global_ref, local_ref=a.local_ref,
                        dup_novelty=a.dup_novelty, ui_threshold=a.ui_threshold, det_conf=a.det_conf,
                        det_interval=a.det_interval, det_hold=a.det_hold)


def layout_from(a: argparse.Namespace) -> HudLayout:
    try:
        return HudLayout(tuple(a.icon_zone), tuple(a.change_ignore))
    except ValueError as e:
        sys.exit(f"HUD layout: {e}")


def cmd_record(a: argparse.Namespace) -> None:
    from .capture import Recorder, RecordConfig

    layout = layout_from(a)
    cfg = RecordConfig(out=a.out, output_idx=a.output, height=a.height, ext=a.format, quality=a.quality,
                       tick=a.min_interval, toggle_key=a.toggle_key, quit_key=a.quit_key, burst_key=a.burst_key,
                       burst_back=a.burst_back, burst_forward=a.burst_forward, any_window=a.any_window,
                       ui_filter=not a.no_ui_filter, template=a.template, max_session_gb=a.max_session_gb,
                       min_free_gb=a.min_free_gb, target_ram_gb=a.target_ram_gb, max_ram_gb=a.max_ram_gb,
                       target_vram_gb=a.target_vram_gb, max_vram_gb=a.max_vram_gb, target_gpu_pct=a.target_gpu_pct,
                       max_gpu_pct=a.max_gpu_pct, beep=not a.no_beep, autostart=a.autostart, duration=a.duration,
                       detector=a.detector, det_imgsz=a.det_imgsz, det_max_height=a.det_max_height,
                       clip_key=a.clip_key, ffmpeg=a.ffmpeg, clip_fps=a.clip_fps, clip_seconds=a.clip_seconds,
                       clip_cq=a.clip_cq, policy=policy_from(a), game_processes=tuple(a.game_processes),
                       icon_zone=layout.icon_zone, change_ignore=layout.change_ignore,
                       config=a.config_path.name if a.config_path else None)
    Recorder(cfg).run()


def cmd_replay(a: argparse.Namespace) -> None:
    from .filters import CapturePolicy, load_template
    from .replay import iter_session, iter_video, label_report, replay

    layout = layout_from(a)
    policy = CapturePolicy(policy_from(a), None if a.no_ui_filter else load_template(a.template or DEFAULT_TEMPLATE),
                           layout)
    detector = None
    if a.replay_detector:
        from .detector import Detector, DetectorConfig

        detector = Detector(DetectorConfig(a.replay_detector, a.det_imgsz, max_height=a.det_max_height,
                                           max_vram_gb=a.replay_vram_gb,
                                           icon_zone=layout.icon_zone))
    src: Path = a.source
    is_video = src.is_file()
    if is_video:
        items = iter_video(src, a.tick, datetime.now(), layout)
        name = src.stem
    else:
        det_labels = None
        if a.det_labels:
            sess = src if (src / "frames").is_dir() else src.parent
            det_labels = sess / "labels_auto" if a.det_labels == "auto" else Path(a.det_labels)
        items = iter_session(src, det_labels, a.det_max_height, full=detector is not None, layout=layout)
        name = src.name
    write_dir = None
    if a.write:
        write_dir = a.write / (f"session_{datetime.now():%Y%m%d_%H%M%S}" if is_video else name) / "frames"
        if write_dir.exists() and any(write_dir.iterdir()):
            sys.exit(f"{write_dir} already has frames; choose another --write folder")
    res = replay(items, policy, detector, write_dir)
    n = len(res.decisions)
    report = {"source": name, "evaluated": n, "kept": len(res.kept),
              "kept_share": round(len(res.kept) / n, 4) if n else None, "by_reason": dict(res.reasons)}
    if not is_video:
        lab = label_report(src if (src / "frames").is_dir() else src.parent, res)
        if lab:
            report["labels"] = lab
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print("\n".join(replay_lines(report)))
    if write_dir:
        print(f"kept frames copied to {write_dir}")


def replay_lines(r: dict) -> list[str]:
    from .report import counts, share

    n = r["evaluated"]
    lines = [f"{r['source']}: {n:,} frames checked, {r['kept']:,} kept ({share(r['kept'], n)})",
             f"  reasons    {counts(r['by_reason'], n)}"]
    if lab := r.get("labels"):
        parts = [f"{k} {v['kept']:,} of {v['total']:,} kept ({share(v['kept'], v['total'])})"
                 for k, v in lab["by_status"].items()]
        lines.append(f"  labels     {' | '.join(parts)}")
        if lab.get("pos_covered") is not None:
            small = ""
            if lab.get("small_pos_covered") is not None:
                small = (f"; small people (under 40 px at 1080p, {lab['small_pos_frames']:,} frames): "
                         f"{100 * lab['small_pos_covered']:.1f}%")
            lines.append(f"  people     frames with people that have a kept frame with people within 2 s: "
                         f"{100 * lab['pos_covered']:.1f}%{small}")
    return lines


def cmd_info(a: argparse.Namespace) -> None:
    from . import report

    stats = []
    for t in a.sessions or [a.out]:
        if (t / "frames").is_dir():
            stats.append(report.session_stats(t))
            continue
        sessions = report.find_sessions(t) if t.is_dir() else []
        if not sessions:
            sys.exit(f"no session in {t} (a session folder has frames/ inside)")
        if a.json:
            stats += [report.session_stats(s) for s in sessions]
        else:
            print(f"{len(sessions)} session(s) in {t}:")
            print("\n".join("  " + line for line in report.session_table(sessions)))
            print("details: imagehook info <session folder>")
    if a.json:
        print(json.dumps(stats[0] if len(stats) == 1 else stats, indent=2))
        return
    for i, st in enumerate(stats):
        if i:
            print()
        print("\n".join(report.format_session(st)))
    if a.legend:
        print()
        print("\n".join(report.legend()))


def cmd_config(a: argparse.Namespace) -> None:
    if a.init:
        target = a.init
        if target.exists():
            sys.exit(f"{target} already exists; edit it, or delete it first")
        target.write_text(config.EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"created {target}: every option is commented out (= built-in default); uncomment what you change")
        return
    subparsers = a.subparsers
    src = a.config_path
    print(f"# settings file: {src if src else 'none, built-in defaults (create one: imagehook config --init)'}")
    print("# '<-' marks values from the settings file; the command line still overrides everything")
    opts: dict[str, object] = {}
    for name in ("record", "clip"):
        for dest in config.options(subparsers[name]):
            opts.setdefault(dest, subparsers[name].get_default(dest))
    shown = set()
    for section, keys in config.example_sections():
        print(f"\n[{section}]")
        for k in keys:
            if k not in opts:
                continue
            shown.add(k)
            v = opts[k]
            mark = "   # <-" if k in a.applied else ""
            print(f"# {k} = (not set){mark}" if v is None else f"{k} = {config.toml_value(v)}{mark}")
    rest = sorted(set(opts) - shown)
    if rest:
        print("\n# not in the example file: " + ", ".join(rest))


def add_clip_args(ap: argparse.ArgumentParser, standalone: bool = False) -> None:
    g = ap.add_argument_group("video clips (ffmpeg ddagrab + NVENC, H.264, native resolution)")
    if not standalone:
        g.add_argument("--clip-key", default="f7", help="start/stop a clip")
    g.add_argument("--clip-seconds", type=float, default=60.0 if standalone else 120.0,
                   help="clip length (standalone) or the longest clip before it stops by itself")
    g.add_argument("--clip-fps", type=int, default=30, help="clip frame rate")
    g.add_argument("--clip-cq", type=int, default=19, help="NVENC constant quality, lower = better and larger")
    g.add_argument("--ffmpeg", help="ffmpeg executable; not set = ffmpeg on PATH or $IMAGEHOOK_FFMPEG")


def cmd_clip(a: argparse.Namespace) -> None:
    import time

    from .clip import ClipRecorder, find_ffmpeg

    ffmpeg = find_ffmpeg(a.ffmpeg)
    if not ffmpeg:
        sys.exit("ffmpeg not found: put it on PATH, set IMAGEHOOK_FFMPEG or pass --ffmpeg")
    idx = a.output
    if idx is None:
        import bettercam

        from .capture import primary_output_index

        idx = primary_output_index(bettercam)
    if a.delay:
        print(f"recording starts in {a.delay:g} s...")
        time.sleep(a.delay)
    rec = ClipRecorder(ffmpeg, a.out, idx, a.clip_fps, a.clip_seconds, a.clip_cq)
    path = rec.start()
    print(f"recording {path} for {a.clip_seconds:g} s (Ctrl+C stops earlier)")
    try:
        while (info := rec.poll()) is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        info = rec.stop()
    print(json.dumps(info, indent=2))
    if not info["ok"]:
        sys.exit(1)


def cmd_template(a: argparse.Namespace) -> None:
    import cv2

    from .filters import build_template, features
    from .replay import IMG_EXT

    layout = layout_from(a)
    icons = []
    for s in a.sessions:
        frames = sorted(p for p in ((s / "frames") if (s / "frames").is_dir() else s).iterdir() if p.suffix.lower() in IMG_EXT)
        for p in frames[:: a.every]:
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None:
                icons.append(features(img, layout).icon)
    if len(icons) < 20:
        sys.exit("need at least 20 frames (mostly gameplay) to build a template")
    cv2.imwrite(str(a.template_out), build_template(icons).astype("uint8"))
    print(f"template from {len(icons)} frames -> {a.template_out}")


def build_parser() -> tuple[argparse.ArgumentParser, dict[str, argparse.ArgumentParser]]:
    from .capture import GAME_PROCESSES

    ap = argparse.ArgumentParser(prog="imagehook", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, help="settings file (default: ./imagehook.toml, then the project folder)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")
    fmt = HelpFormatter

    r = sub.add_parser("record", formatter_class=fmt, usage="imagehook record [options]",
                       help="record gameplay frames (hotkeys: F8 start/stop, mouse5 burst, F7 clip, F9 quit)")
    g = r.add_argument_group("output")
    g.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root folder for sessions ($IMAGEHOOK_OUT)")
    g.add_argument("--output", type=int, help="DXGI output (monitor) index; not set = the primary monitor")
    g.add_argument("--height", type=int, default=0, help="saved frame height; 0 = native monitor resolution")
    g.add_argument("--format", choices=("jpg", "png"), default="jpg", help="frame file format")
    g.add_argument("--quality", type=int, default=95, help="JPEG quality")
    g.add_argument("--max-session-gb", type=float, default=20.0, help="a session stops at this size")
    g.add_argument("--min-free-gb", type=float, default=20.0, help="recording stops below this free disk space")
    g = r.add_argument_group("game")
    g.add_argument("--game-exe", dest="game_processes", nargs="+", default=list(GAME_PROCESSES), metavar="EXE",
                   help="frames are taken only while one of these processes owns the foreground window")
    g.add_argument("--any-window", action="store_true", help="take frames whatever window is in the foreground")
    add_layout_args(r)
    g = r.add_argument_group("keys")
    g.add_argument("--toggle-key", default="f8", help="start / stop recording")
    g.add_argument("--burst-key", default="mouse5", help="keep the last --burst-back s and every new frame for --burst-forward s")
    g.add_argument("--quit-key", default="f9", help="quit")
    g.add_argument("--burst-back", type=float, default=10.0, help="s of unsaved frames a burst keeps")
    g.add_argument("--burst-forward", type=float, default=5.0, help="s a burst keeps every new non-duplicate frame")
    g.add_argument("--no-beep", action="store_true", help="no confirmation beeps")
    add_policy_args(r)
    g = r.add_argument_group("detector")
    g.add_argument("--detector", type=Path, help="optional YOLO .pt weights: frames with (small) people are saved more often")
    g.add_argument("--det-imgsz", type=int, default=0, help="detector input size; 0 = the frame's long side")
    add_clip_args(r)
    g = r.add_argument_group("limits (a target only prints a note; a hard limit turns the detector off)")
    g.add_argument("--target-ram-gb", type=float, default=2.0, help="RAM of the tool and ffmpeg")
    g.add_argument("--max-ram-gb", type=float, default=4.0, help="hard RAM limit: detector off, then recording stops")
    g.add_argument("--target-vram-gb", type=float, default=2.0, help="VRAM of the tool")
    g.add_argument("--max-vram-gb", type=float, default=4.0, help="hard VRAM limit: detector off")
    g.add_argument("--target-gpu-pct", type=float, default=20.0, help="own GPU load the detector is throttled to")
    g.add_argument("--max-gpu-pct", type=float, default=30.0, help="sustained own GPU load that turns the detector off")
    g = r.add_argument_group("run")
    g.add_argument("--autostart", action="store_true", help="start recording immediately")
    g.add_argument("--duration", type=float, help="quit after N seconds")
    r.set_defaults(func=cmd_record)

    i = sub.add_parser("info", help="what the sessions contain: a table of all sessions, or the details of one")
    i.add_argument("sessions", type=Path, nargs="*", help="session folders or a folder of sessions (default: --out)")
    i.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root folder for sessions (default %(default)s)")
    i.add_argument("--json", action="store_true", help="print the numbers as JSON")
    i.add_argument("--legend", action="store_true", help="explain the saved and skipped reasons")
    i.set_defaults(func=cmd_info)

    p = sub.add_parser("replay", formatter_class=fmt, usage="imagehook replay SOURCE [options]", help="run a recorded session or a video through the filters and report")
    p.add_argument("source", type=Path, help="session folder, its frames/ folder, or a video file")
    p.add_argument("--tick", type=float, default=0.5, help="video only: evaluate a frame every N s")
    p.add_argument("--det-labels", nargs="?", const="auto", help="emulate the detector with prelabel output (YOLO txt with a confidence column); without a value: SESSION/labels_auto")
    p.add_argument("--detector", dest="replay_detector", type=Path, help="run real YOLO .pt weights on the frames (GPU, game closed)")
    p.add_argument("--det-imgsz", type=int, default=0, help="0 = the frame's long side")
    p.add_argument("--max-vram-gb", dest="replay_vram_gb", type=float, default=1.0, help="PyTorch allocator cap for --detector")
    p.add_argument("--write", type=Path, help="copy kept frames to WRITE/<session>/frames")
    p.add_argument("--json", action="store_true", help="print the report as JSON")
    add_policy_args(p)
    add_layout_args(p)
    p.set_defaults(func=cmd_replay)

    c = sub.add_parser("clip", formatter_class=fmt, help="record one native-resolution video clip")
    c.add_argument("--out", type=Path, default=DEFAULT_OUT, help="clips go to OUT/clips ($IMAGEHOOK_OUT)")
    c.add_argument("--output", type=int, help="DXGI output (monitor) index; not set = the primary monitor")
    c.add_argument("--delay", type=float, default=0.0, help="wait N s before recording (time to switch to the game)")
    add_clip_args(c, standalone=True)
    c.set_defaults(func=cmd_clip)

    t = sub.add_parser("template", formatter_class=fmt,
                       help="rebuild the HUD icon template from recorded sessions (mostly gameplay)")
    t.add_argument("sessions", type=Path, nargs="+", help="session folders, mostly gameplay")
    t.add_argument("--out", dest="template_out", type=Path, default=DEFAULT_TEMPLATE, help="template PNG to write")
    t.add_argument("--every", type=int, default=10, help="use every N-th frame")
    add_layout_args(t)
    t.set_defaults(func=cmd_template)

    k = sub.add_parser("config", help="show the settings in effect, or create imagehook.toml")
    k.add_argument("--init", type=Path, nargs="?", const=Path(config.FILE_NAME), metavar="PATH",
                   help="create a settings file from imagehook.example.toml (default ./imagehook.toml)")
    k.set_defaults(func=cmd_config)
    return ap, {"record": r, "info": i, "replay": p, "clip": c, "template": t}


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", type=Path)
    known, rest = pre.parse_known_args(argv)
    ap, subparsers = build_parser()
    try:
        path = config.find_config(known.config)
        applied = config.apply(subparsers, path) if path else {}
    except config.ConfigError as e:
        sys.exit(f"settings: {e}")
    a = ap.parse_args(rest)
    a.config_path, a.applied, a.subparsers = path, applied, subparsers
    a.func(a)


if __name__ == "__main__":
    main()
