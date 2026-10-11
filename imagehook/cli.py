"""Command line: record (live capture), replay (offline filter check), template (rebuild the HUD icon template)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from .filters import DEFAULT_TEMPLATE, PolicyConfig

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = Path(os.environ.get("IMAGEHOOK_OUT", PROJECT / "captures"))


def add_policy_args(ap: argparse.ArgumentParser) -> None:
    d = PolicyConfig()
    g = ap.add_argument_group("filtering")
    g.add_argument("--min-interval", type=float, default=d.min_interval, help=f"s between saved frames at most change (default {d.min_interval})")
    g.add_argument("--base-interval", type=float, default=d.base_interval, help=f"s between saved frames for an average change (default {d.base_interval})")
    g.add_argument("--max-interval", type=float, default=d.max_interval, help=f"a nearly static view is still saved this often, s (default {d.max_interval})")
    g.add_argument("--idle-interval", type=float, default=d.idle_interval, help=f"base interval while the detector sees nobody (default {d.idle_interval})")
    g.add_argument("--dup-novelty", type=float, default=d.dup_novelty, help=f"novelty below this = duplicate, never saved (default {d.dup_novelty})")
    g.add_argument("--ui-threshold", type=float, default=d.ui_threshold, help=f"HUD icon similarity below this = interface screen (default {d.ui_threshold})")
    g.add_argument("--no-ui-filter", action="store_true", help="keep interface screens too")
    g.add_argument("--template", type=Path, help=f"HUD icon template PNG (default: the bundled {DEFAULT_TEMPLATE.name})")
    g.add_argument("--det-conf", type=float, default=d.det_conf, help=f"detector confidence that counts as a person (default {d.det_conf})")
    g.add_argument("--det-interval", type=float, default=d.det_interval, help=f"s between saved frames while a person is detected (default {d.det_interval})")
    g.add_argument("--det-max-height", type=float, default=0.15, help="ignore detections taller than this share of the frame (close people), default 0.15")


def policy_from(a: argparse.Namespace) -> PolicyConfig:
    return PolicyConfig(min_interval=a.min_interval, base_interval=a.base_interval, max_interval=a.max_interval,
                        idle_interval=a.idle_interval, dup_novelty=a.dup_novelty, ui_threshold=a.ui_threshold,
                        det_conf=a.det_conf, det_interval=a.det_interval)


def cmd_record(a: argparse.Namespace) -> None:
    from .capture import Recorder, RecordConfig

    cfg = RecordConfig(out=a.out, output_idx=a.output, height=a.height, ext=a.format, quality=a.quality,
                       tick=a.min_interval, toggle_key=a.toggle_key, quit_key=a.quit_key, burst_key=a.burst_key,
                       burst_back=a.burst_back, burst_forward=a.burst_forward, any_window=a.any_window,
                       ui_filter=not a.no_ui_filter, template=a.template, max_session_gb=a.max_session_gb,
                       min_free_gb=a.min_free_gb, max_ram_gb=a.max_ram_gb, max_vram_gb=a.max_vram_gb,
                       beep=not a.no_beep, autostart=a.autostart, duration=a.duration, detector=a.detector,
                       det_imgsz=a.det_imgsz, det_max_height=a.det_max_height, policy=policy_from(a))
    Recorder(cfg).run()


def cmd_replay(a: argparse.Namespace) -> None:
    from .filters import CapturePolicy, load_template
    from .replay import iter_session, iter_video, label_report, replay

    policy = CapturePolicy(policy_from(a), None if a.no_ui_filter else load_template(a.template or DEFAULT_TEMPLATE))
    detector = None
    if a.detector:
        from .detector import Detector, DetectorConfig

        detector = Detector(DetectorConfig(a.detector, a.det_imgsz, max_height=a.det_max_height, max_vram_gb=a.max_vram_gb))
    src: Path = a.source
    is_video = src.is_file()
    if is_video:
        items = iter_video(src, a.tick, datetime.now())
        name = src.stem
    else:
        det_labels = None
        if a.det_labels:
            sess = src if (src / "frames").is_dir() else src.parent
            det_labels = sess / "labels_auto" if a.det_labels == "auto" else Path(a.det_labels)
        items = iter_session(src, det_labels, a.det_max_height, full=detector is not None)
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
    print(json.dumps(report, indent=2))
    if write_dir:
        print(f"kept frames copied to {write_dir}")


def cmd_template(a: argparse.Namespace) -> None:
    import cv2

    from .filters import build_template, features
    from .replay import IMG_EXT

    icons = []
    for s in a.sessions:
        frames = sorted(p for p in ((s / "frames") if (s / "frames").is_dir() else s).iterdir() if p.suffix.lower() in IMG_EXT)
        for p in frames[:: a.every]:
            img = cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_2)
            if img is not None:
                icons.append(features(img).icon)
    if len(icons) < 20:
        sys.exit("need at least 20 frames (mostly gameplay) to build a template")
    cv2.imwrite(str(a.out), build_template(icons).astype("uint8"))
    print(f"template from {len(icons)} frames -> {a.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="imagehook", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("record", help="record gameplay frames (hotkeys: F8 start/stop, mouse5 burst, F9 quit)")
    r.add_argument("--out", type=Path, default=DEFAULT_OUT, help="root folder for sessions (default: ./captures or $IMAGEHOOK_OUT)")
    r.add_argument("--output", type=int, help="DXGI output (monitor) index, default: primary")
    r.add_argument("--height", type=int, default=0, help="saved frame height; 0 = native monitor resolution (default)")
    r.add_argument("--format", choices=("jpg", "png"), default="jpg")
    r.add_argument("--quality", type=int, default=95, help="JPEG quality (default 95)")
    r.add_argument("--toggle-key", default="f8")
    r.add_argument("--quit-key", default="f9")
    r.add_argument("--burst-key", default="mouse5", help="keep the last --burst-back s and the next --burst-forward s (default mouse5 = front side button)")
    r.add_argument("--burst-back", type=float, default=10.0)
    r.add_argument("--burst-forward", type=float, default=5.0)
    r.add_argument("--any-window", action="store_true", help="record even when the game is not the foreground window")
    r.add_argument("--max-session-gb", type=float, default=20.0)
    r.add_argument("--min-free-gb", type=float, default=20.0)
    r.add_argument("--max-ram-gb", type=float, default=4.0)
    r.add_argument("--max-vram-gb", type=float, default=2.0)
    r.add_argument("--detector", type=Path, help="optional YOLO .pt weights: frames with (small) people are saved more often")
    r.add_argument("--det-imgsz", type=int, default=1280)
    r.add_argument("--no-beep", action="store_true")
    r.add_argument("--autostart", action="store_true", help="start recording immediately")
    r.add_argument("--duration", type=float, help="quit after N seconds")
    add_policy_args(r)
    r.set_defaults(func=cmd_record)

    p = sub.add_parser("replay", help="run a recorded session or a video through the filters and report")
    p.add_argument("source", type=Path, help="session folder, its frames/ folder, or a video file")
    p.add_argument("--tick", type=float, default=0.5, help="video only: evaluate a frame every N s (default 0.5)")
    p.add_argument("--det-labels", nargs="?", const="auto", help="emulate the detector with prelabel output (YOLO txt with a confidence column); without a value: SESSION/labels_auto")
    p.add_argument("--detector", type=Path, help="run real YOLO .pt weights on the frames (GPU, game closed)")
    p.add_argument("--det-imgsz", type=int, default=1280)
    p.add_argument("--max-vram-gb", type=float, default=1.0)
    p.add_argument("--write", type=Path, help="copy kept frames to WRITE/<session>/frames")
    add_policy_args(p)
    p.set_defaults(func=cmd_replay)

    t = sub.add_parser("template", help="rebuild the HUD icon template from recorded sessions (mostly gameplay)")
    t.add_argument("sessions", type=Path, nargs="+")
    t.add_argument("--out", type=Path, default=DEFAULT_TEMPLATE)
    t.add_argument("--every", type=int, default=10, help="use every N-th frame (default 10)")
    t.set_defaults(func=cmd_template)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
