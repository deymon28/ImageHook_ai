"""Human-readable summaries: the live status line, the end-of-session summary and `imagehook info`.

Everything here reads the session files (capture.json, capture_log.csv, review.json) or plain counters; nothing is
imported by the capture loop except the small formatting helpers.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

# what each reason in capture.json / capture_log.csv means
SAVED_REASONS = {
    "first": "first frame of the session (or after a pause)",
    "change": "the scene changed enough for the time since the last saved frame",
    "periodic": "a nearly still view, saved every --max-interval s",
    "detection": "the detector saw a person",
    "burst": "kept by the burst key",
}
SKIPPED_REASONS = {
    "ui": "interface screen (no HUD body icon): inventory, map, loading, death or extraction screen",
    "duplicate": "almost the same as the last saved frame",
    "waiting": "changed too little for the time since the last saved frame (kept 10 s for a burst)",
    "not_game": "the game was not the foreground window (not evaluated)",
    "no_new_frame": "nothing on screen changed since the previous grab (not evaluated)",
}
EVALUATED_SKIPS = ("ui", "duplicate", "waiting")
PAUSE_GAP_S = 60.0  # a gap between saved frames longer than this is counted as a pause, not as a filter gap


def duration(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = int(round(seconds))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def share(n: int, total: int) -> str:
    return f"{100 * n / total:.1f}%" if total else "-"


def counts(counter: dict[str, int], total: int | None = None, order: tuple[str, ...] = ()) -> str:
    """'change 5,123 (93.0%), burst 383 (7.0%)': keys in `order` first, in that order, then the rest, largest first."""
    keys = [k for k in order if counter.get(k)] + sorted((k for k in counter if k not in order and counter[k]),
                                                         key=lambda k: -counter[k])
    parts = [f"{k} {counter[k]:,}" + (f" ({share(counter[k], total)})" if total else "") for k in keys]
    return ", ".join(parts) or "none"


def status_line(rec_seconds: float, saved: Counter, skipped: Counter, evaluations: int, gb: float,
                per_min: float | None, tick: float, rss: float, vram: float, gpu: float, extra: str = "") -> str:
    """One line printed every minute while recording."""
    n = sum(saved.values())
    rate = f" (+{per_min:.0f}/min)" if per_min is not None else ""
    skip = {k: skipped.get(k, 0) for k in EVALUATED_SKIPS}
    parts = [
        f"REC {duration(rec_seconds)}",
        f"saved {n:,}{rate}: {counts(saved)}",
        f"skipped of {evaluations:,} checks: " + ", ".join(f"{k} {share(v, evaluations)}" for k, v in skip.items()),
    ]
    if skipped.get("not_game"):
        parts.append(f"outside the game {duration(skipped['not_game'] * tick)}")
    parts.append(f"{gb:.2f} GB")
    parts.append(f"RAM {rss:.2f} GB, VRAM {vram:.2f} GB, own GPU {gpu:.0f}%{extra}")
    return " | ".join(parts)


# ---------------------------------------------------------------- sessions on disk
def frame_times(session: Path, names: list[str]) -> list[float]:
    """Capture times (s since the session's midnight) of frame names, in recording order."""
    from .replay import name_seconds, session_start_seconds

    start = session_start_seconds(session)
    out = []
    for n in names:
        t = name_seconds(n)
        if t is None:
            continue
        if start is not None and t - start < -60:  # the name is a time of day: after midnight
            t += 86400
        out.append(t)
    return sorted(out)


def percentile(values: list[float], q: float) -> float:
    v = sorted(values)
    return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]


def session_stats(session: Path) -> dict:
    """Everything `info` reports about one session folder, as plain data (also printed by `info --json`)."""
    cap = {}
    if (session / "capture.json").is_file():
        cap = json.loads((session / "capture.json").read_text(encoding="utf-8"))
    names: list[str] = []
    log_reasons: Counter = Counter()
    if (session / "capture_log.csv").is_file():
        with open(session / "capture_log.csv", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                names.append(row["name"])
                log_reasons[row["reason"]] += 1
    frames_dir = session / "frames"
    files = [p.stem for p in frames_dir.iterdir() if p.is_file()] if frames_dir.is_dir() else []
    n_files = len(files)
    if not names:  # recorded by an older collector: no log, the frame names still give the times
        names = files
    saved = Counter(cap.get("saved_by_reason") or log_reasons)
    st: dict = {
        "session": session.name,
        "tool": cap.get("tool"),
        "started": cap.get("started"),
        "ended": cap.get("ended"),
        "recorded_seconds": cap.get("recorded_seconds"),
        "frames_on_disk": n_files,
        "saved_total": cap.get("saved_total", sum(saved.values()) or n_files),
        "saved_by_reason": dict(saved),
        "skipped_by_reason": cap.get("skipped_by_reason", {}),
        "evaluations": cap.get("evaluations"),
        "bytes": cap.get("bytes"),
        "dropped_writer_full": cap.get("dropped_writer_full", 0),
        "write_errors": cap.get("write_errors", 0),
        "resources": cap.get("resources", {}),
        "detector": None,
        "clips": cap.get("clips", []),
        "settings": cap.get("settings", {}),
    }
    if cap.get("detector_runs"):
        st["detector"] = {"weights": cap.get("settings", {}).get("detector"), "runs": cap["detector_runs"],
                          "ms_avg": cap.get("detector_ms_avg"), "stride_end": cap.get("detector_stride_end")}
    times = frame_times(session, names)
    gaps = [b - a for a, b in zip(times, times[1:])]
    filt = [g for g in gaps if g <= PAUSE_GAP_S]
    if filt:
        st["gaps"] = {"median_s": round(percentile(filt, 0.5), 2), "p90_s": round(percentile(filt, 0.9), 2),
                      "max_s": round(max(filt), 2), "pauses": len(gaps) - len(filt)}
    if (session / "review.json").is_file():
        status = json.loads((session / "review.json").read_text(encoding="utf-8")).get("status", {})
        lab = Counter(status.values())
        lab["unlabeled"] = max(0, len(names) - sum(lab.values())) if names else 0
        st["labels"] = {k: v for k, v in lab.items() if v}
    return st


def format_session(st: dict) -> list[str]:
    lines = [f"{st['session']}" + (f"   ({st['tool']})" if st.get("tool") else "")]
    if st.get("started"):
        end = f", ended {st['ended'].replace('T', ' ')}" if st.get("ended") else ""
        lines.append(f"  time       started {st['started'].replace('T', ' ')}{end}, "
                     f"recorded {duration(st.get('recorded_seconds'))}")
    n = st["saved_total"]
    size = ""
    if st.get("bytes"):
        size = f", {st['bytes'] / 2**30:.2f} GB ({st['bytes'] / max(n, 1) / 2**20:.2f} MB per frame)"
    rate = ""
    if st.get("recorded_seconds"):
        rate = f", {n / st['recorded_seconds'] * 60:.0f} per recorded minute"
    lines.append(f"  saved      {n:,} frames{size}{rate}")
    if st["saved_by_reason"]:
        lines.append(f"             {counts(st['saved_by_reason'], n, tuple(SAVED_REASONS))}")
    if not st.get("tool"):
        lines.append("  note       no capture.json: recorded by an older collector, only frames and labels are known")
    if st["frames_on_disk"] != n:
        lines.append(f"  NOTE       {st['frames_on_disk']:,} files in frames/ (the labeling tool may have removed or added some)")
    sk = st["skipped_by_reason"]
    ev = st.get("evaluations")
    if ev:
        tick = st.get("settings", {}).get("tick", 0.5)
        lines.append(f"  checked    {ev:,} frames (one every {tick:g} s while recording in the game)")
        lines.append("  skipped    " + ", ".join(f"{k} {sk.get(k, 0):,} ({share(sk.get(k, 0), ev)})" for k in EVALUATED_SKIPS))
        outside = [f"{k} {sk[k]:,}" for k in ("not_game", "no_new_frame") if sk.get(k)]
        if outside:
            lines.append(f"             not checked: {', '.join(outside)}")
    elif sk:
        lines.append(f"  skipped    {counts(sk)}")
    if g := st.get("gaps"):
        pauses = f"; {g['pauses']} pause(s) over {PAUSE_GAP_S:g} s" if g["pauses"] else ""
        lines.append(f"  gaps       between saved frames: median {g['median_s']:g} s, 90% within {g['p90_s']:g} s, "
                     f"longest {g['max_s']:g} s{pauses}")
    if st.get("dropped_writer_full") or st.get("write_errors"):
        lines.append(f"  PROBLEMS   {st['dropped_writer_full']} frames dropped (disk writer behind), "
                     f"{st['write_errors']} write errors")
    r = st.get("resources") or {}
    if r:
        gpu = (f"own GPU avg {r.get('gpu_sm_avg_pct', 0):g}% (peak {r.get('gpu_sm_peak_pct', 0):g}%)"
               if r.get("gpu_per_process") else "own GPU load not available")
        lines.append(f"  resources  RAM peak {r.get('rss_peak_gb', 0):.2f} GB, CPU avg {r.get('cpu_avg_pct', 0):g}% "
                     f"(peak {r.get('cpu_peak_pct', 0):g}%), VRAM peak {r.get('vram_peak_gb', 0):.2f} GB, {gpu}")
    d = st.get("detector")
    if st.get("tool"):
            lines.append("  detector   " + (f"{d['weights']}: {d['runs']:,} runs, {d['ms_avg']} ms avg, "
                                        f"ran on every {d['stride_end']}. frame at the end" if d else "not used"))
    for c in st.get("clips") or []:
        state = "ok" if c.get("ok") else f"FAILED: {c.get('error', '')}"
        lines.append(f"  clip       {c['file']}: {c['seconds']} s, {c['bytes'] / 2**20:.0f} MB, {state}")
    if lab := st.get("labels"):
        total = sum(lab.values())
        lines.append(f"  labels     {counts(lab, total, ('pos', 'neg', 'skip', 'unlabeled'))}")
    return lines


def legend() -> list[str]:
    out = ["reasons:"]
    out += [f"  saved   {k:<13} {v}" for k, v in SAVED_REASONS.items()]
    out += [f"  skipped {k:<13} {v}" for k, v in SKIPPED_REASONS.items()]
    return out


def find_sessions(root: Path) -> list[Path]:
    return sorted(p for p in root.glob("session_*") if p.is_dir() and (p / "frames").is_dir())


def session_table(sessions: list[Path]) -> list[str]:
    rows = [("session", "recorded", "frames", "GB", "interface", "labels")]
    for s in sessions:
        st = session_stats(s)
        ev = st.get("evaluations") or 0
        ui = share(st["skipped_by_reason"].get("ui", 0), ev) if ev else "-"
        lab = st.get("labels")
        lab_s = (f"{sum(v for k, v in lab.items() if k != 'unlabeled'):,} reviewed" if lab else "-")
        gb = f"{st['bytes'] / 2**30:.2f}" if st.get("bytes") else "-"
        rows.append((s.name, duration(st.get("recorded_seconds")) if st.get("recorded_seconds") else "-",
                     f"{st['saved_total']:,}", gb, ui, lab_s))
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    return ["  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(r, widths))) for r in rows]
