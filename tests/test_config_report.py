from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from imagehook import cli, config, report
from imagehook.filters import CHANGE_MASK, DEFAULT_LAYOUT, CapturePolicy, HudLayout, PolicyConfig, change, features
from imagehook.session import Session
from tests.conftest import scene


def parse(argv: list[str], cfg: Path | None):
    ap, subs = cli.build_parser()
    if cfg is not None:
        config.apply(subs, cfg)
    return ap.parse_args(argv)


def write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "imagehook.toml"
    p.write_text(text, encoding="utf-8")
    return p


def test_example_file_lists_every_record_and_clip_option():
    _, subs = cli.build_parser()
    documented = {k for _, keys in config.example_sections() for k in keys}
    options = set(config.options(subs["record"])) | set(config.options(subs["clip"]))
    assert documented == options
    # the example only has commented-out keys: loading it changes nothing
    assert config.load(config.EXAMPLE) == {}


def test_config_overrides_defaults_and_command_line_overrides_config(tmp_path):
    cfg = write(tmp_path, '[filters]\nui-threshold = 0.25\nmin_interval = 1\n[output]\nout = "caps"\n'
                          '[game]\ngame_processes = ["Game.exe"]\nicon_zone = [0, 0.7, 0.1, 0.9]\n'
                          '[keys]\nno_beep = true\n')
    a = parse(["record"], cfg)
    assert a.ui_threshold == 0.25 and a.min_interval == 1.0 and isinstance(a.min_interval, float)
    assert a.out == tmp_path / "caps"  # relative to the settings file
    assert a.game_processes == ["Game.exe"] and a.icon_zone == [0.0, 0.7, 0.1, 0.9] and a.no_beep is True
    a = parse(["record", "--ui-threshold", "0.4", "--game-exe", "other.exe"], cfg)
    assert a.ui_threshold == 0.4 and a.game_processes == ["other.exe"]
    # shared filter options reach replay too, record-only ones (detector) do not
    cfg2 = write(tmp_path, 'ui_threshold = 0.2\ndetector = "w.pt"\n')
    a = parse(["replay", "x"], cfg2)
    assert a.ui_threshold == 0.2 and a.replay_detector is None


@pytest.mark.parametrize("text,msg", [
    ("[filters]\nui_treshold = 0.2\n", "did you mean 'ui_threshold'"),
    ("quality = 'high'\n", "whole number"),
    ("quality = 9.5\n", "whole number"),
    ("no_beep = 1\n", "true or false"),
    ("format = 'bmp'\n", "one of jpg, png"),
    ("icon_zone = [0, 0.7, 0.1]\n", "4 values"),
    ("change_ignore = [[0, 0, 2, 1]]\n", "inside the frame"),
    ("[a]\nquality = 90\n[b]\nquality = 80\n", "set twice"),
    ("quality = \n", "imagehook.toml"),
])
def test_config_errors_are_explained(tmp_path, text, msg):
    with pytest.raises(config.ConfigError, match=msg):
        parse(["record"], write(tmp_path, text))


def test_find_config_order(tmp_path, monkeypatch):
    monkeypatch.delenv("IMAGEHOOK_CONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "PROJECT", tmp_path / "nowhere")
    assert config.find_config() is None
    local = write(tmp_path, "")
    assert config.find_config() == Path.cwd() / config.FILE_NAME == local.resolve()
    other = tmp_path / "other.toml"
    other.write_text("", encoding="utf-8")
    monkeypatch.setenv("IMAGEHOOK_CONFIG", str(other))
    assert config.find_config() == other
    assert config.find_config(local) == local
    with pytest.raises(config.ConfigError, match="not found"):
        config.find_config(tmp_path / "missing.toml")


def test_default_layout_is_unchanged_and_custom_layout_is_used():
    assert np.array_equal(DEFAULT_LAYOUT.change_mask, CHANGE_MASK)
    a, b = features(scene(3)).thumb, features(scene(4)).thumb
    assert change(a, b) == change(a, b, DEFAULT_LAYOUT.change_mask)
    # ignoring the whole frame but one strip changes the measurement; the icon follows icon_zone
    layout = HudLayout(icon_zone=(0.5, 0.5, 0.6, 0.6), change_ignore=((0.0, 0.0, 1.0, 0.9),))
    img = scene(5)
    assert not np.array_equal(features(img, layout).icon, features(img).icon)
    policy = CapturePolicy(PolicyConfig(), None, layout)
    policy.saved(0.0, policy.features(img))
    moved = img.copy()
    moved[:1000] = 0  # change only the ignored top part
    assert policy.novelty(policy.features(moved)) == 0.0
    with pytest.raises(ValueError):
        HudLayout(icon_zone=(0.5, 0.5, 0.4, 0.6))


def test_session_report_reads_summary_log_and_labels(tmp_path):
    t0 = datetime(2026, 10, 8, 12, 0, 0)
    s = Session(tmp_path, t0)
    img = scene(1, 144, 256)
    for i, reason in enumerate(["first", "change", "change", "burst"]):
        s.save(img, datetime(2026, 10, 8, 12, 0, i), reason)
    s.save(img, datetime(2026, 10, 8, 12, 5, 0), "change")  # after a pause
    s.skipped.update({"ui": 4, "waiting": 2, "not_game": 3})
    s.close({"tool": "imagehook test", "recorded_seconds": 60.0, "evaluations": 11,
             "settings": {"tick": 0.5}, "resources": {"rss_peak_gb": 0.2, "gpu_per_process": True}})
    (s.dir / "review.json").write_text(json.dumps({"status": {"120000_000": "pos", "120001_000": "neg"}}),
                                       encoding="utf-8")
    st = report.session_stats(s.dir)
    assert st["saved_total"] == 5 and st["frames_on_disk"] == 5
    assert st["gaps"] == {"median_s": 1.0, "p90_s": 1.0, "max_s": 1.0, "pauses": 1}
    assert st["labels"] == {"pos": 1, "neg": 1, "unlabeled": 3}
    text = "\n".join(report.format_session(st))
    assert "change 3 (60.0%)" in text and "ui 4 (36.4%)" in text and "not_game 3" in text
    assert "1 pause(s)" in text and "detector   not used" in text
    table = report.session_table(report.find_sessions(tmp_path))
    assert table[1].startswith(s.dir.name) and "36.4%" in table[1]


def test_info_and_replay_print_readable_reports(tmp_path, capsys):
    t0 = datetime(2026, 10, 8, 12, 0, 0)
    s = Session(tmp_path, t0)
    for i in range(3):
        s.save(scene(i, 540, 960), datetime(2026, 10, 8, 12, 0, i), "change")
    s.close({"tool": "imagehook test"})
    empty = write(tmp_path, "")
    cli.main(["--config", str(empty), "info", "--out", str(tmp_path)])
    out = capsys.readouterr().out
    assert "1 session(s)" in out and s.dir.name in out
    cli.main(["--config", str(empty), "info", str(s.dir), "--legend"])
    out = capsys.readouterr().out
    assert "saved      3 frames" in out and "skipped ui" in out
    cli.main(["--config", str(empty), "replay", str(s.dir)])
    out = capsys.readouterr().out
    assert out.startswith(f"{s.dir.name}: 3 frames checked, 3 kept (100.0%)")
    cli.main(["--config", str(empty), "replay", str(s.dir), "--json"])
    assert json.loads(capsys.readouterr().out)["kept"] == 3


def test_status_line_is_readable():
    line = report.status_line(3725, Counter(change=90, burst=10), Counter(ui=50, waiting=30, not_game=120), 200,
                              1.234, 41.4, 0.5, 0.31, 0.0, 1.0)
    assert line.startswith("REC 1:02:05 | saved 100 (+41/min): change 90, burst 10")
    assert "ui 25.0%, duplicate 0.0%, waiting 15.0%" in line and "outside the game 0:01:00" in line
