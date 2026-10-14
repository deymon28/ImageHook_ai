from __future__ import annotations

import numpy as np

from imagehook.filters import CapturePolicy, PolicyConfig, change, features, ui_score
from tests.conftest import scene


def test_features_shapes_any_resolution():
    for h, w in ((1440, 2560), (1080, 1920), (540, 960)):
        f = features(scene(0, h, w))
        assert f.thumb.shape == (180, 320) and f.thumb.dtype == np.uint8
        assert f.icon.shape == (92, 68) and f.icon.dtype == np.float32


def test_ui_score_separates_gameplay_from_interface(template):
    assert ui_score(features(scene(1)).icon, template) > 0.9
    assert ui_score(features(scene(1, icon=False)).icon, template) < 0.3
    black = np.zeros((1440, 2560, 3), np.uint8)
    assert ui_score(features(black).icon, template) == 0.0  # flat area: loading screen, no correlation


def test_ui_score_does_not_depend_on_resolution(template):
    a = ui_score(features(scene(2, 1440, 2560)).icon, template)
    b = ui_score(features(scene(2, 1080, 1920)).icon, template)
    assert abs(a - b) < 0.05


def test_change_identical_is_zero():
    t = features(scene(3)).thumb
    assert change(t, t) == (0.0, 0.0)


def test_change_new_view_is_global():
    g, local = change(features(scene(3)).thumb, features(scene(4)).thumb)
    assert g > 20 and local > 0.5


def test_change_small_object_in_still_view_is_local():
    base = scene(5)
    moved = base.copy()
    moved[700:820, 1200:1260] = 255  # ~120 px tall bright figure in the middle of a still view
    g, local = change(features(base).thumb, features(moved).thumb)
    assert g < 1.0
    assert local > 0.2  # well above the duplicate floor (0.15 * local_ref = 0.09): saved within a few seconds


def test_change_ignores_hud_areas():
    base = scene(6)
    hud = base.copy()
    hud[0:60, :] = 255  # top overlay: FPS, timer, perf counters
    hud[1400:, :] = 0  # bottom status bar
    hud[1300:1440, 2300:2560] = 128  # bottom-right panel
    assert change(features(base).thumb, features(hud).thumb) == (0.0, 0.0)


def policy(template, **kw) -> CapturePolicy:
    return CapturePolicy(PolicyConfig(**kw), template)


def test_first_frame_saved_and_interface_skipped(template):
    p = policy(template)
    d = p.evaluate(0.0, features(scene(1, icon=False)))
    assert not d.save and d.reason == "ui"
    d = p.evaluate(0.5, features(scene(1)))
    assert d.save and d.reason == "first"


def test_duplicate_never_saved_even_after_max_interval(template):
    p = policy(template)
    f = features(scene(1))
    p.saved(0.0, f)
    for t in (0.5, 1.0, 10.0, 60.0):
        d = p.evaluate(t, f)
        assert not d.save and d.reason == "duplicate"


def test_big_change_saved_at_min_interval_not_before(template):
    p = policy(template)
    p.saved(0.0, features(scene(1)))
    new_view = features(scene(2))
    assert p.evaluate(0.3, new_view).reason == "waiting"
    d = p.evaluate(0.5, new_view)
    assert d.save and d.reason == "change"


def test_small_change_waits_longer_then_periodic(template):
    cfg = dict(global_ref=15.0, local_ref=0.6, dup_novelty=0.15, max_interval=10.0)
    p = policy(template, **cfg)
    base = scene(7)
    p.saved(0.0, features(base))
    drift = np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8)  # lighting drift: novelty ~0.27
    f = features(drift)
    nov = p.novelty(f)
    assert 0.15 < nov < 0.5
    assert p.evaluate(1.0, f).reason == "waiting"
    assert p.evaluate(2.0, f).reason == "waiting"
    d = p.evaluate(1.0 / nov + 0.1, f)  # novelty * elapsed >= base_interval
    assert d.save and d.reason == "change"


def test_periodic_saves_a_nearly_static_view(template):
    p = policy(template, max_interval=10.0, base_interval=100.0)  # base so long that only the periodic rule fires
    base = scene(8)
    p.saved(0.0, features(base))
    f = features(np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8))
    assert p.evaluate(9.0, f).reason == "waiting"
    d = p.evaluate(10.0, f)
    assert d.save and d.reason == "periodic"


def test_burst_keeps_waiting_frames_but_not_ui_or_duplicates(template):
    p = policy(template)
    base = scene(9)
    p.saved(0.0, features(base))
    f = features(np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8))
    assert p.evaluate(0.5, f).reason == "waiting"
    assert p.evaluate(0.5, f, burst=True).reason == "burst"
    assert p.evaluate(0.5, features(base), burst=True).reason == "duplicate"
    assert p.evaluate(0.5, features(scene(9, icon=False)), burst=True).reason == "ui"


def test_detector_hit_saves_more_often_and_idle_less_often(template):
    base = scene(10)
    drift = features(np.clip(base.astype(np.int16) + 4, 0, 255).astype(np.uint8))  # small change, novelty ~0.27
    idle = policy(template, idle_interval=2.0, det_conf=0.25, det_interval=1.0)
    idle.saved(0.0, features(base))
    assert idle.evaluate(1.0, drift, det_conf=0.05).reason == "waiting"
    hit = policy(template, idle_interval=2.0, det_conf=0.25, det_interval=1.0)
    hit.saved(0.0, features(base))
    assert hit.evaluate(0.5, drift, det_conf=0.6).reason == "waiting"  # det_interval not reached yet
    d = hit.evaluate(1.0, drift, det_conf=0.6)
    assert d.save and d.reason == "detection"
    # the hit is held for det_hold seconds even if the detector misses the person in the next frame
    hit.saved(1.0, drift)
    drift2 = features(np.clip(base.astype(np.int16) + 8, 0, 255).astype(np.uint8))
    assert hit.evaluate(2.0, drift2, det_conf=0.0).reason == "detection"
    assert hit.evaluate(5.0, drift2, det_conf=0.0).reason != "detection"


def test_idle_interval_is_slower_than_base_without_detector(template):
    base = scene(11)
    f = features(scene(12))  # completely new view
    plain = policy(template, base_interval=1.0)
    plain.saved(0.0, features(base))
    with_det = policy(template, base_interval=1.0, idle_interval=100.0)
    with_det.saved(0.0, features(base))
    nov = plain.novelty(f)
    t = 1.0 / nov + 0.6  # enough for base_interval, far from enough for idle_interval
    assert plain.evaluate(t, f).save
    assert with_det.evaluate(t, f, det_conf=0.0).reason == "waiting"


def test_without_template_every_frame_counts_as_gameplay():
    p = CapturePolicy(PolicyConfig(), None)
    assert p.evaluate(0.0, features(scene(1, icon=False))).reason == "first"
