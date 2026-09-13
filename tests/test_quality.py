"""Tests for the frame quality gate.

Synthetic frames stand in for real camera input: a covered lens really does produce
a dark, featureless field, and an out-of-focus frame really does have a low Laplacian
variance, so these are faithful to the property being tested.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.quality import FrameQuality, QualityMonitor, assess  # noqa: E402

RNG = np.random.default_rng(11)
SHAPE = (240, 320, 3)


def covered() -> np.ndarray:
    """Thumb over the lens: near-black, almost no variation."""
    return np.clip(RNG.normal(6, 2, SHAPE), 0, 255).astype(np.uint8)


def dim_room() -> np.ndarray:
    """Genuinely dark, but with real structure in it.

    Mean luma stays under the darkness threshold while the standard deviation stays
    well above the flatness threshold — which is exactly what a dim but non-empty
    room looks like, and exactly what must not be mistaken for a covered lens.
    """
    frame = np.full(SHAPE, 8, dtype=np.uint8)
    for start in range(0, SHAPE[0], 32):
        frame[start:start + 16] = 38
    return frame


def blank_wall() -> np.ndarray:
    """Lens pressed to something bright and uniform."""
    return np.full(SHAPE, 210, dtype=np.uint8)


def sharp_scene() -> np.ndarray:
    """A well-lit scene with hard edges."""
    frame = RNG.integers(60, 200, SHAPE, dtype=np.uint8)
    frame[40:120, 60:180] = 20
    frame[:, ::16] = 250
    return frame


def blurred_scene() -> np.ndarray:
    """Well-lit and varied at low frequency, but with no sharp edges."""
    rows = np.linspace(40, 210, SHAPE[0], dtype=np.float32)[:, None]
    cols = np.linspace(0, 40, SHAPE[1], dtype=np.float32)[None, :]
    gradient = rows + cols
    return np.repeat(gradient[..., None], 3, axis=2).astype(np.uint8)


def test_covered_lens_is_reported_as_occluded():
    q = assess(covered())
    assert not q.ok and q.reason == "occluded"
    assert q.message == "Camera is covered."


def test_bright_uniform_field_is_also_occlusion():
    q = assess(blank_wall())
    assert not q.ok and q.reason == "occluded"


def test_dark_but_textured_scene_is_darkness_not_occlusion():
    """The remedy differs — turn on a light versus move your hand — so the app must not conflate them."""
    q = assess(dim_room())
    assert not q.ok and q.reason == "dark"
    assert q.message == "Too dark to see."


def test_blurred_scene_is_detected():
    q = assess(blurred_scene())
    assert not q.ok and q.reason == "blurred"


def test_good_frame_passes():
    q = assess(sharp_scene())
    assert q.ok and q.reason is None and q.message is None


def test_grayscale_input_is_accepted():
    gray = sharp_scene()[..., 0]
    assert assess(gray).ok


def test_single_bad_frame_does_not_trigger_an_announcement():
    """One dropped or motion-blurred frame must not make the assistant cry wolf."""
    monitor = QualityMonitor()
    assert monitor.update(assess(covered())) is None
    assert monitor.update(assess(sharp_scene())) is None
    assert not monitor.degraded


def test_sustained_occlusion_is_announced_exactly_once():
    monitor = QualityMonitor(degrade_frames=3)
    bad = assess(covered())
    assert monitor.update(bad) is None
    assert monitor.update(bad) is None
    assert monitor.update(bad) == "Camera is covered."
    # Still covered on later frames, but the user has already been told.
    assert monitor.update(bad) is None
    assert monitor.update(bad) is None
    assert monitor.degraded


def test_recovery_is_announced_so_silence_stops_being_ambiguous():
    monitor = QualityMonitor(degrade_frames=2, recover_frames=2)
    bad, good = assess(covered()), assess(sharp_scene())
    monitor.update(bad)
    assert monitor.update(bad) == "Camera is covered."
    assert monitor.update(good) is None
    assert monitor.update(good) == "Camera clear."
    assert not monitor.degraded


def test_a_changed_failure_mode_is_re_announced():
    """Going from covered to merely dark is new information with a new remedy."""
    monitor = QualityMonitor(degrade_frames=2)
    covered_q, dark_q = assess(covered()), assess(dim_room())
    monitor.update(covered_q)
    assert monitor.update(covered_q) == "Camera is covered."
    # The bad streak is already past threshold, so a *different* problem is
    # announced on the very next frame rather than waiting out the debounce again.
    assert monitor.update(dark_q) == "Too dark to see."
    assert monitor.update(dark_q) is None


def test_monitor_stays_quiet_through_a_normal_session():
    monitor = QualityMonitor()
    messages = [monitor.update(assess(sharp_scene())) for _ in range(60)]
    assert all(m is None for m in messages)


def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok    {fn.__name__}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
