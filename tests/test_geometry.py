"""Tests for the disparity-to-metres calibration.

Like the policy tests these are pure — no model, no camera. They pin down the one
thing that makes distance announcements trustworthy.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drishti.geometry import (  # noqa: E402
    DisparityCalibrator,
    focal_length_pixels,
    pinhole_distance_m,
)


def test_focal_length_is_plausible_for_a_webcam():
    focal = focal_length_pixels(720)
    # A 720-row sensor at ~55 deg vertical FOV lands around 690 px.
    assert 600 < focal < 800


def test_pinhole_distance_halves_when_object_looks_twice_as_tall():
    focal = focal_length_pixels(720)
    far = pinhole_distance_m("person", bbox_height_px=100, focal_px=focal)
    near = pinhole_distance_m("person", bbox_height_px=200, focal_px=focal)
    assert far is not None and near is not None
    assert abs(far / near - 2.0) < 1e-6


def test_pinhole_returns_none_for_unknown_class():
    assert pinhole_distance_m("teddy bear", 120, 700.0) is None


def test_uncalibrated_calibrator_still_returns_a_finite_distance():
    calib = DisparityCalibrator()
    assert not calib.is_calibrated
    distance = calib.to_metres(2.0)
    assert 0.2 <= distance <= 50.0


def test_calibrator_recovers_a_known_linear_relationship():
    # Ground truth: disparity = 12 * (1/z) + 0.5
    truth_a, truth_b = 12.0, 0.5
    anchors = [(truth_a * (1.0 / z) + truth_b, z) for z in (1.0, 2.0, 4.0, 8.0)]

    calib = DisparityCalibrator(smoothing=0.0)
    calib.update(anchors)
    assert calib.is_calibrated

    for disparity, expected_z in anchors:
        assert abs(calib.to_metres(disparity) - expected_z) < 0.05


def test_one_bad_frame_does_not_move_the_estimate():
    """A poster mistaken for a person must not shift every announced distance."""
    stable = [(12.0 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)]
    calib = DisparityCalibrator(smoothing=0.9)
    calib.update(stable)
    settled = calib.to_metres(6.0)

    calib.update([(100.0 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)])
    assert calib.to_metres(6.0) == settled


def test_a_persistently_different_scene_is_eventually_adopted():
    """Rejecting outliers must not mean never adapting to a genuinely new scene."""
    calib = DisparityCalibrator(smoothing=0.9)
    calib.update([(12.0 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)])
    settled = calib.to_metres(6.0)

    new_scene = [(100.0 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)]
    for _ in range(8):
        calib.update(new_scene)

    assert calib.to_metres(6.0) != settled
    assert abs(calib.to_metres(6.0) - 100.0 / 6.0) < 0.5


def test_gentle_drift_is_tracked_smoothly():
    calib = DisparityCalibrator(smoothing=0.85)
    calib.update([(12.0 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)])
    before = calib.to_metres(6.0)
    # A 20% change is within tolerance, so it should be absorbed, not rejected.
    calib.update([(14.4 * (1.0 / z), z) for z in (1.0, 2.0, 4.0)])
    after = calib.to_metres(6.0)
    assert before < after < before * 1.2


def test_empty_or_degenerate_anchors_are_ignored():
    calib = DisparityCalibrator()
    calib.update([])
    assert not calib.is_calibrated
    calib.update([(1.0, 0.0)])  # zero distance is not usable
    assert not calib.is_calibrated


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
