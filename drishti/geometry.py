"""Turning pixels into metres.

Midas-V2 predicts *relative inverse depth* (disparity), not distance in metres. That
is a real limitation and pretending otherwise would produce an assistant that says
"chair, two metres" when the chair is five metres away — worse than saying nothing.

Drishti resolves it by fitting the relative depth map to metric units every frame,
using detected objects of known physical size as anchors. A pinhole camera model
gives an independent metric estimate for any object whose real-world height we know
reasonably well:

    z = (f_pixels * real_height_m) / bbox_height_pixels

Those anchor pairs (disparity, 1/z) are then fitted with a one-dimensional least
squares to recover the affine transform that maps the whole disparity map to metres.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Median standing/physical heights in metres. Only classes with a tight distribution
# are used as calibration anchors — a "person" is a good ruler, a "backpack" is not.
CLASS_HEIGHT_M: dict[str, float] = {
    "person": 1.68,
    "car": 1.50,
    "bus": 3.20,
    "truck": 3.20,
    "bicycle": 1.10,
    "motorcycle": 1.20,
    "chair": 0.90,
    "dining table": 0.75,
    "bench": 0.90,
    "stop sign": 2.10,
    "fire hydrant": 0.75,
    "parking meter": 1.20,
    "traffic light": 3.00,
    "door": 2.00,
}

# Classes trustworthy enough to calibrate against.
ANCHOR_CLASSES = frozenset({"person", "car", "bus", "truck", "chair", "bicycle"})


def focal_length_pixels(frame_height_px: int, vertical_fov_deg: float = 55.0) -> float:
    """Approximate focal length from frame height and an assumed vertical FOV.

    Laptop webcams cluster tightly around a 55-65 degree vertical field of view, so
    this is a serviceable prior — but it is a prior, and every distance the assistant
    announces scales off it. Measure the real value for the target machine (photograph
    an object of known height at a known distance and solve f = z * h_px / h_m) and
    pass it explicitly. A one-time calibration helper is outstanding work, listed in
    the README.
    """
    half_fov_rad = np.radians(vertical_fov_deg / 2.0)
    return (frame_height_px / 2.0) / np.tan(half_fov_rad)


def pinhole_distance_m(
    label: str, bbox_height_px: float, focal_px: float
) -> float | None:
    """Metric distance to an object of known height, or None if we cannot know."""
    real_height = CLASS_HEIGHT_M.get(label)
    if real_height is None or bbox_height_px <= 1.0:
        return None
    return float(focal_px * real_height / bbox_height_px)


# A new frame's scale estimate this far off the running one is treated as an outlier
# — a misdetection, a poster of a person, a partially-occluded car — and rejected.
OUTLIER_RATIO = 2.0

# ...unless it keeps happening, which means the scene really did change (different
# camera, different room) and the running estimate is the thing that is now wrong.
OUTLIER_PATIENCE = 5


@dataclass
class DisparityCalibrator:
    """Maintains the affine fit from Midas disparity to inverse metres.

    Model:  disparity ≈ a * (1 / z) + b   =>   z = a / (disparity - b)

    The fit is smoothed across frames because individual frames may contain only one
    anchor, and a single-anchor fit is noisy. Smoothing alone is not enough: an
    exponential average is not robust to large outliers, and one frame that mistakes
    a poster for a person would visibly shift every distance the assistant announces.
    So a wildly inconsistent fit is rejected outright, and only adopted if it persists.
    """

    smoothing: float = 0.85
    fallback_scale: float = 8.0
    _a: float | None = None
    _b: float = 0.0
    _outlier_streak: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._a is not None

    def update(self, anchors: list[tuple[float, float]]) -> None:
        """Refit from this frame's anchors.

        Args:
            anchors: (disparity, metric_distance_m) pairs from anchor-class detections.
        """
        usable = [(d, z) for d, z in anchors if z > 0.1 and np.isfinite(d)]
        if not usable:
            return

        disparity = np.array([d for d, _ in usable], dtype=np.float64)
        inv_z = np.array([1.0 / z for _, z in usable], dtype=np.float64)

        if len(usable) == 1:
            # One anchor cannot determine an offset; assume b=0 and solve for scale.
            a_new, b_new = float(disparity[0] / max(inv_z[0], 1e-6)), 0.0
        else:
            design = np.stack([inv_z, np.ones_like(inv_z)], axis=1)
            (a_new, b_new), *_ = np.linalg.lstsq(design, disparity, rcond=None)
            a_new, b_new = float(a_new), float(b_new)

        if not np.isfinite(a_new) or a_new <= 1e-6:
            return

        if self._a is None:
            self._a, self._b = a_new, b_new
            self._outlier_streak = 0
            return

        ratio = max(a_new / self._a, self._a / a_new)
        if ratio > OUTLIER_RATIO:
            self._outlier_streak += 1
            if self._outlier_streak < OUTLIER_PATIENCE:
                return  # one bad frame should not move the world
            # Persistently disagreeing: the scene changed, so re-anchor outright.
            self._a, self._b = a_new, b_new
            self._outlier_streak = 0
            return

        self._outlier_streak = 0
        k = self.smoothing
        self._a = k * self._a + (1 - k) * a_new
        self._b = k * self._b + (1 - k) * b_new

    def to_metres(self, disparity: float) -> float:
        """Convert one disparity sample to metres, clamped to a sane range."""
        a = self._a if self._a is not None else self.fallback_scale
        denom = disparity - self._b
        if denom <= 1e-6:
            return 50.0  # effectively "far away"
        return float(np.clip(a / denom, 0.2, 50.0))
