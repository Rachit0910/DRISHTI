"""Frame quality gating — telling the user when the camera cannot see.

This module exists because of an asymmetry that is easy to miss when you design
assistive software with working eyes: a sighted user whose thumb is over the lens
notices instantly. A blind user cannot. And because Drishti's correct behaviour in
an empty corridor is silence, a covered lens and an empty corridor produce exactly
the same output. The user has no way to tell "there is nothing to report" from
"I have been blind for the last four minutes".

So degraded input is not a silent failure. It is an announcement.

Three failure modes are distinguished, because the user's remedy differs:

  occluded - something is physically covering the lens      -> move your hand
  dark     - the scene is too dark to see                   -> turn on a light
  blurred  - the image is out of focus or badly motion-blurred -> hold steadier

Detection is deliberately cheap (a mean, a standard deviation, and one 3x3 Laplacian
on a downsampled frame) so it costs a negligible slice of the frame budget and runs
on CPU without competing with the NPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Mean luma below this (0-255) means the sensor is getting almost no light.
DARK_LUMA = 25.0
# Standard deviation below this means the frame is featureless — a flat field.
FLAT_STD = 8.0
# Variance of the Laplacian below this indicates an image with no sharp edges.
BLUR_VARIANCE = 60.0

# Quality must stay bad/good for this many consecutive frames before we say anything.
# Without debouncing, one dropped or motion-blurred frame triggers a spurious warning.
DEGRADE_FRAMES = 12   # ~0.4 s at 30 fps
RECOVER_FRAMES = 8    # recover faster than we complain

REASON_MESSAGES = {
    "occluded": "Camera is covered.",
    "dark": "Too dark to see.",
    "blurred": "Camera image is blurred.",
}


@dataclass(frozen=True)
class FrameQuality:
    """The verdict on a single frame."""

    ok: bool
    reason: str | None = None
    luma: float = 0.0
    detail: float = 0.0

    @property
    def message(self) -> str | None:
        return REASON_MESSAGES.get(self.reason) if self.reason else None


def _to_luma(frame: np.ndarray) -> np.ndarray:
    """Grayscale, downsampled 4x. Quality assessment does not need full resolution."""
    if frame.ndim == 3:
        # Rec. 601 luma. Channel order does not matter for a magnitude check.
        luma = frame[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], dtype=np.float32)
    else:
        luma = frame.astype(np.float32)
    return luma[::4, ::4]


def _laplacian_variance(luma: np.ndarray) -> float:
    """Variance of a 3x3 Laplacian — the standard cheap focus measure.

    Implemented with array slicing rather than a convolution library so this module
    has no dependency beyond numpy and stays testable anywhere.
    """
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return 0.0
    centre = luma[1:-1, 1:-1]
    lap = (
        luma[:-2, 1:-1] + luma[2:, 1:-1] + luma[1:-1, :-2] + luma[1:-1, 2:] - 4.0 * centre
    )
    return float(lap.var())


def assess(frame: np.ndarray) -> FrameQuality:
    """Classify one frame as usable, or name why it is not."""
    luma = _to_luma(frame)
    mean = float(luma.mean())
    std = float(luma.std())

    # Dark *and* featureless means something is against the lens. Dark but textured
    # is a dim room, which is a different problem with a different remedy.
    if mean < DARK_LUMA and std < FLAT_STD:
        return FrameQuality(ok=False, reason="occluded", luma=mean, detail=std)
    if mean < DARK_LUMA:
        return FrameQuality(ok=False, reason="dark", luma=mean, detail=std)

    # A uniform bright field (lens against a white wall or a bright light) is also
    # occlusion — no information, regardless of brightness.
    if std < FLAT_STD:
        return FrameQuality(ok=False, reason="occluded", luma=mean, detail=std)

    variance = _laplacian_variance(luma)
    if variance < BLUR_VARIANCE:
        return FrameQuality(ok=False, reason="blurred", luma=mean, detail=variance)

    return FrameQuality(ok=True, luma=mean, detail=variance)


@dataclass
class QualityMonitor:
    """Debounces frame verdicts into at most one announcement per state change.

    Emits a message when quality has been bad for DEGRADE_FRAMES consecutive frames,
    and a single "Camera clear." when it recovers — so the user knows the assistant
    is watching again, rather than having to guess whether the silence is meaningful.
    """

    degrade_frames: int = DEGRADE_FRAMES
    recover_frames: int = RECOVER_FRAMES

    _bad_streak: int = 0
    _good_streak: int = 0
    _announced: str | None = field(default=None)

    @property
    def degraded(self) -> bool:
        return self._announced is not None

    def update(self, quality: FrameQuality) -> str | None:
        """Feed one verdict. Returns a message to speak, or None."""
        if not quality.ok:
            self._good_streak = 0
            self._bad_streak += 1
            if self._bad_streak < self.degrade_frames:
                return None
            # Re-announce only if the *kind* of problem changed.
            if self._announced == quality.reason:
                return None
            self._announced = quality.reason
            return quality.message

        self._bad_streak = 0
        self._good_streak += 1
        if self._announced is None or self._good_streak < self.recover_frames:
            return None
        self._announced = None
        return "Camera clear."
