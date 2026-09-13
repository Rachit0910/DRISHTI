"""Central configuration. Everything tunable lives here, nothing tunable lives elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "models"

# Artifact names written by scripts/export_models.py.
DETECTOR_ONNX = MODEL_DIR / "yolov8_det_w8a8.onnx"
DEPTH_ONNX = MODEL_DIR / "midas_v2_w8a8.onnx"
OCR_DETECTOR_ONNX = MODEL_DIR / "easyocr_detector.onnx"
OCR_RECOGNIZER_ONNX = MODEL_DIR / "easyocr_recognizer.onnx"

# AI Hub device string used for cloud compile/profile jobs.
# "Snapdragon X Elite CRD" is the Windows-on-Snapdragon compute reference design.
AI_HUB_DEVICE = "Snapdragon X Elite CRD"


@dataclass(frozen=True)
class ScheduleConfig:
    """How often each tier runs.

    Tier A is cheap enough to run every frame (detection + depth together are around
    1.3 ms of NPU time on X2 Elite). Tier B is an order of magnitude heavier and is
    only useful when the user asks for it, so it is event-driven, not periodic.
    """

    target_fps: int = 30
    detect_every_n_frames: int = 1
    depth_every_n_frames: int = 1
    # OCR is user-triggered (hotkey / voice). This is only a rate limit on that.
    ocr_min_interval_s: float = 0.5


@dataclass(frozen=True)
class DetectorConfig:
    input_size: tuple[int, int] = (640, 640)
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.50
    max_detections: int = 20


@dataclass(frozen=True)
class DepthConfig:
    input_size: tuple[int, int] = (256, 256)
    # Exponential smoothing on the fitted disparity->metres transform.
    calibration_smoothing: float = 0.85
    # Fallback scale used before any anchor object has been seen.
    fallback_scale: float = 8.0


@dataclass(frozen=True)
class SpeechConfig:
    rate_wpm: int = 190
    volume: float = 1.0
    # Locale for both the phrasebook and the system voice. "hi" for Hindi.
    voice_hint: str = "en"


@dataclass(frozen=True)
class AppConfig:
    camera_index: int = 0
    locale: str = "en"
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    speech: SpeechConfig = field(default_factory=SpeechConfig)
    # Set by --force-provider on the benchmark harness to get a CPU baseline.
    force_provider: str | None = None
