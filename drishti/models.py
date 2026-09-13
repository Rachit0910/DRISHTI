"""Model wrappers: pre-processing, NPU inference, post-processing.

Each wrapper owns exactly one ONNX artifact and knows nothing about the others.
Fusion happens in app.py, scoring happens in policy.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .config import DepthConfig, DetectorConfig
from .geometry import (
    ANCHOR_CLASSES,
    DisparityCalibrator,
    focal_length_pixels,
    pinhole_distance_m,
)
from .runtime import LoadedModel, load_model

log = logging.getLogger(__name__)

# Standard 80-class COCO label set, the output vocabulary of YOLOv8-Detection.
COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]  # normalised x1, y1, x2, y2


def _letterbox(frame: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio, padding the remainder with grey."""
    target_w, target_h = size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))

    try:
        import cv2
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:  # pragma: no cover - cv2 is a hard runtime dep, soft here
        idx_y = (np.arange(new_h) * h // new_h).clip(0, h - 1)
        idx_x = (np.arange(new_w) * w // new_w).clip(0, w - 1)
        resized = frame[idx_y][:, idx_x]

    canvas = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    """Plain greedy non-maximum suppression. Runs on CPU; the NPU does the real work."""
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = (xx2 - xx1).clip(0) * (yy2 - yy1).clip(0)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_threshold]
    return keep


class ObjectDetector:
    """YOLOv8-Detection, quantized to w8a8, running on the Hexagon NPU."""

    def __init__(self, model_path, config: DetectorConfig, *, preferred: str | None = None):
        self.config = config
        self.model: LoadedModel = load_model(model_path, preferred=preferred)

    @property
    def provider(self) -> str:
        return self.model.provider

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        canvas, scale, pad = _letterbox(frame, self.config.input_size)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]  # NCHW
        return np.ascontiguousarray(tensor), scale, pad

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        tensor, scale, pad = self.preprocess(frame)
        outputs = self.model.run({self.model.input_names[0]: tensor})
        return self.postprocess(outputs, frame.shape[:2], scale, pad)

    def postprocess(
        self,
        outputs,
        frame_hw: tuple[int, int],
        scale: float,
        pad: tuple[int, int],
    ) -> list[Detection]:
        """Decode raw YOLOv8 output into normalised, de-letterboxed detections.

        AI Hub's YOLOv8 export emits (boxes, scores, class_ids). If your export
        instead produces the raw (1, 84, 8400) head, decode it here — the shape check
        below tells you which one you have.
        """
        if len(outputs) >= 3:
            boxes, scores, class_ids = outputs[0], outputs[1], outputs[2]
            boxes = np.asarray(boxes).reshape(-1, 4)
            scores = np.asarray(scores).reshape(-1)
            class_ids = np.asarray(class_ids).reshape(-1).astype(int)
        else:
            raw = np.asarray(outputs[0])
            if raw.ndim == 3 and raw.shape[1] in (84, 85):
                raw = raw[0].T  # (8400, 84)
            cx, cy, w, h = raw[:, 0], raw[:, 1], raw[:, 2], raw[:, 3]
            cls_scores = raw[:, 4:]
            class_ids = cls_scores.argmax(axis=1)
            scores = cls_scores.max(axis=1)
            boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

        keep_mask = scores >= self.config.confidence_threshold
        boxes, scores, class_ids = boxes[keep_mask], scores[keep_mask], class_ids[keep_mask]
        if len(boxes) == 0:
            return []

        keep = _nms(boxes, scores, self.config.iou_threshold)[: self.config.max_detections]

        pad_x, pad_y = pad
        frame_h, frame_w = frame_hw
        results: list[Detection] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            # Undo letterbox padding and scaling, then normalise to [0, 1].
            x1 = (x1 - pad_x) / scale / frame_w
            x2 = (x2 - pad_x) / scale / frame_w
            y1 = (y1 - pad_y) / scale / frame_h
            y2 = (y2 - pad_y) / scale / frame_h
            cid = int(class_ids[i])
            label = COCO_LABELS[cid] if 0 <= cid < len(COCO_LABELS) else f"class_{cid}"
            results.append(
                Detection(
                    label=label,
                    confidence=float(scores[i]),
                    bbox=(
                        float(np.clip(x1, 0, 1)),
                        float(np.clip(y1, 0, 1)),
                        float(np.clip(x2, 0, 1)),
                        float(np.clip(y2, 0, 1)),
                    ),
                )
            )
        return results


class DepthEstimator:
    """Midas-V2 disparity, calibrated to metres using detections as anchors."""

    def __init__(self, model_path, config: DepthConfig, *, preferred: str | None = None):
        self.config = config
        self.model: LoadedModel = load_model(model_path, preferred=preferred)
        self.calibrator = DisparityCalibrator(
            smoothing=config.calibration_smoothing,
            fallback_scale=config.fallback_scale,
        )
        self._focal_px: float | None = None

    @property
    def provider(self) -> str:
        return self.model.provider

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        canvas, _, _ = _letterbox(frame, self.config.input_size)
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        outputs = self.model.run({self.model.input_names[0]: np.ascontiguousarray(tensor)})
        return np.asarray(outputs[0]).squeeze()

    def sample(self, disparity_map: np.ndarray, bbox: tuple[float, float, float, float]) -> float:
        """Median disparity inside a normalised bbox — median resists edge artefacts."""
        h, w = disparity_map.shape[:2]
        x1 = int(np.clip(bbox[0] * w, 0, w - 1))
        x2 = int(np.clip(bbox[2] * w, x1 + 1, w))
        y1 = int(np.clip(bbox[1] * h, 0, h - 1))
        y2 = int(np.clip(bbox[3] * h, y1 + 1, h))
        patch = disparity_map[y1:y2, x1:x2]
        return float(np.median(patch)) if patch.size else 0.0

    def distances(
        self,
        disparity_map: np.ndarray,
        detections: list[Detection],
        frame_hw: tuple[int, int],
    ) -> list[float]:
        """Metric distance per detection, recalibrating from this frame's anchors."""
        frame_h, frame_w = frame_hw
        if self._focal_px is None:
            self._focal_px = focal_length_pixels(frame_h)

        anchors: list[tuple[float, float]] = []
        samples: list[float] = []
        for det in detections:
            disparity = self.sample(disparity_map, det.bbox)
            samples.append(disparity)
            if det.label in ANCHOR_CLASSES:
                bbox_h_px = (det.bbox[3] - det.bbox[1]) * frame_h
                metric = pinhole_distance_m(det.label, bbox_h_px, self._focal_px)
                if metric is not None:
                    anchors.append((disparity, metric))

        self.calibrator.update(anchors)
        return [self.calibrator.to_metres(d) for d in samples]


class TextReader:
    """EasyOCR detector + recognizer, run on demand rather than every frame."""

    def __init__(self, detector_path, recognizer_path, *, preferred: str | None = None):
        self.detector: LoadedModel = load_model(detector_path, preferred=preferred)
        self.recognizer: LoadedModel = load_model(recognizer_path, preferred=preferred)

    @property
    def provider(self) -> str:
        return self.detector.provider

    def __call__(self, frame: np.ndarray) -> str:
        """Return everything readable in frame as one string.

        The recognizer's character decoding depends on the charset baked into the
        exported artifact, so `scripts/export_models.py` writes the matching charset
        alongside it. Wire the greedy CTC decode here once you have that file.
        """
        canvas, _, _ = _letterbox(frame, (800, 608))
        tensor = canvas.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        _regions = self.detector.run({self.detector.input_names[0]: np.ascontiguousarray(tensor)})
        raise NotImplementedError(
            "OCR decode is not implemented. Crop each region returned above, run "
            "self.recognizer on it, and greedy CTC-decode the result. This needs the "
            "character set that matches the exported recognizer artifact, which "
            "scripts/export_models.py does not currently write out — exporting the "
            "charset is part of the same piece of work. See README, 'Finishing the "
            "OCR path'."
        )
