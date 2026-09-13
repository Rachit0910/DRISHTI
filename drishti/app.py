"""Drishti main loop: capture -> NPU inference -> fusion -> policy -> speech.

Run with:
    python -m drishti.app
    python -m drishti.app --force-provider CPUExecutionProvider   # CPU baseline
"""

from __future__ import annotations

import argparse
import logging
import time

from .config import AppConfig, SpeechConfig, DEPTH_ONNX, DETECTOR_ONNX
from .models import DepthEstimator, ObjectDetector
from .phrases import get as get_phrasebook
from .policy import NarrationPolicy, Observation, Utterance
from .quality import QualityMonitor, assess
from .speech import Speaker

log = logging.getLogger(__name__)


def build_observations(detections, distances) -> list[Observation]:
    """Fuse detector boxes with per-object metric distance."""
    return [
        Observation(
            label=det.label,
            confidence=det.confidence,
            bbox=det.bbox,
            distance_m=distance,
        )
        for det, distance in zip(detections, distances)
    ]


def run(config: AppConfig) -> int:
    try:
        import cv2
    except ImportError:
        raise SystemExit("opencv-python is required for camera capture: pip install opencv-python")

    detector = ObjectDetector(DETECTOR_ONNX, config.detector, preferred=config.force_provider)
    depth = DepthEstimator(DEPTH_ONNX, config.depth, preferred=config.force_provider)

    log.info("detector on %s | depth on %s", detector.provider, depth.provider)
    if not (detector.model.on_npu and depth.model.on_npu):
        log.warning(
            "Not running on the Hexagon NPU — falling back to %s. Expect higher "
            "latency and much higher power draw.", detector.provider
        )

    phrases = get_phrasebook(config.locale)
    policy = NarrationPolicy(phrases=phrases)
    quality = QualityMonitor()
    speaker = Speaker(config.speech)

    camera = cv2.VideoCapture(config.camera_index)
    if not camera.isOpened():
        raise SystemExit(f"could not open camera {config.camera_index}")

    frame_index = 0
    schedule = config.schedule
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                log.error("camera read failed")
                break

            frame_index += 1
            detections = []
            distances = []

            # A blind user cannot see that the lens is covered, and Drishti's correct
            # behaviour in an empty room is silence — so degraded input has to be
            # announced, or the user cannot tell the two situations apart.
            verdict = assess(frame)
            message = quality.update(verdict)
            if message is not None:
                speaker.say(Utterance(text=message, salience=1.0, interrupt=True))
            if not verdict.ok:
                continue

            if frame_index % schedule.detect_every_n_frames == 0:
                detections = detector(frame)

            if detections and frame_index % schedule.depth_every_n_frames == 0:
                disparity = depth(frame)
                distances = depth.distances(disparity, detections, frame.shape[:2])

            if detections and distances:
                observations = build_observations(detections, distances)
                for utterance in policy.tick(observations, now=time.monotonic()):
                    log.info("say: %s (salience %.2f)", utterance.text, utterance.salience)
                    speaker.say(utterance)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        speaker.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Drishti on-device visual assistant")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--locale", default="en", help="speech locale, e.g. en or hi")
    parser.add_argument(
        "--force-provider",
        default=None,
        help="Pin an execution provider, e.g. CPUExecutionProvider for a baseline run",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )
    config = AppConfig(
        camera_index=args.camera,
        locale=args.locale,
        force_provider=args.force_provider,
        speech=SpeechConfig(voice_hint=args.locale),
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
