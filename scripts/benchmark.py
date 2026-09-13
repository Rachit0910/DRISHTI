"""End-to-end benchmark: NPU versus CPU, measured on this machine.

AI Hub's device farm gives you per-model inference latency. This gives you the number
that actually matters to a user: sustained end-to-end frames per second including
capture, pre-processing, NMS and fusion — and the honest gap between the two.

    python scripts/benchmark.py --frames 300
    python scripts/benchmark.py --frames 300 --compare-cpu
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drishti.config import DEPTH_ONNX, DETECTOR_ONNX, DepthConfig, DetectorConfig  # noqa: E402
from drishti.models import DepthEstimator, ObjectDetector  # noqa: E402
from drishti.policy import NarrationPolicy  # noqa: E402
from drishti.app import build_observations  # noqa: E402

BENCH_DIR = REPO_ROOT / "benchmarks"


def synthetic_frame(rng: np.random.Generator, size=(720, 1280)) -> np.ndarray:
    """A deterministic stand-in so the harness runs without a camera attached."""
    return rng.integers(0, 255, size=(*size, 3), dtype=np.uint8)


def measure(provider: str | None, frames: int, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    detector = ObjectDetector(DETECTOR_ONNX, DetectorConfig(), preferred=provider)
    depth = DepthEstimator(DEPTH_ONNX, DepthConfig(), preferred=provider)
    policy = NarrationPolicy()

    stage_times: dict[str, list[float]] = {"detect": [], "depth": [], "fuse": []}
    end_to_end: list[float] = []

    # Warm up: first inference includes graph finalization on the HTP.
    warmup = synthetic_frame(rng)
    detector(warmup)
    depth(warmup)

    for i in range(frames):
        frame = synthetic_frame(rng)
        loop_start = time.perf_counter()

        t0 = time.perf_counter()
        detections = detector(frame)
        stage_times["detect"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        disparity = depth(frame)
        distances = depth.distances(disparity, detections, frame.shape[:2])
        stage_times["depth"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        observations = build_observations(detections, distances)
        policy.tick(observations, now=float(i) / 30.0)
        stage_times["fuse"].append((time.perf_counter() - t0) * 1000)

        end_to_end.append((time.perf_counter() - loop_start) * 1000)

    def stats(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "mean_ms": round(statistics.fmean(values), 3),
            "median_ms": round(statistics.median(values), 3),
            "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 3),
        }

    return {
        "provider": detector.provider,
        "frames": frames,
        "stages": {name: stats(v) for name, v in stage_times.items()},
        "end_to_end": stats(end_to_end),
        "sustained_fps": round(1000.0 / statistics.fmean(end_to_end), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--compare-cpu", action="store_true",
                        help="also run a CPUExecutionProvider baseline for the speedup figure")
    args = parser.parse_args()

    results = {"default": measure(None, args.frames)}
    print(f"\n{results['default']['provider']}: "
          f"{results['default']['sustained_fps']} fps sustained "
          f"({results['default']['end_to_end']['mean_ms']} ms/frame mean)")

    if args.compare_cpu:
        results["cpu_baseline"] = measure("CPUExecutionProvider", args.frames)
        print(f"CPUExecutionProvider: {results['cpu_baseline']['sustained_fps']} fps sustained "
              f"({results['cpu_baseline']['end_to_end']['mean_ms']} ms/frame mean)")
        speedup = (results["cpu_baseline"]["end_to_end"]["mean_ms"]
                   / results["default"]["end_to_end"]["mean_ms"])
        results["speedup_vs_cpu"] = round(speedup, 2)
        print(f"\nSpeedup: {speedup:.2f}x")

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "end_to_end.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
