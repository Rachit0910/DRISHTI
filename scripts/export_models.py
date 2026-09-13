"""Compile, quantize and profile every Drishti model on real Snapdragon hardware.

This is the script that makes the submission's benchmark numbers real rather than
estimated. Qualcomm AI Hub provisions an actual physical device from its device farm,
runs the model on it, and returns measured latency, memory and compute-unit split —
so the numbers come from the target silicon even before you hold the laptop.

    pip install qai-hub qai-hub-models
    qai-hub configure --api_token <token>     # from aihub.qualcomm.com
    python scripts/export_models.py --all

Artifacts land in models/ and the profile JSON in benchmarks/.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from drishti.config import AI_HUB_DEVICE  # noqa: E402

MODEL_DIR = REPO_ROOT / "models"
BENCH_DIR = REPO_ROOT / "benchmarks"


@dataclass(frozen=True)
class ModelSpec:
    """One model to pull from AI Hub, quantize, profile and download."""

    key: str
    hub_model: str          # qai_hub_models package name
    output_name: str        # filename written into models/
    input_shape: tuple[int, ...]
    quantize: bool = True

    @property
    def precision(self) -> str:
        return "w8a8" if self.quantize else "float"


SPECS: dict[str, ModelSpec] = {
    "detector": ModelSpec(
        key="detector",
        hub_model="yolov8_det",
        output_name="yolov8_det_w8a8.onnx",
        input_shape=(1, 3, 640, 640),
    ),
    "depth": ModelSpec(
        key="depth",
        hub_model="midas",
        output_name="midas_v2_w8a8.onnx",
        input_shape=(1, 3, 256, 256),
    ),
    "ocr": ModelSpec(
        key="ocr",
        hub_model="easyocr",
        output_name="easyocr_detector.onnx",
        input_shape=(1, 3, 608, 800),
        # OCR is duty-cycled rather than per-frame, so float is acceptable and
        # avoids a quantization accuracy hit on small text. Flip this to compare.
        quantize=False,
    ),
}


def export(spec: ModelSpec, device_name: str) -> dict:
    """Compile -> profile -> download one model. Returns the profile summary."""
    import qai_hub as hub
    from importlib import import_module

    print(f"\n=== {spec.key}: {spec.hub_model} ({spec.precision}) ===")

    module = import_module(f"qai_hub_models.models.{spec.hub_model}")
    torch_model = module.Model.from_pretrained()

    device = hub.Device(device_name)
    options = f"--target_runtime onnx --quantize_full_type {spec.precision}" if spec.quantize \
        else "--target_runtime onnx"

    print(f"  compiling for {device_name} ...")
    compile_job = hub.submit_compile_job(
        model=torch_model,
        device=device,
        input_specs={"image": spec.input_shape},
        options=options,
    )
    target_model = compile_job.get_target_model()

    print("  profiling on a real device from the AI Hub device farm ...")
    profile_job = hub.submit_profile_job(model=target_model, device=device)
    profile = profile_job.download_profile()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_model.download(str(MODEL_DIR / spec.output_name))
    print(f"  wrote models/{spec.output_name}")

    summary = summarize(spec, device_name, profile)
    print(
        f"  {summary['inference_ms']:.3f} ms  |  peak {summary['peak_memory_mb']} MB"
        f"  |  {summary['npu_layers']}/{summary['total_layers']} layers on NPU"
    )
    return summary


def summarize(spec: ModelSpec, device_name: str, profile: dict) -> dict:
    """Pull the handful of numbers the submission actually reports."""
    execution = profile.get("execution_summary", {})
    detail = profile.get("execution_detail", []) or []

    def unit_of(layer: dict) -> str:
        return (layer.get("compute_unit") or layer.get("computeUnit") or "").upper()

    npu_layers = sum(1 for layer in detail if unit_of(layer) == "NPU")
    micros = execution.get("estimated_inference_time") or 0
    peak_bytes = execution.get("estimated_inference_peak_memory") or 0

    return {
        "model": spec.key,
        "hub_model": spec.hub_model,
        "precision": spec.precision,
        "device": device_name,
        "inference_ms": micros / 1000.0,
        "peak_memory_mb": round(peak_bytes / (1024 * 1024), 1),
        "npu_layers": npu_layers,
        "total_layers": len(detail),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="export every model")
    parser.add_argument("--model", choices=sorted(SPECS), action="append", default=[])
    parser.add_argument("--device", default=AI_HUB_DEVICE)
    args = parser.parse_args()

    selected = sorted(SPECS) if args.all else args.model
    if not selected:
        parser.error("pass --all or --model <name>")

    try:
        import qai_hub  # noqa: F401
    except ImportError:
        raise SystemExit(
            "qai-hub is not installed.\n"
            "    pip install qai-hub qai-hub-models\n"
            "    qai-hub configure --api_token <token from aihub.qualcomm.com>"
        )

    results = [export(SPECS[key], args.device) for key in selected]

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    out = BENCH_DIR / "aihub_profile.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out.relative_to(REPO_ROOT)}")

    total = sum(r["inference_ms"] for r in results if r["model"] != "ocr")
    budget = 1000.0 / 30.0
    print(f"\nPer-frame (tier A) NPU time: {total:.2f} ms of a {budget:.1f} ms budget "
          f"({total / budget * 100:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
