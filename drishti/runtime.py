"""ONNX Runtime session management with an explicit NPU -> GPU -> CPU fallback chain.

The whole premise of Drishti is that inference runs on the Snapdragon Hexagon NPU.
But a submission that only works on one machine is a demo, not a product, so every
model is loaded through this module, which walks a provider chain and records which
execution provider actually accepted the graph. That record is what the benchmark
harness reports and what the UI surfaces to the user.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "onnxruntime is required. On Snapdragon/Windows-on-ARM install the QNN build:\n"
        "    pip install onnxruntime-qnn\n"
        "On other platforms the CPU build is enough for development:\n"
        "    pip install onnxruntime"
    ) from exc

log = logging.getLogger(__name__)


# Ordered best-to-worst. Each entry is (provider_name, provider_options).
# QNN targets the Hexagon NPU; DirectML targets the Adreno GPU; CPU is the floor.
PROVIDER_CHAIN: list[tuple[str, dict[str, Any]]] = [
    (
        "QNNExecutionProvider",
        {
            # QnnHtp.dll is the Hexagon Tensor Processor backend shipped with
            # onnxruntime-qnn. "burst" asks the HTP for maximum clocks, which is what
            # we want for a latency-sensitive, duty-cycled perception loop.
            "backend_path": "QnnHtp.dll",
            "htp_performance_mode": "burst",
            "htp_graph_finalization_optimization_mode": "3",
        },
    ),
    ("DmlExecutionProvider", {}),
    ("CPUExecutionProvider", {}),
]


def context_cache_path(model_path: Path) -> Path:
    """Where the pre-compiled QNN context binary for this model lives."""
    return model_path.with_name(model_path.stem + "_ctx.onnx")


@dataclass(frozen=True)
class LoadedModel:
    """An ORT session plus the provenance we need for honest benchmarking."""

    session: "ort.InferenceSession"
    provider: str
    model_path: Path
    load_seconds: float
    from_context_cache: bool = False

    @property
    def on_npu(self) -> bool:
        return self.provider == "QNNExecutionProvider"

    @property
    def input_names(self) -> list[str]:
        return [i.name for i in self.session.get_inputs()]

    @property
    def output_names(self) -> list[str]:
        return [o.name for o in self.session.get_outputs()]

    def run(self, feeds: dict[str, np.ndarray]) -> Sequence[np.ndarray]:
        return self.session.run(self.output_names, feeds)


def available_providers() -> list[str]:
    return list(ort.get_available_providers())


def load_model(
    model_path: str | Path,
    *,
    preferred: str | None = None,
    intra_op_threads: int = 2,
    use_context_cache: bool = True,
) -> LoadedModel:
    """Load `model_path`, walking PROVIDER_CHAIN until one accepts the graph.

    Args:
        model_path: path to the .onnx artifact produced by scripts/export_models.py.
        preferred: pin a specific provider (used by the benchmark harness to force a
            CPU-only baseline for the NPU-vs-CPU comparison).
        intra_op_threads: kept low because the NPU does the work and we do not want
            the CPU pool competing with camera capture and speech synthesis.
        use_context_cache: on QNN, compile the graph once and reuse the cached
            context binary on every later launch. See below.

    Raises:
        RuntimeError: if no provider in the chain could load the model.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found. Run: python scripts/export_models.py --all"
        )

    chain = PROVIDER_CHAIN
    if preferred is not None:
        chain = [(name, opts) for name, opts in PROVIDER_CHAIN if name == preferred]
        if not chain:
            raise ValueError(f"{preferred!r} is not in the known provider chain")

    installed = set(available_providers())
    errors: list[str] = []

    for name, options in chain:
        if name not in installed:
            errors.append(f"{name}: not installed in this onnxruntime build")
            continue

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = intra_op_threads
        sess_options.log_severity_level = 3

        # --- QNN context binary caching -------------------------------------
        # Finalizing a graph on the Hexagon Tensor Processor is expensive, and by
        # default it happens on every single launch. For an assistive app that cost
        # is paid at exactly the wrong moment: the user has pressed the button and
        # is standing still, waiting, unable to see a progress indicator.
        #
        # ORT can dump the compiled QNN context to disk once and load it directly
        # afterwards, which turns graph finalization into a file read. We therefore
        # load the cached context when it exists, and write it when it does not.
        source = model_path
        from_cache = False
        if name == "QNNExecutionProvider" and use_context_cache:
            cache = context_cache_path(model_path)
            if cache.exists():
                source, from_cache = cache, True
            else:
                sess_options.add_session_config_entry("ep.context_enable", "1")
                sess_options.add_session_config_entry("ep.context_file_path", str(cache))
                sess_options.add_session_config_entry("ep.context_embed_mode", "1")
                log.info("No context cache for %s — compiling and caching", model_path.name)

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(
                str(source),
                sess_options=sess_options,
                providers=[name],
                provider_options=[options],
            )
        except Exception as exc:  # noqa: BLE001 - we genuinely want to try the next one
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            log.warning("Provider %s rejected %s, falling back", name, source.name)
            continue

        elapsed = time.perf_counter() - started
        actual = session.get_providers()[0]
        log.info(
            "Loaded %s on %s in %.2fs%s",
            source.name, actual, elapsed, " (context cache)" if from_cache else "",
        )
        return LoadedModel(
            session=session,
            provider=actual,
            model_path=model_path,
            load_seconds=elapsed,
            from_context_cache=from_cache,
        )

    raise RuntimeError(
        f"No execution provider could load {model_path}:\n  " + "\n  ".join(errors)
    )
