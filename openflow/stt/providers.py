"""ONNX Runtime execution-provider selection.

The default ``onnxruntime`` wheel is CPU-only, which is what
``onnx-asr[cpu,hub]`` pulls in. A GPU shows up here only if the user installs a
different runtime package in its place:

    pip uninstall onnxruntime
    pip install onnxruntime-directml     # any DirectX 12 GPU -- NVIDIA/AMD/Intel
    pip install onnxruntime-gpu          # NVIDIA only, needs CUDA + cuDNN

DirectML is the better default suggestion on Windows: no CUDA toolkit to
install, and it covers integrated graphics as well as discrete cards.

TensorRT is deliberately not in the preference order. It builds an optimized
engine on first use, which costs minutes -- the wrong trade for a tool whose
whole promise is that the text is there when you let go of the key.

Measured, August 2026, RTX 4070 Ti, parakeet-tdt-0.6b-v3 int8, 9.9 s of
synthesized speech, median of 5 (``scripts/bench_providers.py``):

    DmlExecutionProvider    3258 ms
    CPUExecutionProvider    2609 ms   (onnxruntime-directml wheel)
    CPUExecutionProvider    3172 ms   (stock onnxruntime wheel)

So the GPU is **not** a win here, and "auto" therefore means CPU. Two reasons,
both structural rather than a tuning problem:

  * A TDT transducer decodes one step per emitted token, sequentially. That is
    bound by per-op dispatch latency, not by arithmetic throughput, which is
    the only thing a GPU would win.
  * int8 weights are the CPU's home turf (VNNI); a GPU path may dequantize or
    fall back per-op, paying copies at every boundary.

Run the benchmark before believing a GPU will help on your machine. Enabling
one is a deliberate ``stt.device`` setting, not something inferred.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# What "auto" picks. CPU, because that is what measured fastest -- see the
# module docstring. This is not a claim that no GPU can ever win, only that
# switching to one silently would be a regression on the hardware we measured.
AUTO_PREFERENCE = ("CPUExecutionProvider",)

# Everything selectable by name, fastest-plausible first. Only reachable by
# setting stt.device explicitly.
PREFERRED = (
    "CUDAExecutionProvider",     # NVIDIA, onnxruntime-gpu
    "DmlExecutionProvider",      # DirectML, any DX12 GPU on Windows
    "CoreMLExecutionProvider",   # Apple silicon
    "CPUExecutionProvider",
)

# Present in the stock wheel but not an accelerator; selecting it would send
# inference to a remote endpoint, which this app must never do silently.
NEVER = frozenset({"AzureExecutionProvider"})

_ALIASES = {
    "auto": None,
    "cpu": "CPUExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "directml": "DmlExecutionProvider",
    "dml": "DmlExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
}


def available_providers() -> list[str]:
    try:
        import onnxruntime
    except ImportError:
        return []
    return [p for p in onnxruntime.get_available_providers() if p not in NEVER]


def select_providers(device: str = "auto") -> list[str]:
    """Return an onnxruntime provider list for ``device``.

    ``auto`` resolves to CPU, which is what benchmarked fastest for this model
    (see the module docstring) -- a GPU has to be asked for by name. Any
    explicit provider keeps CPU behind it as a fallback, so a GPU that fails to
    initialize degrades to working software rather than to a crash, and a
    device that is not installed warns instead of raising: a stale config value
    should not cost someone their dictation.
    """
    installed = available_providers()
    if not installed:
        return ["CPUExecutionProvider"]

    key = (device or "auto").strip().lower()
    if key not in _ALIASES:
        log.warning("unknown stt.device %r; using auto", device)
        key = "auto"

    wanted = _ALIASES[key]
    if wanted is None:
        chosen = [p for p in AUTO_PREFERENCE if p in installed]
        if not chosen:
            chosen = [installed[0]]
    elif wanted in installed:
        chosen = [wanted]
    else:
        log.warning(
            "stt.device=%r but %s is not installed (available: %s); using CPU. "
            "See openflow/stt/providers.py for how to enable GPU inference.",
            device, wanted, ", ".join(installed),
        )
        chosen = ["CPUExecutionProvider"]

    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return chosen


def describe(providers: list[str]) -> str:
    """One-line summary for the log, naming the accelerator actually in use."""
    head = providers[0] if providers else "CPUExecutionProvider"
    if head == "CPUExecutionProvider":
        return "CPU (install onnxruntime-directml or onnxruntime-gpu for GPU inference)"
    return f"{head.replace('ExecutionProvider', '')} with CPU fallback"
