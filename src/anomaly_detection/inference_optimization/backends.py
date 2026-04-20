from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from src.anomaly_detection.inference_optimization.data import PreparedBatch


@dataclass
class BackendResult:
    timings_sec: list[float]
    pred_scores: list[np.ndarray]
    anomaly_maps: list[np.ndarray]

    @property
    def mean_sec(self) -> float:
        return float(np.mean(self.timings_sec))

    @property
    def std_sec(self) -> float:
        return float(np.std(self.timings_sec))


class InferenceBackend(Protocol):
    name: str

    def run(self, batches: list[PreparedBatch], *, warmup_runs: int, repeat_runs: int) -> BackendResult:
        ...


class TorchBackend:
    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        device: torch.device,
        use_amp: bool = False,
        compile_model: bool = False,
        compile_mode: str = "reduce-overhead",
    ):
        self.name = name
        self.device = device
        self.use_amp = use_amp and device.type == "cuda"
        self.model = model
        if compile_model:
            self.model = torch.compile(self.model, mode=compile_mode)
        self.model.eval()

    def run(self, batches: list[PreparedBatch], *, warmup_runs: int, repeat_runs: int) -> BackendResult:
        with torch.inference_mode():
            for _ in range(warmup_runs):
                self._run_once(batches, collect=False)
            timings = []
            for _ in range(repeat_runs):
                _sync(self.device)
                start = time.perf_counter()
                self._run_once(batches, collect=False)
                _sync(self.device)
                timings.append(time.perf_counter() - start)
            pred_scores, anomaly_maps = self._run_once(batches, collect=True)
        return BackendResult(timings, pred_scores, anomaly_maps)

    def _run_once(self, batches: list[PreparedBatch], *, collect: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pred_scores: list[np.ndarray] = []
        anomaly_maps: list[np.ndarray] = []
        for batch in batches:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
                scores, maps = self.model(batch.images)
            if collect:
                pred_scores.append(scores.detach().float().cpu().numpy())
                anomaly_maps.append(maps.detach().float().cpu().numpy())
        return pred_scores, anomaly_maps


class OnnxRuntimeBackend:
    def __init__(
        self,
        *,
        name: str,
        model: torch.nn.Module,
        sample_input: torch.Tensor,
        export_dir: Path,
        providers: list[str | tuple[str, dict]],
        use_cuda_iobinding: bool,
    ):
        self.name = name
        self.sample_input = sample_input
        self.use_cuda_iobinding = use_cuda_iobinding and sample_input.device.type == "cuda"
        self.onnx_path = export_dir / f"{name}.onnx"
        export_dir.mkdir(parents=True, exist_ok=True)
        _export_onnx(model, sample_input, self.onnx_path)

        if _uses_tensorrt(providers):
            _load_tensorrt_shared_libraries()

        import onnxruntime as ort

        available = set(ort.get_available_providers())
        requested = [p[0] if isinstance(p, tuple) else p for p in providers]
        missing = [p for p in requested if p not in available]
        if missing:
            raise RuntimeError(f"ONNX Runtime providers are not available: missing={missing}, available={sorted(available)}")
        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        actual_providers = self.session.get_providers()
        if _uses_tensorrt(providers) and "TensorrtExecutionProvider" not in actual_providers:
            raise RuntimeError(
                "TensorRT Execution Provider was requested but ONNX Runtime did not enable it. "
                f"actual_providers={actual_providers}"
            )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

    def run(self, batches: list[PreparedBatch], *, warmup_runs: int, repeat_runs: int) -> BackendResult:
        for _ in range(warmup_runs):
            self._run_once(batches, collect=False)
        timings = []
        for _ in range(repeat_runs):
            if self.sample_input.device.type == "cuda":
                torch.cuda.synchronize(self.sample_input.device)
            start = time.perf_counter()
            self._run_once(batches, collect=False)
            if self.sample_input.device.type == "cuda":
                torch.cuda.synchronize(self.sample_input.device)
            timings.append(time.perf_counter() - start)
        pred_scores, anomaly_maps = self._run_once(batches, collect=True)
        return BackendResult(timings, pred_scores, anomaly_maps)

    def _run_once(self, batches: list[PreparedBatch], *, collect: bool) -> tuple[list[np.ndarray], list[np.ndarray]]:
        pred_scores: list[np.ndarray] = []
        anomaly_maps: list[np.ndarray] = []
        for batch in batches:
            if self.use_cuda_iobinding and not collect:
                binding = self.session.io_binding()
                image = batch.images.contiguous()
                binding.bind_input(
                    name=self.input_name,
                    device_type="cuda",
                    device_id=image.device.index or 0,
                    element_type=np.float32,
                    shape=tuple(image.shape),
                    buffer_ptr=image.data_ptr(),
                )
                for output_name in self.output_names:
                    binding.bind_output(output_name, device_type="cuda", device_id=image.device.index or 0)
                self.session.run_with_iobinding(binding)
                continue

            outputs = self.session.run(self.output_names, {self.input_name: batch.images.detach().cpu().numpy().astype(np.float32, copy=False)})
            if collect:
                pred_scores.append(np.asarray(outputs[0]))
                anomaly_maps.append(np.asarray(outputs[1]))
        return pred_scores, anomaly_maps


def build_backend(
    *,
    variant: str,
    model: torch.nn.Module,
    device: torch.device,
    sample_input: torch.Tensor,
    export_dir: Path,
    torch_compile_mode: str,
) -> InferenceBackend:
    if variant == "baseline_eager_fp32":
        return TorchBackend(name=variant, model=model, device=device)
    if variant == "eager_amp_fp16":
        return TorchBackend(name=variant, model=model, device=device, use_amp=True)
    if variant == "torch_compile_fp32":
        return TorchBackend(name=variant, model=model, device=device, compile_model=True, compile_mode=torch_compile_mode)
    if variant == "torch_compile_amp_fp16":
        return TorchBackend(name=variant, model=model, device=device, use_amp=True, compile_model=True, compile_mode=torch_compile_mode)
    if variant == "onnxruntime_cuda_fp32":
        return OnnxRuntimeBackend(
            name=variant,
            model=model,
            sample_input=sample_input,
            export_dir=export_dir,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            use_cuda_iobinding=True,
        )
    if variant == "onnxruntime_tensorrt_fp32":
        return OnnxRuntimeBackend(
            name=variant,
            model=model,
            sample_input=sample_input,
            export_dir=export_dir,
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
            use_cuda_iobinding=True,
        )
    if variant == "onnxruntime_tensorrt_fp16":
        return OnnxRuntimeBackend(
            name=variant,
            model=model,
            sample_input=sample_input,
            export_dir=export_dir,
            providers=[
                ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            use_cuda_iobinding=True,
        )
    raise ValueError(f"unsupported inference variant: {variant}")


def _export_onnx(model: torch.nn.Module, sample_input: torch.Tensor, onnx_path: Path) -> None:
    model.eval()
    with torch.inference_mode():
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            input_names=["image"],
            output_names=["pred_score", "anomaly_map"],
            dynamic_axes={
                "image": {0: "batch"},
                "pred_score": {0: "batch"},
                "anomaly_map": {0: "batch"},
            },
            opset_version=18,
            dynamo=False,
        )


def _uses_tensorrt(providers: list[str | tuple[str, dict]]) -> bool:
    return any((provider[0] if isinstance(provider, tuple) else provider) == "TensorrtExecutionProvider" for provider in providers)


def _load_tensorrt_shared_libraries() -> None:
    """Make TensorRT wheel libraries visible to ONNX Runtime's TensorRT EP.

    The `tensorrt-cu12` wheel installs shared libraries under site-packages,
    but that directory is not necessarily in LD_LIBRARY_PATH inside Docker.
    Loading them with RTLD_GLOBAL before creating the ORT session lets
    libonnxruntime_providers_tensorrt.so resolve libnvinfer.so.*.
    """
    import ctypes
    import importlib.util

    spec = importlib.util.find_spec("tensorrt_libs")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("tensorrt_libs package is not installed. Run `uv sync` after updating dependencies.")
    lib_dir = Path(next(iter(spec.submodule_search_locations)))
    libs = sorted(lib_dir.glob("*.so*"))
    if not libs:
        raise RuntimeError(f"No TensorRT shared libraries found under {lib_dir}")

    priority = ("libnvinfer.so", "libnvinfer_plugin.so", "libnvonnxparser.so")
    ordered = sorted(libs, key=lambda path: next((i for i, name in enumerate(priority) if path.name.startswith(name)), len(priority)))
    failures = []
    for lib_path in ordered:
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as exc:
            failures.append(f"{lib_path.name}: {exc}")
    if failures:
        raise RuntimeError("Failed to load TensorRT shared libraries: " + "; ".join(failures[:5]))


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
