"""Unified detection backends for apples-to-apples consistency comparison.

Every backend exposes the same interface::

    det = Detector.create("onnx", path, device, num_classes)
    raw = det.forward(lb.image)        # -> np.ndarray (1, 4+nc, N) in letterboxed px
    boxes, scores, cls = decode_and_nms(raw, cfg)

Pre-/post-processing are therefore IDENTICAL across backends — any mAP or latency
delta reflects only the numerical backend, which is exactly what the consistency
check is meant to isolate.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np

from preprocess import LetterboxResult, to_nchw


class Detector(Protocol):
    name: str

    @staticmethod
    def create(kind: str, path: str, device: str, num_classes: int) -> "Detector": ...

    def forward(self, chw: np.ndarray) -> np.ndarray: ...


class PyTorchDetector:
    """Runs the ultralytics model head directly to obtain the raw (1, 4+nc, N) tensor."""

    name = "pytorch"

    def __init__(self, path: str, device: str, num_classes: int, mode: str = "sparse"):
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.device = device
        self.num_classes = num_classes
        self.yolo = YOLO(path)
        self.model = self.yolo.model.to(device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        if mode != "sparse":
            # ONNX/MNN export always traces _dense_forward. Match it:
            #   "combine" : dense expert combine, KEEP top-k routing (== ONNX/MNN export).
            #               Numerically identical to the export (diff ~1e-4). Use for ONNX/MNN.
            #   "full"    : also force full-softmax routing (use_top_k=False). Use for NCNN,
            #               whose graph cannot contain topk/one_hot.
            from ultralytics.nn.modules.moe.routers import DynamicRoutingLayer
            import ultralytics.nn.modules.moe.modules as mm
            for m in self.model.modules():
                if mode == "full" and isinstance(m, DynamicRoutingLayer):
                    m.use_top_k = False
                if isinstance(m, getattr(mm, "ES_MOE")):
                    m.use_sparse_inference = False

    def forward(self, chw: np.ndarray) -> np.ndarray:
        t = self.torch.from_numpy(chw).unsqueeze(0).to(self.device)
        t = (t / 255.0).float() if t.max() > 1.5 else t.float()
        with self.torch.no_grad():
            out = self.model(t)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.detach().cpu().numpy()


class ONNXDetector:
    name = "onnx"

    def __init__(self, path: str, device: str, num_classes: int):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device or "0" in device else ["CPUExecutionProvider"]
        )
        self.sess = ort.InferenceSession(path, providers=providers)
        self.in_name = self.sess.get_inputs()[0].name
        self.num_classes = num_classes

    def forward(self, chw: np.ndarray) -> np.ndarray:
        inp = to_nchw(chw)  # already 0..1 float32
        out = self.sess.run(None, {self.in_name: inp})[0]
        return out


class NCNNDetector:
    name = "ncnn"

    def __init__(self, path: str, device: str, num_classes: int):
        import ncnn

        # path may be the _ncnn_model dir, the .param, or the parent
        path = Path(path)
        if path.is_dir():
            param = path / "model.ncnn.param"
            binf = path / "model.ncnn.bin"
        elif path.suffix == ".param":
            param = path
            binf = path.with_suffix(".bin")
        else:
            param = path / "model.ncnn.param"
            binf = path / "model.ncnn.bin"
        self.param, self.binf = str(param), str(binf)
        self.num_classes = num_classes
        self.in_name, self.out_candidates = self._probe_io_names()
        self._net = None

    def _probe_io_names(self) -> tuple[str, list]:
        """pnnx-generated ncnn models name the single I/O blob in0/out0 (confirmed
        by the model_ncnn.py reference pnnx emits alongside the .param/.bin)."""
        return "in0", ["out0", "output0", "output"]

    def _net_lazy(self):
        if self._net is None:
            import ncnn

            net = ncnn.Net()
            net.opt.use_vulkan_compute = False  # CPU-only — vulkan is unstable on this host
            net.load_param(self.param)
            net.load_model(self.binf)
            self._net = net
        return self._net

    def forward(self, chw: np.ndarray) -> np.ndarray:
        import ncnn

        ex = self._net_lazy().create_extractor()
        # ncnn expects a 3D Mat (C,H,W) for a single image — match model_ncnn.py.
        ex.input(self.in_name, ncnn.Mat(np.ascontiguousarray(chw)))
        ret = None
        for name in self.out_candidates:
            try:
                code, out = ex.extract(name)
                if code == 0:
                    ret = out
                    break
            except Exception:
                continue
        if ret is None:
            raise RuntimeError(f"NCNN: could not extract output among {self.out_candidates}")
        out = np.array(ret)
        if out.ndim == 3 and out.shape[0] != 1:
            out = out[np.newaxis]
        return out


class MNNDetector:
    name = "mnn"

    def __init__(self, path: str, device: str, num_classes: int):
        import MNN

        self.MNN = MNN
        self.num_classes = num_classes
        self.interp = MNN.Interpreter(str(path))
        self._session = None

    def _session_lazy(self):
        if self._session is None:
            self._session = self.interp.createSession({"backend": "CPU", "thread": 4})
        return self._session

    def forward(self, chw: np.ndarray) -> np.ndarray:
        MNN = self.MNN
        s = self._session_lazy()
        tin = self.interp.getSessionInput(s)
        host = MNN.Tensor(
            list(tin.getShape()),
            MNN.Halide_Type_Float,
            np.ascontiguousarray(to_nchw(chw)).astype(np.float32),
            MNN.Tensor_DimensionType_Caffe,
        )
        tin.copyFromHostTensor(host)
        self.interp.runSession(s)
        out = self.interp.getSessionOutput(s).getNumpyData()
        return np.array(out)


_REGISTRY = {"pytorch": PyTorchDetector, "onnx": ONNXDetector, "ncnn": NCNNDetector, "mnn": MNNDetector}


def create_detector(kind: str, path: str, device: str = "cpu", num_classes: int = 10) -> Detector:
    kind = kind.lower()
    if kind not in _REGISTRY:
        raise ValueError(f"unknown backend '{kind}', choose from {list(_REGISTRY)}")
    return _REGISTRY[kind](path, device, num_classes)
