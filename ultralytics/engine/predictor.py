# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Run prediction on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ yolo mode=predict model=yolo26n.pt source=0                               # webcam
                                                img.jpg                         # image
                                                vid.mp4                         # video
                                                screen                          # screenshot
                                                path/                           # directory
                                                list.txt                        # list of images
                                                list.streams                    # list of streams
                                                'path/*.jpg'                    # glob
                                                'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP, TCP stream

Usage - formats:
    $ yolo mode=predict model=yolo26n.pt                 # PyTorch
                              yolo26n.torchscript        # TorchScript
                              yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                              yolo26n_openvino_model     # OpenVINO
                              yolo26n.engine             # TensorRT
                              yolo26n.mlpackage          # CoreML (macOS-only)
                              yolo26n_saved_model        # TensorFlow SavedModel
                              yolo26n.pb                 # TensorFlow GraphDef
                              yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                              yolo26n_paddle_model       # PaddlePaddle
                              yolo26n.mnn                # MNN
                              yolo26n_ncnn_model         # NCNN
                              yolo26n_imx_model          # Sony IMX
                              yolo26n_rknn_model         # Rockchip RKNN
                              yolo26n_executorch_model   # PyTorch Executorch
                              yolo26n_axelera_model      # Axelera AI
                              yolo26n_deepx_model        # DEEPX
                              yolo26n_qnn.onnx           # Qualcomm QNN
                              yolo26n.tflite             # LiteRT
"""

from __future__ import annotations

import platform
import re
import threading
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import load_inference_source
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import DEFAULT_CFG, LOGGER, MACOS, WINDOWS, callbacks, colorstr, ops
from ultralytics.utils.checks import check_imgsz, check_imshow
from ultralytics.utils.files import increment_path
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode

STREAM_WARNING = """
Inference results will accumulate in RAM unless `stream=True` is passed, which can cause out-of-memory errors for large
sources or long-running streams and videos. See https://docs.ultralytics.com/modes/predict/ for help.

Example:
    results = model(source=..., stream=True)  # generator of Results objects
    for r in results:
        boxes = r.boxes  # Boxes object for bbox outputs
        masks = r.masks  # Masks object for segment masks outputs
        probs = r.probs  # Class probabilities for classification outputs
"""


class BasePredictor:
    """A base class for creating predictors.

    This class provides the foundation for prediction functionality, handling model setup, inference, and result
    processing across various input sources.

    Attributes:
        args (SimpleNamespace): Configuration for the predictor.
        save_dir (Path): Directory to save results.
        done_warmup (bool): Whether the predictor has finished setup.
        model (torch.nn.Module): Model used for prediction.
        data (str): Data configuration.
        device (torch.device): Device used for prediction.
        dataset (Dataset): Dataset used for prediction.
        vid_writer (dict[Path, cv2.VideoWriter]): Dictionary of {save_path: video_writer} for saving video output.
        plotted_img (np.ndarray): Last plotted image.
        source_type (SimpleNamespace): Type of input source.
        seen (int): Number of images processed.
        windows (list[str]): List of window names for visualization.
        batch (tuple): Current batch data.
        results (list[Any]): Current batch results.
        transforms (Callable): Image transforms for classification.
        callbacks (dict[str, list[Callable]]): Callback functions for different events.
        txt_path (Path): Path to save text results.
        _lock (threading.Lock): Lock for thread-safe inference.

    Methods:
        preprocess: Prepare input image before inference.
        inference: Run inference on a given image.
        postprocess: Process raw predictions into structured results.
        predict_cli: Run prediction for command line interface.
        setup_source: Set up input source and inference mode.
        stream_inference: Stream inference on input source.
        setup_model: Initialize and configure the model.
        write_results: Write inference results to files.
        save_predicted_images: Save prediction visualizations.
        show: Display results in a window.
        run_callbacks: Execute registered callbacks for an event.
        add_callback: Register a new callback function.
    """

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        _callbacks: dict | None = None,
    ):
        """Initialize the BasePredictor class.

        Args:
            cfg (str | Path | dict | SimpleNamespace): Path to a configuration file or a configuration dictionary.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.args = get_cfg(cfg, overrides)
        self.save_dir = get_save_dir(self.args)
        if self.args.conf is None:
            self.args.conf = 0.25  # default conf=0.25
        self.done_warmup = False
        if self.args.show:
            self.args.show = check_imshow(warn=True)

        # Usable if setup is done
        self.model = None
        self.data = self.args.data  # data_dict
        self.imgsz = None
        self.device = None
        self.dataset = None
        self.vid_writer = {}  # dict of {save_path: video_writer, ...}
        self.plotted_img = None
        self.source_type = None
        self.seen = 0
        self.windows = []
        self.screen = None  # cached screen resolution (width, height) for show=True scaling
        self.batch = None
        self.results = None
        self.transforms = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.txt_path = None
        self._lock = threading.Lock()  # for automatic thread-safe inference
        callbacks.add_integration_callbacks(self)

    def preprocess(self, im: torch.Tensor | list[np.ndarray]) -> torch.Tensor:
        """Prepare input image before inference.

        Args:
            im (torch.Tensor | list[np.ndarray]): Images of shape (N, 3, H, W) for tensor, [(H, W, 3) x N] for list.

        Returns:
            (torch.Tensor): Preprocessed image tensor of shape (N, 3, H, W).
        """
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            if im.shape[-1] == 3:
                im = im[..., ::-1]  # BGR to RGB
            im = im.transpose((0, 3, 1, 2))  # BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        if not_tensor:
            im /= 255  # 0 - 255 to 0.0 - 1.0
        return im

    def inference(self, im: torch.Tensor, *args, **kwargs):
        """Run inference on a given image using the specified model and arguments."""
        visualize = (
            increment_path(self.save_dir / Path(self.batch[0][0]).stem, mkdir=True)
            if self.args.visualize and (not self.source_type.tensor)
            else False
        )
        return self.model(im, augment=self.args.augment, visualize=visualize, embed=self.args.embed, *args, **kwargs)

    def pre_transform(self, im: list[np.ndarray]) -> list[np.ndarray]:
        """Pre-transform input image before inference.

        Args:
            im (list[np.ndarray]): List of images with shape [(H, W, 3) x N].

        Returns:
            (list[np.ndarray]): List of transformed images.
        """
        same_shapes = len({x.shape for x in im}) == 1
        letterbox = LetterBox(
            self.imgsz,
            auto=same_shapes
            and self.args.rect
            and (self.model.format == "pt" or (getattr(self.model, "dynamic", False) and self.model.format != "imx")),
            stride=self.model.stride,
        )
        return [letterbox(image=x) for x in im]

    def postprocess(self, preds, img, orig_imgs):
        """Post-process predictions for an image and return them."""
        return preds

    def __call__(self, source=None, model=None, stream: bool = False, *args, **kwargs):
        """Perform inference on an image or stream.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            stream (bool): Whether to stream the inference results. If True, returns a generator.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Returns:
            (list[ultralytics.engine.results.Results] | generator): Results objects or generator of Results objects.
        """
        self.stream = stream
        if stream:
            return self.stream_inference(source, model, *args, **kwargs)
        else:
            return list(self.stream_inference(source, model, *args, **kwargs))  # merge list of Results into one

    def predict_cli(self, source=None, model=None):
        """Method used for Command Line Interface (CLI) prediction.

        This function is designed to run predictions using the CLI. It sets up the source and model, then processes the
        inputs in a streaming manner. This method ensures that no outputs accumulate in memory by consuming the
        generator without storing results.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.

        Notes:
            Do not modify this function or remove the generator. The generator ensures that no outputs are
            accumulated in memory, which is critical for preventing memory issues during long-running predictions.
        """
        gen = self.stream_inference(source, model)
        for _ in gen:  # sourcery skip: remove-empty-nested-block, noqa
            pass

    def setup_source(self, source, stride: int | None = None):
        """Set up source and inference mode.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor): Source for
                inference.
            stride (int, optional): Model stride for image size checking.
        """
        self.imgsz = check_imgsz(self.args.imgsz, stride=stride or self.model.stride, min_dim=2)  # check image size
        self.dataset = load_inference_source(
            source=source,
            batch=self.args.batch,
            vid_stride=self.args.vid_stride,
            buffer=self.args.stream_buffer,
            channels=getattr(self.model, "channels", 3),
        )
        self.source_type = self.dataset.source_type
        if (
            self.source_type.stream
            or self.source_type.screenshot
            or len(self.dataset) > 1000  # many images
            or any(getattr(self.dataset, "video_flag", [False]))
        ):  # long sequence
            import torchvision  # noqa (import here triggers torchvision NMS use in nms.py)

            if not getattr(self, "stream", True):  # videos
                LOGGER.warning(STREAM_WARNING)
        self.vid_writer = {}

    @smart_inference_mode()
    def stream_inference(self, source=None, model=None, *args, **kwargs):
        """Stream inference on input source and save results to file.

        Args:
            source (str | Path | list[str] | list[Path] | list[np.ndarray] | np.ndarray | torch.Tensor, optional):
                Source for inference.
            model (str | Path | torch.nn.Module, optional): Model for inference.
            *args (Any): Additional arguments for the inference method.
            **kwargs (Any): Additional keyword arguments for the inference method.

        Yields:
            (ultralytics.engine.results.Results): Results objects.
        """
        if self.args.verbose:
            LOGGER.info("")

        # Setup model
        if self.model is None:
            self.setup_model(model)

        with self._lock:  # for thread-safe inference
            # Setup source every time predict is called
            self.setup_source(source if source is not None else self.args.source)

            # Check if save_dir/ label file exists
            if self.args.save or self.args.save_txt:
                (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)

            # Warmup model
            if not self.done_warmup:
                self.model.warmup(
                    imgsz=(
                        1 if self.model.format in {"pt", "triton"} else self.dataset.bs,
                        self.model.channels,
                        *self.imgsz,
                    )
                )
                self.done_warmup = True

            self.seen, self.windows, self.batch = 0, [], None
            profilers = (
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
                ops.Profile(device=self.device),
            )
            self.run_callbacks("on_predict_start")
            for batch in self.dataset:
                self.batch = batch
                self.run_callbacks("on_predict_batch_start")
                paths, im0s, s = self.batch

                # Preprocess
                with profilers[0]:
                    im = self.preprocess(im0s)

                # Inference
                with profilers[1]:
                    preds = self.inference(im, *args, **kwargs)
                    if self.args.embed:
                        yield from [preds] if isinstance(preds, torch.Tensor) else preds  # yield embedding tensors
                        continue

                # Postprocess
                with profilers[2]:
                    if getattr(self.args, "sparse_sahi", False):
                        global_results = self.postprocess(preds, im, im0s)
                        self.results = [
                            self._run_sparse_sahi_single(im0s[i], res, *args, **kwargs)
                            for i, res in enumerate(global_results)
                        ]
                    else:
                        self.results = self.postprocess(preds, im, im0s)
                self.run_callbacks("on_predict_postprocess_end")

                # Visualize, save, write results
                n = len(im0s)
                try:
                    for i in range(n):
                        self.seen += 1
                        self.results[i].speed = {
                            "preprocess": profilers[0].dt * 1e3 / n,
                            "inference": profilers[1].dt * 1e3 / n,
                            "postprocess": profilers[2].dt * 1e3 / n,
                        }
                        if (
                            self.args.verbose
                            or self.args.save
                            or self.args.save_txt
                            or self.args.save_crop
                            or self.args.show
                        ):
                            s[i] += self.write_results(i, Path(paths[i]), im, s)
                except StopIteration:
                    break

                # Print batch results
                if self.args.verbose:
                    LOGGER.info("\n".join(s))

                self.run_callbacks("on_predict_batch_end")
                yield from self.results

        # Release assets
        for v in self.vid_writer.values():
            if isinstance(v, cv2.VideoWriter):
                v.release()

        if self.args.show:
            cv2.destroyAllWindows()  # close any open windows

        # Print final results
        if self.args.verbose and self.seen:
            t = tuple(x.t / self.seen * 1e3 for x in profilers)  # speeds per image
            LOGGER.info(
                f"Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess per image at shape "
                f"{(min(self.args.batch, self.seen), getattr(self.model, 'channels', 3), *im.shape[2:])}" % t
            )
        if self.args.save or self.args.save_txt or self.args.save_crop:
            nl = len(list(self.save_dir.glob("labels/*.txt")))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {self.save_dir / 'labels'}" if self.args.save_txt else ""
            LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}{s}")
        self.run_callbacks("on_predict_end")

    def setup_model(self, model, verbose: bool = True):
        """Initialize YOLO model with given parameters and set it to evaluation mode.

        Args:
            model (str | Path | torch.nn.Module): Model to load or use.
            verbose (bool): Whether to print verbose output.
        """
        if hasattr(model, "end2end"):
            if self.args.end2end is not None:
                model.end2end = self.args.end2end
            if model.end2end:
                # Keep head top-k >= 300 so `classes` filtering in NMS sees all candidates before `max_det` truncation
                model.set_head_attr(max_det=max(self.args.max_det, 300), agnostic_nms=self.args.agnostic_nms)
        self.model = AutoBackend(
            model=model or self.args.model,
            device=select_device(self.args.device, verbose=verbose),
            dnn=self.args.dnn,
            data=self.args.data,
            fp16=self.args.quantize == 16,
            fuse=True,
            verbose=verbose,
        )

        self.device = self.model.device  # update device
        self.args.quantize = 16 if self.model.fp16 else None  # record actual inference precision
        if hasattr(self.model, "imgsz") and not getattr(self.model, "dynamic", False):
            self.args.imgsz = self.model.imgsz  # reuse imgsz from export metadata
        self.model.eval()
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (list[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # None if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        if self.args.save or self.args.show:
            self.plotted_img = result.plot(
                line_width=self.args.line_width,
                boxes=self.args.show_boxes,
                conf=self.args.show_conf,
                labels=self.args.show_labels,
                im_gpu=None if self.args.retina_masks else im[i],
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def save_predicted_images(self, save_path: Path, frame: int = 0):
        """Save video predictions as mp4/avi or images as jpg at specified path.

        Args:
            save_path (Path): Path to save the results.
            frame (int): Frame number for video mode.
        """
        im = self.plotted_img

        # Save videos and streams
        if self.dataset.mode in {"stream", "video"}:
            fps = self.dataset.fps if self.dataset.mode == "video" else 30
            frames_path = self.save_dir / f"{save_path.stem}_frames"  # save frames to a separate directory
            if save_path not in self.vid_writer:  # new video
                if self.args.save_frames:
                    Path(frames_path).mkdir(parents=True, exist_ok=True)
                suffix, fourcc = (".mp4", "avc1") if MACOS else (".avi", "WMV2") if WINDOWS else (".avi", "MJPG")
                self.vid_writer[save_path] = cv2.VideoWriter(
                    filename=str(Path(save_path).with_suffix(suffix)),
                    fourcc=cv2.VideoWriter_fourcc(*fourcc),
                    fps=fps,  # integer required, floats produce error in MP4 codec
                    frameSize=(im.shape[1], im.shape[0]),  # (width, height)
                )

            # Save video
            self.vid_writer[save_path].write(im)
            if self.args.save_frames:
                cv2.imwrite(f"{frames_path}/{save_path.stem}_{frame}.jpg", im)

        # Save images
        else:
            cv2.imwrite(str(save_path.with_suffix(".jpg")), im)  # save to JPG for best support

    def show(self, p: str = ""):
        """Display an image in a window."""
        im = self.plotted_img
        if platform.system() in {"Linux", "Windows"} and p not in self.windows:  # macOS scales natively
            self.windows.append(p)
            name = p.encode("unicode_escape").decode()  # match patched cv2.imshow window name
            cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize and scaling
            h, w = im.shape[:2]
            try:  # size window to fit screen once on creation if image larger than screen resolution
                if self.screen is None:
                    root = __import__("tkinter").Tk()
                    root.withdraw()  # hide the empty Tk window
                    self.screen = 0.9 * root.winfo_screenwidth(), 0.9 * root.winfo_screenheight()  # 0.9 taskbar margin
                    root.destroy()
                r = min(self.screen[0] / w, self.screen[1] / h, 1.0)
                cv2.resizeWindow(name, max(1, int(w * r)), max(1, int(h * r)))  # (width, height)
            except Exception:
                cv2.resizeWindow(name, w, h)
        cv2.imshow(p, im)
        if cv2.waitKey(300 if self.dataset.mode == "image" else 1) & 0xFF == ord("q"):  # 300ms if image; else 1ms
            raise StopIteration

    def run_callbacks(self, event: str):
        """Run all registered callbacks for a specific event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def add_callback(self, event: str, func: Callable):
        """Add a callback function for a specific event."""
        self.callbacks[event].append(func)

    def _pad_slice(self, slice_img, slice_h, slice_w):
        """Pad slice to match slice_h and slice_w."""
        h, w = slice_img.shape[:2]
        if h == slice_h and w == slice_w:
            return slice_img
        pad_img = np.full((slice_h, slice_w, 3), 114, dtype=np.uint8)
        pad_img[:h, :w] = slice_img
        return pad_img

    def _perform_batched_nms(self, boxes, scores, cls, iou_thres):
        """Perform batched NMS using torchvision."""
        from torchvision.ops import nms

        if boxes.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        max_coordinate = boxes.max()
        offsets = cls.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        return nms(boxes_for_nms, scores, iou_thres)

    def _run_sparse_sahi_single(self, img, global_result, *args, **kwargs):
        """Run Sparse SAHI on a single image.

        Pipeline: low-res global inference -> objectness mask -> adaptive sparse slicing ->
        high-res slice inference -> batched NMS merge. Attaches `sparse_sahi_metadata` to the
        returned Results for downstream visualization.
        """
        from ultralytics.engine.results import Boxes

        slice_size = int(getattr(self.args, "slice_size", 640))
        overlap_ratio = float(getattr(self.args, "overlap_ratio", 0.2))
        objectness_threshold = float(getattr(self.args, "objectness_threshold", 0.15))
        slice_h = slice_w = slice_size
        overlap = int(min(slice_h, slice_w) * overlap_ratio)
        step_y = max(1, slice_h - overlap)
        step_x = max(1, slice_w - overlap)

        img_h, img_w = img.shape[:2]

        # --- Step 1: Objectness Mask (1/8 scale, painted from global detections) ---
        mask_scale = 8
        m_h, m_w = (img_h // mask_scale) + 1, (img_w // mask_scale) + 1
        objectness_mask = np.zeros((m_h, m_w), dtype=np.float32)

        if len(global_result.boxes) > 0:
            g_boxes = (global_result.boxes.xyxy.cpu().numpy() / mask_scale).astype(int)
            g_scores = global_result.boxes.conf.cpu().numpy()
            for box, score in zip(g_boxes, g_scores):
                x1, y1, x2, y2 = box
                y1_i, y2_i = max(0, y1), min(m_h, y2)
                x1_i, x2_i = max(0, x1), min(m_w, x2)
                objectness_mask[y1_i:y2_i, x1_i:x2_i] = np.maximum(objectness_mask[y1_i:y2_i, x1_i:x2_i], score)

        # --- Step 2: Adaptive Sparse Slicing ---
        active_slice_coords = []
        for y_min in range(0, img_h, step_y):
            for x_min in range(0, img_w, step_x):
                y_max = min(y_min + slice_h, img_h)
                x_max = min(x_min + slice_w, img_w)
                m_y1, m_x1 = y_min // mask_scale, x_min // mask_scale
                m_y2, m_x2 = y_max // mask_scale, x_max // mask_scale
                if objectness_mask[m_y1:m_y2, m_x1:m_x2].max() > objectness_threshold:
                    active_slice_coords.append((x_min, y_min, x_max, y_max))

        # Collect all boxes (Global + Slices)
        all_boxes = []
        all_scores = []
        all_cls = []
        all_sources = []  # 0 = global, 1 = slice

        if len(global_result.boxes) > 0:
            all_boxes.append(global_result.boxes.xyxy)
            all_scores.append(global_result.boxes.conf)
            all_cls.append(global_result.boxes.cls)
            all_sources.extend([0] * len(global_result.boxes))

        # --- Step 3: Batch Inference on Slices ---
        if active_slice_coords:
            batch_imgs = [
                self._pad_slice(img[y1:y2, x1:x2], slice_h, slice_w) for (x1, y1, x2, y2) in active_slice_coords
            ]
            slice_tensor = self.preprocess(batch_imgs)
            slice_preds = self.inference(slice_tensor, *args, **kwargs)
            slice_results = self.postprocess(slice_preds, slice_tensor, batch_imgs)

            for res, (off_x, off_y, _, _) in zip(slice_results, active_slice_coords):
                if len(res.boxes) > 0:
                    boxes = res.boxes.xyxy.clone()
                    boxes[:, [0, 2]] += off_x
                    boxes[:, [1, 3]] += off_y
                    all_boxes.append(boxes)
                    all_scores.append(res.boxes.conf)
                    all_cls.append(res.boxes.cls)
                    all_sources.extend([1] * len(boxes))

        # --- Step 4: Merge & NMS ---
        if not all_boxes:
            global_result.sparse_sahi_metadata = {
                "objectness_map": objectness_mask,
                "slices": active_slice_coords,
                "final_sources": [],
            }
            return global_result

        cat_boxes = torch.cat(all_boxes).to(self.device)
        cat_scores = torch.cat(all_scores).to(self.device)
        cat_cls = torch.cat(all_cls).to(self.device)

        keep = self._perform_batched_nms(cat_boxes, cat_scores, cat_cls, self.args.iou)
        final_boxes = cat_boxes[keep]
        final_scores = cat_scores[keep]
        final_cls = cat_cls[keep]

        keep_indices = keep.cpu().numpy()
        current_sources = np.array(all_sources)
        final_sources = current_sources[keep_indices].tolist() if len(current_sources) > 0 else []

        if len(final_boxes) > 0:
            det = torch.cat([final_boxes, final_scores.unsqueeze(1), final_cls.unsqueeze(1)], dim=1)
        else:
            det = torch.empty((0, 6), device=self.device)

        global_result.boxes = Boxes(det, global_result.orig_shape)
        global_result.sparse_sahi_metadata = {
            "objectness_map": objectness_mask,
            "slices": active_slice_coords,
            "final_sources": final_sources,
        }
        return global_result
