"""ONNX Runtime host orchestrator for split dynamic-expert bundles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .bundle import BUNDLE_SCHEMA_VERSION
from .dispatch import (
    DynamicDispatchAudit,
    DynamicDispatchContractError,
    dispatch_numpy_experts,
    sparsify_topk_probabilities,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ORTDynamicExpertRuntime:
    """Execute a split bundle while invoking only selected ONNX expert sessions."""

    def __init__(
        self,
        manifest_path: str | Path,
        *,
        providers: Sequence[str] | None = None,
        verify_hashes: bool = True,
    ):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError("onnxruntime is required for ORTDynamicExpertRuntime") from error

        self.manifest_path = Path(manifest_path).resolve()
        self.root = self.manifest_path.parent
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION:
            raise DynamicDispatchContractError(
                f"unsupported dynamic bundle schema {self.manifest.get('schema_version')!r}; "
                f"expected {BUNDLE_SCHEMA_VERSION}"
            )
        if self.manifest.get("execution_semantics") != "host_conditional_expert_dispatch":
            raise DynamicDispatchContractError("manifest does not declare host conditional expert dispatch")
        if self.manifest.get("masked_dense_allowed") is not False:
            raise DynamicDispatchContractError("manifest must explicitly forbid masked-dense execution")

        self._ort = ort
        self.providers = list(providers or ["CPUExecutionProvider"])
        self._session_options = ort.SessionOptions()
        self._session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._verify_hashes = bool(verify_hashes)

        artifacts = self.manifest["artifacts"]
        self._router_record = artifacts["router"]
        self._expert_records = list(artifacts["experts"])
        self._postprocess_record = artifacts["postprocess"]
        if len(self._expert_records) != int(self.manifest["num_experts"]):
            raise DynamicDispatchContractError("manifest expert count does not match num_experts")

        if self._verify_hashes:
            for record in (self._router_record, *self._expert_records, self._postprocess_record):
                self._verify_artifact(record)

        self._router_session = self._new_session(self._resolve_artifact(self._router_record))
        self._postprocess_session = self._new_session(self._resolve_artifact(self._postprocess_record))
        self._expert_sessions: dict[int, object] = {}
        self.last_audit: DynamicDispatchAudit | None = None
        self.last_dense_routing_probabilities: np.ndarray | None = None
        self.last_routing_weights: np.ndarray | None = None

    def _resolve_artifact(self, record: dict) -> Path:
        path = (self.root / record["path"]).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise DynamicDispatchContractError(f"artifact escapes bundle root: {record['path']}") from error
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    def _verify_artifact(self, record: dict) -> None:
        path = self._resolve_artifact(record)
        expected_size = int(record["bytes"])
        if path.stat().st_size != expected_size:
            raise DynamicDispatchContractError(
                f"artifact size mismatch for {path.name}: expected {expected_size}, got {path.stat().st_size}"
            )
        actual_hash = _sha256(path)
        if actual_hash != record["sha256"]:
            raise DynamicDispatchContractError(
                f"artifact SHA256 mismatch for {path.name}: expected {record['sha256']}, got {actual_hash}"
            )

    def _new_session(self, path: Path):
        return self._ort.InferenceSession(
            str(path),
            sess_options=self._session_options,
            providers=self.providers,
        )

    def _expert_session(self, expert_index: int):
        if expert_index not in self._expert_sessions:
            path = self._resolve_artifact(self._expert_records[expert_index])
            self._expert_sessions[expert_index] = self._new_session(path)
        return self._expert_sessions[expert_index]

    def _expert_runner(self, expert_index: int):
        record = self._expert_records[expert_index]

        def run(selected_x: np.ndarray) -> np.ndarray:
            session = self._expert_session(expert_index)
            return session.run(record["outputs"], {record["inputs"][0]: selected_x})[0]

        return run

    @property
    def loaded_expert_ids(self) -> tuple[int, ...]:
        """Return expert sessions loaded by calls so far; unselected experts remain unloaded."""
        return tuple(sorted(self._expert_sessions))

    def run(
        self,
        x: np.ndarray,
        *,
        require_reduction: bool = False,
    ) -> tuple[np.ndarray, DynamicDispatchAudit]:
        """Run router → selected experts → postprocess and return execution evidence."""
        inputs = np.asarray(x)
        routing_weights = self.route(inputs)
        expert_runners = [self._expert_runner(index) for index in range(int(self.manifest["num_experts"]))]
        mixture, audit = dispatch_numpy_experts(
            inputs,
            routing_weights,
            expert_runners,
            top_k=int(self.manifest["top_k"]),
            routing_granularity=self.manifest["routing_granularity"],
            dynamic_threshold=float(self.manifest["dynamic_threshold"]),
            zero_tolerance=float(self.manifest["zero_tolerance"]),
            require_reduction=require_reduction,
        )

        post_inputs = self._postprocess_record["inputs"]
        if post_inputs == ["mixture"]:
            feeds = {"mixture": mixture}
        elif post_inputs == ["x", "mixture"]:
            feeds = {"x": inputs, "mixture": mixture}
        else:
            raise DynamicDispatchContractError(f"unsupported postprocess inputs: {post_inputs}")
        output = self._postprocess_session.run(self._postprocess_record["outputs"], feeds)[0]
        self.last_audit = audit
        return output, audit

    def route(self, x: np.ndarray) -> np.ndarray:
        """Run only the ONNX router and return normalized sparse host Top-K weights.

        Expert and postprocess sessions are not invoked. This public boundary is
        used by hybrid validation runtimes that keep the real checkpoint experts
        in PyTorch/CUDA while the exported router defines the deployment route.
        """
        inputs = np.asarray(x)
        expected_shape = tuple(int(value) for value in self.manifest["sample_shape"])
        if inputs.ndim != 4 or tuple(inputs.shape[1:]) != expected_shape[1:]:
            raise DynamicDispatchContractError(
                f"runtime input must be [B,{','.join(map(str, expected_shape[1:]))}], got {tuple(inputs.shape)}"
            )
        if str(inputs.dtype) != self.manifest["sample_dtype"]:
            raise DynamicDispatchContractError(
                f"runtime input dtype must be {self.manifest['sample_dtype']}, got {inputs.dtype}"
            )

        router_output = self._router_session.run(
            self._router_record["outputs"],
            {self._router_record["inputs"][0]: inputs},
        )[0]
        router_semantics = self.manifest.get("router_output_semantics", "sparse_topk")
        if router_semantics == "dense_probabilities_host_topk":
            self.last_dense_routing_probabilities = router_output.copy()
            routing_weights = sparsify_topk_probabilities(
                router_output,
                top_k=int(self.manifest["top_k"]),
                zero_tolerance=float(self.manifest["zero_tolerance"]),
                tie_tolerance=float(self.manifest.get("host_topk_tie_tolerance", 0.0)),
            )
        elif router_semantics == "sparse_topk":
            self.last_dense_routing_probabilities = None
            routing_weights = router_output
        else:
            raise DynamicDispatchContractError(f"unsupported router_output_semantics: {router_semantics!r}")
        self.last_routing_weights = routing_weights.copy()
        return routing_weights


__all__ = ("ORTDynamicExpertRuntime",)
