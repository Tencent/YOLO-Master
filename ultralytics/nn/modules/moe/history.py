"""Persistence helpers for lightweight MoE routing diagnostics."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from .diagnostics import MoELayerDiagnostic


def _sanitize_layer_name(name: str) -> str:
    """Convert a module path into a filesystem-safe stem."""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip())
    return sanitized.replace(".", "_") or "unknown_layer"


class MoEDiagnosticsRecorder:
    """Persist MoE routing snapshots and detect sustained routing issues."""

    csv_fieldnames = (
        "stage",
        "step",
        "epoch",
        "layer_name",
        "module_type",
        "num_experts",
        "top_k",
        "expert_id",
        "usage",
        "count",
        "dominant_expert",
        "dominant_share",
        "aux_loss",
        "collapse_flag",
    )

    def __init__(
        self,
        save_dir: str | Path,
        dead_threshold: float = 0.01,
        dead_window: int = 5,
        collapse_threshold: float = 0.8,
        collapse_window: int = 3,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir = self.save_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)

        self.dead_threshold = float(dead_threshold)
        self.dead_window = max(int(dead_window), 1)
        self.collapse_threshold = float(collapse_threshold)
        self.collapse_window = max(int(collapse_window), 1)

        self.history_jsonl = self.save_dir / "routing_history.jsonl"
        self.history_csv = self.save_dir / "routing_history.csv"
        self.alerts_jsonl = self.save_dir / "alerts.jsonl"
        self.alerts_jsonl.touch(exist_ok=True)

        self._rolling_history: dict[str, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=max(self.dead_window, self.collapse_window))
        )
        self._active_alerts: set[tuple[Any, ...]] = set()

    def record(
        self,
        *,
        step: int,
        epoch: int,
        diagnostics: list[MoELayerDiagnostic],
        stage: str = "train",
    ) -> list[dict[str, Any]]:
        """Append diagnostics to history files and return newly triggered alerts."""
        alerts: list[dict[str, Any]] = []
        for diag in diagnostics:
            payload = self._build_layer_payload(diag=diag, step=step, epoch=epoch, stage=stage)
            self._append_jsonl(self.history_jsonl, payload)
            self._append_csv_rows(payload)
            layer_alerts = self._evaluate_alerts(payload)
            for alert in layer_alerts:
                self._append_jsonl(self.alerts_jsonl, alert)
            alerts.extend(layer_alerts)
        return alerts

    def export_plots(self) -> list[Path]:
        """Export usage and aux-loss plots from the current history file."""
        return export_moe_history_plots(self.save_dir)

    def _build_layer_payload(
        self, *, diag: MoELayerDiagnostic, step: int, epoch: int, stage: str
    ) -> dict[str, Any]:
        return {
            "stage": stage,
            "step": int(step),
            "epoch": int(epoch),
            "layer_name": diag.name,
            "module_type": diag.module_type,
            "num_experts": int(diag.num_experts),
            "top_k": int(diag.top_k),
            "aux_loss": float(diag.aux_loss),
            "usage": [float(v) for v in diag.usage],
            "counts": [float(v) for v in diag.counts],
            "dominant_expert": int(diag.dominant_expert),
            "dominant_share": float(diag.dominant_share),
            "collapse_flag": bool(diag.collapse_flag),
            "mean_router_probs": diag.mean_router_probs,
            "mean_topk_weight": diag.mean_topk_weight,
        }

    def _append_jsonl(self, path: Path, payload: dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True) + "\n")

    def _append_csv_rows(self, payload: dict[str, Any]) -> None:
        needs_header = not self.history_csv.exists() or self.history_csv.stat().st_size == 0
        with self.history_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.csv_fieldnames)
            if needs_header:
                writer.writeheader()
            for expert_id, usage in enumerate(payload["usage"]):
                writer.writerow(
                    {
                        "stage": payload["stage"],
                        "step": payload["step"],
                        "epoch": payload["epoch"],
                        "layer_name": payload["layer_name"],
                        "module_type": payload["module_type"],
                        "num_experts": payload["num_experts"],
                        "top_k": payload["top_k"],
                        "expert_id": expert_id,
                        "usage": usage,
                        "count": payload["counts"][expert_id] if expert_id < len(payload["counts"]) else 0.0,
                        "dominant_expert": payload["dominant_expert"],
                        "dominant_share": payload["dominant_share"],
                        "aux_loss": payload["aux_loss"],
                        "collapse_flag": payload["collapse_flag"],
                    }
                )

    def _evaluate_alerts(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        history = self._rolling_history[payload["layer_name"]]
        history.append(payload)
        alerts: list[dict[str, Any]] = []

        for expert_id in range(payload["num_experts"]):
            dead_key = ("dead_expert", payload["layer_name"], expert_id)
            dead_active = self._is_dead_expert(history, expert_id)
            if dead_active and dead_key not in self._active_alerts:
                self._active_alerts.add(dead_key)
                alerts.append(
                    self._build_alert(
                        alert_type="dead_expert",
                        payload=payload,
                        expert_id=expert_id,
                        threshold=self.dead_threshold,
                        window=self.dead_window,
                    )
                )
            elif not dead_active:
                self._active_alerts.discard(dead_key)

        collapse_expert = payload["dominant_expert"]
        collapse_key = ("routing_collapse", payload["layer_name"], collapse_expert)
        collapse_active = self._is_collapsed(history, collapse_expert)
        if collapse_active and collapse_key not in self._active_alerts:
            self._active_alerts.add(collapse_key)
            alerts.append(
                self._build_alert(
                    alert_type="routing_collapse",
                    payload=payload,
                    expert_id=collapse_expert,
                    threshold=self.collapse_threshold,
                    window=self.collapse_window,
                )
            )
        elif not collapse_active:
            self._active_alerts.discard(collapse_key)
            for active_key in list(self._active_alerts):
                if active_key[:2] == ("routing_collapse", payload["layer_name"]) and active_key != collapse_key:
                    self._active_alerts.discard(active_key)

        return alerts

    def _is_dead_expert(self, history: deque[dict[str, Any]], expert_id: int) -> bool:
        if len(history) < self.dead_window:
            return False
        samples = list(history)[-self.dead_window :]
        return all(expert_id < len(item["usage"]) and item["usage"][expert_id] <= self.dead_threshold for item in samples)

    def _is_collapsed(self, history: deque[dict[str, Any]], expert_id: int) -> bool:
        if len(history) < self.collapse_window or expert_id < 0:
            return False
        samples = list(history)[-self.collapse_window :]
        return all(
            item["dominant_expert"] == expert_id and item["dominant_share"] >= self.collapse_threshold for item in samples
        )

    def _build_alert(
        self,
        *,
        alert_type: str,
        payload: dict[str, Any],
        expert_id: int,
        threshold: float,
        window: int,
    ) -> dict[str, Any]:
        return {
            "alert_type": alert_type,
            "stage": payload["stage"],
            "step": payload["step"],
            "epoch": payload["epoch"],
            "layer_name": payload["layer_name"],
            "expert_id": int(expert_id),
            "threshold": float(threshold),
            "window": int(window),
            "dominant_share": float(payload["dominant_share"]),
            "aux_loss": float(payload["aux_loss"]),
        }


def export_moe_history_plots(save_dir: str | Path) -> list[Path]:
    """Render static plots from a persisted `routing_history.jsonl` file."""
    save_dir = Path(save_dir)
    history_path = save_dir / "routing_history.jsonl"
    if not history_path.exists() or history_path.stat().st_size == 0:
        return []

    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        return []

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = save_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    by_layer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_layer[record["layer_name"]].append(record)

    for layer_name, layer_records in by_layer.items():
        layer_records.sort(key=lambda item: item["step"])
        steps = [item["step"] for item in layer_records]
        num_experts = max((len(item["usage"]) for item in layer_records), default=0)
        plt.figure(figsize=(8, 4.5))
        for expert_id in range(num_experts):
            values = [item["usage"][expert_id] if expert_id < len(item["usage"]) else 0.0 for item in layer_records]
            plt.plot(steps, values, marker="o", linewidth=1.5, label=f"E{expert_id}")
        plt.title(f"{layer_name} Expert Usage")
        plt.xlabel("step")
        plt.ylabel("usage")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=2 if num_experts > 4 else 1, fontsize=8)
        out_path = plot_dir / f"{_sanitize_layer_name(layer_name)}_usage.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()
        written.append(out_path)

    plt.figure(figsize=(8, 4.5))
    for layer_name, layer_records in by_layer.items():
        layer_records.sort(key=lambda item: item["step"])
        plt.plot(
            [item["step"] for item in layer_records],
            [item["aux_loss"] for item in layer_records],
            marker="o",
            linewidth=1.5,
            label=layer_name,
        )
    plt.title("MoE Aux Loss vs Step")
    plt.xlabel("step")
    plt.ylabel("aux_loss")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    aux_out = plot_dir / "aux_loss_vs_step.png"
    plt.tight_layout()
    plt.savefig(aux_out, dpi=160)
    plt.close()
    written.append(aux_out)

    return written
