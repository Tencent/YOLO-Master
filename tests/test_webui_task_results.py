"""Check actual UI result parsing without requiring Gradio or downloaded models."""

import ast
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from ultralytics.engine.results import Results


@pytest.mark.parametrize("task,count", [("obb", 1), ("detect", 1), ("empty", 0), ("cls", 2)])
def test_webui_preserves_task_results(task, count):
    """Use real Results containers and the unchanged public inference method signature."""
    tree = ast.parse((Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8"))
    ui = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "YOLO_Master_WebUI")
    method = next(node for node in ui.body if isinstance(node, ast.FunctionDef) and node.name == "inference")
    scope = {"np": np, "pd": pd, "cv2": cv2, "Path": Path, "List": list}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "app.py", "exec"), scope)  # noqa: S102 - trusted local AST
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    fields = (
        {"obb": torch.tensor([[16.0, 16.0, 8.0, 6.0, 0.2, 0.9, 0.0]])}
        if task == "obb"
        else (
            {"probs": torch.tensor([0.8, 0.2])}
            if task == "cls"
            else {"boxes": torch.tensor([[1.0, 1.0, 8.0, 8.0, 0.9, 0.0]]) if task == "detect" else torch.empty(0, 6)}
        )
    )
    names = {0: "first", 1: "second"}
    result = Results(image, "fixture.jpg", names, **fields)
    result.speed = {"inference": 1.0}

    class Model:
        def __call__(self, *args, **kwargs):
            return [result]

    model = Model()
    model.names = names
    manager = SimpleNamespace(
        load_model=lambda *args: model, current_model_path="fixture.pt", get_current_model_info=lambda: "cpu"
    )
    _, table, summary = scope["inference"](
        SimpleNamespace(model_manager=manager), task, image, "fixture.pt", "", 0.25, 0.7, "cpu", 100, 2, True, []
    )
    assert len(table) == count
    assert f"{'Classes shown' if task == 'cls' else 'Objects'}:** {count}" in summary
    if task == "obb":
        assert table.iloc[0]["angle (rad)"] == pytest.approx(0.2)
    if task == "cls":
        assert table.iloc[0]["Class Name"] == "first"
