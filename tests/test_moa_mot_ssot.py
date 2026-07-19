"""Single-source and serialization contracts for the split MoA/MoT packages."""

import ast
import importlib
from pathlib import Path

import torch

from ultralytics.nn.modules.moa import C2fMoA, MoABlock
from ultralytics.nn.modules.moa.block import MoABlock as CanonicalMoABlock
from ultralytics.nn.modules.moa.heads import _GlobalAttnHead
from ultralytics.nn.modules.mot import C2fMoT, MoTBlock
from ultralytics.nn.modules.mot.block import MoTBlock as CanonicalMoTBlock
from ultralytics.nn.modules.mot.experts import _WindowTransformerExpert
from ultralytics.utils.patches import torch_load


def test_old_module_paths_resolve_to_canonical_objects():
    old_moa = importlib.import_module("ultralytics.nn.modules.moa.moa")
    old_mot = importlib.import_module("ultralytics.nn.modules.mot.mot")

    assert old_moa.MoABlock is CanonicalMoABlock is MoABlock
    assert old_moa._GlobalAttnHead is _GlobalAttnHead
    assert old_mot.MoTBlock is CanonicalMoTBlock is MoTBlock
    assert old_mot._WindowTransformerExpert is _WindowTransformerExpert


def test_compatibility_shims_contain_no_class_definitions():
    root = Path(__file__).parents[1] / "ultralytics" / "nn" / "modules"
    for path in (root / "moa" / "moa.py", root / "mot" / "mot.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]


def test_split_modules_preserve_state_dict_keys():
    old_moa = importlib.import_module("ultralytics.nn.modules.moa.moa")
    old_mot = importlib.import_module("ultralytics.nn.modules.mot.mot")

    moa_old = old_moa.C2fMoA(32, 32, n=1, num_heads=3)
    moa_new = C2fMoA(32, 32, n=1, num_heads=3)
    mot_old = old_mot.C2fMoT(32, 32, n=1, num_heads=2)
    mot_new = C2fMoT(32, 32, n=1, num_heads=2)

    assert tuple(moa_old.state_dict()) == tuple(moa_new.state_dict())
    assert tuple(mot_old.state_dict()) == tuple(mot_new.state_dict())


def test_split_modules_pickle_roundtrip(tmp_path):
    modules = [
        MoABlock(32, num_heads=3).eval(),
        MoTBlock(32, num_heads=2).eval(),
    ]
    path = tmp_path / "mixtures.pt"
    torch.save(modules, path)
    restored = torch_load(path, map_location="cpu", weights_only=False)

    assert isinstance(restored[0], MoABlock)
    assert isinstance(restored[1], MoTBlock)
    assert tuple(restored[0].state_dict()) == tuple(modules[0].state_dict())
    assert tuple(restored[1].state_dict()) == tuple(modules[1].state_dict())
