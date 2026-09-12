"""Exercise actual autocast head outputs through assignment and complete detection loss."""

from types import SimpleNamespace

import pytest
import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel  # noqa: F401 - normal loss import initialization
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("mode", ["pure", "fixed", "adaptive"])
@pytest.mark.parametrize("small_scene", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable; CPU BF16 is not a substitute for CUDA FP16"
            ),
        ),
    ],
)
def test_actual_autocast_assignment_and_detection_gradients(mode, device, small_scene):
    """Compare masks, top-k, counts, three losses and head gradients on a deterministic non-tied scene."""
    torch.manual_seed(14)
    head = torch.nn.Conv1d(4, 65, 1).to(device)
    if small_scene:
        # Decode boxes on the same pixel scale as the tiny GTs; random wide boxes can have zero CIoU.
        with torch.no_grad():
            head.weight[:64].mul_(0.1)
            bias = head.bias[:64].reshape(4, 16)
            bias.fill_(-4.0)
            bias[:, 0] = 4.0
            bias[:, 1] = 3.0
    head.args = get_cfg(
        overrides={
            "stal_candidate_mode": mode,
            "stal_warmup_epochs": 0.0,
            **({"stal_small_topk": 6, "stal_small_topk_min": 6} if small_scene else {}),
        }
    )
    head.model = [SimpleNamespace(stride=torch.tensor([8.0, 16.0, 32.0], device=device), nc=1, reg_max=16)]
    features = torch.randn(1, 4, 84, device=device)
    batch = {
        "batch_idx": torch.tensor([0.0, 0.0], device=device),
        "cls": torch.zeros(2, 1, device=device),
        "bboxes": torch.tensor(
            [[0.3125, 0.3125, 0.05, 0.05], [0.6875, 0.6875, 0.06, 0.06]]
            if small_scene
            else [[0.25, 0.25, 0.15, 0.15], [0.70, 0.70, 0.2, 0.2]],
            device=device,
        ),
        "epoch": 10,
    }
    results = []
    for amp in (False, True):
        head.zero_grad(set_to_none=True)
        criterion = v8DetectionLoss(head)
        selected = []
        original = criterion.assigner.get_pos_mask

        def capture(*args, original=original, selected=selected, criterion=criterion, **kwargs):
            if small_scene:
                boxes, anchors, mask = args[3:6]
                gate = criterion.assigner.small_target_mask(boxes, kwargs["image_size"])
                assert gate.all(), "Fixture must activate the small-target area gate"
                expanded = criterion.assigner.select_candidates_in_gts(anchors, boxes, mask, **kwargs)
                base = criterion.assigner.select_candidates_in_gts(
                    anchors, boxes, mask, **kwargs, relaxation_override=0.0
                )
                if mode == "adaptive":
                    assert (expanded & ~base).any(), "Adaptive expansion must add candidates beyond fixed STAL"
            result = original(*args, **kwargs)
            selected.append(result[0].detach().clone())
            return result

        criterion.assigner.get_pos_mask = capture
        dtype = torch.float16 if device == "cuda" else torch.bfloat16
        with torch.autocast(device, dtype=dtype, enabled=amp):
            output = head(features)
            assert output.dtype == (dtype if amp else torch.float32)
            predictions = {
                "boxes": output[:, :64],
                "scores": output[:, 64:],
                "feats": [torch.zeros(1, 1, s, s, device=device) for s in (8, 4, 2)],
            }
            assigned, loss, _ = criterion.get_assigned_targets_and_loss(predictions, batch)
        loss.sum().backward()
        gradients = torch.cat([p.grad.flatten() for p in head.parameters()])
        assert torch.isfinite(loss).all() and torch.isfinite(gradients).all()
        assert assigned[0].any()
        results.append((assigned[0].clone(), assigned[1].clone(), selected[0], loss.detach(), gradients.clone()))
    for index in range(3):
        assert torch.equal(results[0][index], results[1][index])
    # BF16 is coarser than FP16. This bounded tolerance is for the synthetic fixture, not a training-equivalence claim.
    for index in (3, 4):
        torch.testing.assert_close(results[0][index], results[1][index], rtol=0.05, atol=0.01)


def test_adaptive_amp_scene_rejects_disabled_expansion(monkeypatch):
    """The new fixture must fail if the adaptive geometry silently becomes fixed STAL."""
    monkeypatch.setattr(TaskAlignedAssigner, "stal_relaxation_at_epoch", lambda self, epoch: 0.0)
    with pytest.raises(AssertionError, match="Adaptive expansion must add candidates"):
        test_actual_autocast_assignment_and_detection_gradients("adaptive", "cpu", True)
