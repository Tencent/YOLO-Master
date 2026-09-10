# E3 result summary

## P0 — unified structured routing events

The formal four-image coco8 validation run covered six MoE, four MoT and three Latent modules. It emitted 52
schema-valid events; an independent repeat matched all 52 stable fingerprints. Batch sizes 2 and 4 produced a
maximum aggregate expert-load difference of 0 relative to batch size 1. These checks establish telemetry and
aggregation correctness, not learned routing quality.

## P1 — live view and overhead

| Family | Pairs | Median slowdown | 95% bootstrap CI | Result |
| --- | ---: | ---: | ---: | --- |
| MoE | 30 | 2.113% | [-2.514%, 8.361%] | PASS / STRONG |
| MoT | 48 | -0.050% | [-6.374%, 9.451%] | PASS / STRONG |
| Latent | 30 | -5.741% | [-16.195%, 1.570%] | PASS / STRONG |

All medians and CI upper bounds are below the 10% gate. The 108 pairs include detection loss, backward, gradient
clipping, grouped SGD and JSONL write/flush; loss, gradients, model state and optimizer state remain equivalent.
The separate Trainer integration layer produced 1,040 events and HTTP 200, but its very short CPU epoch timing is
explicitly descriptive rather than verdict-eligible.

## P2 — truthful spatial routing lens

The five-family audit supports original-image overlays only for MoT and MoA. The primary run generated 32 true
spatial captures, 240 raw arrays and 192 selectable views. Hooks changed model output by 0; single-versus-batch
probabilities differed by 0 and the maximum logit difference was `1.24e-10`.

The extended evidence covers 32 images and three seeds. Three perturbation families at three strengths generated
3,840 captures; seed-averaged probability response was non-decreasing for 32/32 images in every family. Held-middle
span-normalized median errors were 0.115% for brightness, 0.161% for contrast and 2.647% for blur. A preregistered
blur follow-up found no Holm-significant association for content fraction, luminance or edge total variation; the
smallest adjusted p-value was 0.318. This is retained as a controlled null result.

## Integrity summary

- Tests: P0 16, P1 19, P2 73; 108 total, all passing in the phase repositories.
- Formal manifests: seven key runs, 395 entries, independently recomputed with zero mismatch.
- Combined contribution suite: 113 tests, all passing.
- Demo video prepared for PR attachment: 1:55, SHA256
  `6a686af4f3d2e307a8c4b08d862c80da29a2dceab364988df670692a36e4149f`.
- Latest-main compatibility: upstream `7bfbfd374a6b44720f98366e088f3a962e16321d`; five-family smoke PASS;
  full P0 52/52 replay and batch gate PASS; full P2 32 captures/192 demo views PASS; 13 upstream model/config
  regression tests PASS.
