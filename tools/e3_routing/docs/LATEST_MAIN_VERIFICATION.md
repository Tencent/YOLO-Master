# Latest-main compatibility verification

## Source identity

- Upstream commit: `7bfbfd374a6b44720f98366e088f3a962e16321d`
- Upstream tree: `cfdc006e9da0f0daa6ead7aa66af742983ea14bc`
- Live-checkout identity: Git commit/tree plus critical-file SHA-256; no codeload hash is claimed for this refresh.
- The refresh is 30 commits ahead of `3bd0a602...`; 74 paths changed.
- The nine P2-pinned router/model files are byte-identical to the formal `07d3303` runtime.
- Of the changed paths, only `ultralytics/nn/modules/mot/block.py` is in the E3 routing execution area; the change
  replaces tuple-dimension `Tensor.any` with a Torch 1.8-compatible equivalent and does not alter the router contract.
- `ultralytics/engine/trainer.py` changed; the active P1 engine config was refreshed to SHA256
  `d1aa2e8d56371e24014cd934d83b86eccef57d086038b0b365def40ef5c3c7a9`.

## Checks run on the latest-main contribution checkout

| Gate | Result |
| --- | --- |
| Combined phase and CLI-contract tests | 113 passed |
| Ruff on `src`, `tests`, `scripts` | PASS |
| Upstream default-config and master-model tests | 13 passed, one expected MoA head-adjustment warning |
| Five-family real-forward smoke | PASS; P0 modules 6/4/3; MoT/MoA spatial modules 4/4; hook delta 0 |
| Full P0 | PASS; 52 primary + 52 repeat events; batch 2/4 max load delta 0 |
| P1 dashboard over latest-main P0 events | PASS; HTTP 200; 52 events; three families |
| Full P2 committed-source release gate | PASS; five families; 32 spatial captures; 240 arrays; 192 demo views |

A preliminary codeload compatibility run used a local-only config with `require_committed_source=false`, because a
ZIP archive has no Git metadata. The final release gate then ran full P2 from the contribution branch with the active
`require_committed_source=true` configuration. It passed with a clean committed implementation and all 9 source
fingerprints matched; every data, spatial, equivalence, geometry and manifest gate remained enabled.

## Correct interpretation

This verification establishes compatibility with the named upstream commit. It does not relabel the larger formal
P1/P2 numerical experiments, which remain attached to the frozen runtime and their original manifests. Rebase and
rerun this compatibility gate if upstream main moves again before the PR is opened.
