# Latent Init-Perturb Low-Temperature / Anneal Controls Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Validate whether latent init perturbation becomes useful when paired with a lower router temperature or a stronger annealing curve on the same coco128 training protocol.

**Architecture:** Reuse the existing latent-mixture runtime controller and shared temperature schedule; no trainer logic changes are needed. Add one latent model YAML for a fixed low-temperature control (`temperature=0.25`) and one latent YAML for a true annealed control (`temperature=0.5` -> `0.25`) while keeping `router_init_std=0.02` in both. Add focused regression tests that the configs build and that latent temperature annealing still touches the latent layers. Finally run two matched coco128 experiments: one fixed low-temperature control and one annealed control.

**Tech Stack:** Python, PyTorch, Ultralytics YOLO CLI, pytest, YAML.

---

### Task 1: Add a low-temperature latent init-perturb YAML

**Files:**
- Create: `ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp025.yaml`

**Step 1: Write the config**

Copy `yolo26-master-latent-n-initperturb020.yaml` and change the three latent blocks to use `temperature=0.25` while keeping `residual_init=0.01` and `router_init_std=0.02`.

**Step 2: Verify the file is syntactically valid**

Run: `python - <<'PY' ... DetectionModel(...) ... PY`

Expected: the model builds and exposes 3 latent layers.

### Task 2: Add a focused regression test

**Files:**
- Modify: `tests/test_latent_mixture.py`

**Step 1: Write the failing test**

Add a test that builds the new YAML, asserts 3 `LatentMixture` layers, checks `router_init_std == 0.02`, `temperature == 0.25`, and runs one zero-input forward.

Also add a tiny unit test that `anneal_mixture_temperatures(..., factor=0.95, min_temp=0.2)` updates a latent module temperature in place.

**Step 2: Run the focused test**

Run: `pytest tests/test_latent_mixture.py -v`

Expected: new test passes alongside the existing latent tests.

### Task 3: Run the fixed low-temperature coco128 control

**Files:**
- Use: `ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp025.yaml`

**Step 1: Launch training**

Run coco128 with the same protocol as the prior controls, but set:
- `moa_mot_temperature_factor=1.0`
- `moa_mot_min_temperature=0.25`

**Step 2: Inspect the artifacts**

Verify `results.csv`, `best.pt`, and the final routing snapshot.

### Task 4: Run the annealed latent control

**Files:**
- Create: `ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp05.yaml`
- Use: `ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp05.yaml`

**Step 1: Launch training**

Copy the fixed low-temperature YAML and change the three latent blocks to use `temperature=0.5` while keeping `residual_init=0.01` and `router_init_std=0.02`.

Run the same coco128 protocol, but set:
- `moa_mot_temperature_factor=0.85`
- `moa_mot_min_temperature=0.25`

**Step 2: Compare against the fixed run**

Compare mAP and routing entropy / mean router probs against the fixed low-temperature control and the earlier temp05/init-perturb baselines.

### Task 5: Summarize the experiment decision

**Files:**
- None

**Step 1: Summarize**

Decide whether low temperature or annealing is the more promising next lever than noise on coco128.
