# Reproducibility and version policy

## Two gates, two meanings

The configurations under `configs/frozen/` are frozen to YOLO-Master commit
`07d330325b5a26b75aabfc75389f9bcbc0d40245` and archive SHA256
`774abaa264028f1aca4b6d14c3db249f8a75bc253b4131c272f6dc8a8c744c49`. Their numerical results must not be
silently relabelled as results from another source revision.

The independent `scripts/latest_main_smoke.py` gate answers a different question: whether the contribution still
loads the current repository models, preserves outputs and observes the same spatial/non-spatial contracts. Its
output records the actual source commit and tree. Run it again after every rebase.

## Formal reproduction

The original phase repositories preserve resolved configs, logs, inputs, raw events, reports and manifests. To
reproduce the archived numbers, use `configs/frozen/`, place the frozen source at the configured `runtime_root`,
or update only that path to an independently verified checkout of the same commit. Do not change the reference,
tree or source fingerprints. The phase-level active configs target the containing current-main checkout instead.

## Latest-main gate

From the repository root:

```bat
cd tools\e3_routing
set PYTHONPATH=%CD%\src
python scripts\latest_main_smoke.py --repo-root ..\.. --output artifacts\latest-main-smoke.json
```

PASS requires:

1. real forward evidence for MoE, MoT and Latent P0 telemetry;
2. unchanged hooked model outputs;
3. valid `[B,E,H,W]` records for MoT and MoA;
4. explicit non-spatial decisions for MoE, Latent and MoLoRA;
5. recorded current commit and tree.
