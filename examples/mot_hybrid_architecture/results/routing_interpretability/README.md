# VisDrone MoT routing interpretability provenance

This directory archives the compact outputs of the full token-level MoT routing
analysis. It is separate from the 50-epoch four-arm performance comparison: the
routing study uses the stronger 100-epoch `v10_mot` checkpoint listed below.

## Inputs

- Checkpoint: `/root/autodl-tmp/runs/visdrone_mot_ablation/v10_mot/weights/best.pt`
- Checkpoint SHA-256: `c358b9a22a5c5c4e2148395a1a647ee9d5916c5bb4d512f17c038475405a9f83`
- Dataset: `/root/autodl-tmp/datasets/VisDrone/VisDrone.yaml`
- Split: all 548 validation images
- Training run: 100 epochs, batch 16, imgsz 640, seed 42
- Natural checkpoint validation: mAP50-95 `0.169861`, mAP50 `0.306542`

The performance table in the parent directory remains the controlled 50-epoch
four-arm comparison. Its MoT metric must not be substituted for the value above,
and the routing claims below apply only to this 100-epoch checkpoint.

## Archived outputs

- `heatmap_spatial.png`: per-token LocalConv, Window and Deformable maps.
- `routing_analysis.json`: scene/layer summaries, collapse and forced-routing results.
- `token_level_tests.csv`: within-image and size-matched contrasts with BH-FDR.
- `image_level_occlusion_tests.csv`: density/scale-stratified image contrasts.
- `verdict.json`: machine-readable verdict.
- `exemplars.json`: identities and metadata of visualized examples.

The original 10 MB `spatial_maps.json` is intentionally not duplicated in Git.
Its SHA-256 is `f57adf8ef437cb665189d7249fe199220f83352391a0c747a051c45cff89acc0`.
On the experiment host it is at
`/root/autodl-tmp/mot_routing_interpret/visdrone/spatial_maps.json`.

## Main result

Deformable activation rises with occlusion in specific layers after controlling
the important confounders. The strongest size-matched single-object-token result:

- block: `model.20.m.1`
- pairs: 488
- occluded mean weight: `0.154569`
- clear mean weight: `0.136347`
- absolute lift: `0.018222`
- relative lift: `13.36%`
- Wilcoxon p: `8.16e-05`
- BH-FDR q: `2.35e-04`

This is layer-specific. `model.23.m.1` is fully collapsed to LocalConv (`[1, 0, 0]`)
and must not be used as evidence of specialization. Forcing any single expert lowers
mAP50-95 by 2.98%-5.25%, supporting learned mixed routing despite that final block.

## Reproduce

```bash
python scripts/run_mot_routing_interpret.py \
  --checkpoint /root/autodl-tmp/runs/visdrone_mot_ablation/v10_mot/weights/best.pt \
  --data /root/autodl-tmp/datasets/VisDrone/VisDrone.yaml \
  --dataset-kind visdrone --split val --device 0 --imgsz 640 \
  --output /root/autodl-tmp/mot_routing_interpret/visdrone

python scripts/mot_routing_figures.py \
  --input /root/autodl-tmp/mot_routing_interpret/visdrone
```
