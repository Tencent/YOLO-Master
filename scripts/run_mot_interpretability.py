#!/usr/bin/env python3
"""Orchestrate MoT routing interpretability analysis on COCO 2017 val.

Phases:
  1. Categorize COCO val images into scene types using annotations
  2. Run batch routing diagnosis (diagnose_mot_routing.py)
  3. Visualize exemplar images with per-expert heatmaps
  4. Detect routing collapse
  5. Analyze expert specialization
  6. Causal analysis (forced routing)
  7. Generate summary report
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CHECKPOINT = Path("/root/autodl-tmp/runs/Coco_2017_mot_ablation/v10_mot/weights/best.pt")
COCO_IMAGES = Path("/root/autodl-tmp/datasets/COCO_2017/images/val2017")
COCO_ANNOTATIONS = Path("/root/autodl-tmp/datasets/COCO_2017/annotations/instances_val2017.json")
OUTPUT_DIR = Path("/root/autodl-tmp/Interpretability_of_routing_behavior")

EXPERT_NAMES = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")
MAX_PER_SCENE = 100
EXEMPLARS_PER_SCENE = 3


def bbox_iou_pairs(bboxes):
    """Count pairs with IoU > 0.3 among COCO-format [x,y,w,h] bboxes."""
    n = len(bboxes)
    if n < 2:
        return 0
    boxes = np.array(bboxes, dtype=np.float64)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]

    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            ix1 = max(x1[i], x1[j])
            iy1 = max(y1[i], y1[j])
            ix2 = min(x2[i], x2[j])
            iy2 = min(y2[i], y2[j])
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = areas[i] + areas[j] - inter
            if union > 0 and inter / union > 0.3:
                count += 1
            if count >= 3:
                return count
    return count


def categorize_coco_scenes(annotations_path, images_dir, output_dir, max_per_scene=100):
    """Classify COCO val images into 4 scene categories, create symlink dirs."""
    print("[Phase 1] Loading COCO annotations...")
    with open(annotations_path) as f:
        coco = json.load(f)

    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    dense_small = []
    sparse_large = []
    occluded_irregular = []
    regular = []

    for img_id, anns in anns_by_image.items():
        fname = img_id_to_file.get(img_id)
        if not fname:
            continue
        fpath = images_dir / fname
        if not fpath.exists():
            continue

        num_small = sum(1 for a in anns if a.get("area", 0) < 1024 and not a.get("iscrowd", 0))
        num_large = sum(1 for a in anns if a.get("area", 0) > 9216 and not a.get("iscrowd", 0))
        num_medium = sum(1 for a in anns if 1024 <= a.get("area", 0) <= 9216 and not a.get("iscrowd", 0))
        has_crowd = any(a.get("iscrowd", 0) for a in anns)
        total_non_crowd = sum(1 for a in anns if not a.get("iscrowd", 0))

        if num_small > 10:
            dense_small.append((num_small, fpath))
        elif total_non_crowd <= 3 and num_large >= 1:
            sparse_large.append((num_large, fpath))

        if has_crowd:
            occluded_irregular.append((1, fpath))
        else:
            bboxes = [a["bbox"] for a in anns if not a.get("iscrowd", 0) and "bbox" in a]
            iou_pairs = bbox_iou_pairs(bboxes)
            if iou_pairs >= 3:
                occluded_irregular.append((iou_pairs, fpath))

        if 4 <= num_medium <= 10 and total_non_crowd == num_medium:
            bboxes = [a["bbox"] for a in anns if not a.get("iscrowd", 0) and "bbox" in a]
            if bbox_iou_pairs(bboxes) == 0:
                regular.append((num_medium, fpath))

    dense_small.sort(key=lambda x: -x[0])
    sparse_large.sort(key=lambda x: -x[0])
    occluded_irregular.sort(key=lambda x: -x[0])
    random.seed(42)
    random.shuffle(regular)

    scenes = {
        "dense_small": [p for _, p in dense_small[:max_per_scene]],
        "sparse_large": [p for _, p in sparse_large[:max_per_scene]],
        "occluded_irregular": [p for _, p in occluded_irregular[:max_per_scene]],
        "regular": [p for _, p in regular[:max_per_scene]],
    }

    scenes_dir = output_dir / "scenes"
    for scene_name, paths in scenes.items():
        scene_dir = scenes_dir / scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        for p in paths:
            link = scene_dir / p.name
            if not link.exists():
                link.symlink_to(p)

    for scene_name, paths in scenes.items():
        print(f"  {scene_name}: {len(paths)} images")

    return scenes


def run_diagnose_routing(model_path, scenes_dir, output_dir, device="cuda:0", imgsz=640, batch=4, max_images=100):
    """Run diagnose_mot_routing.py on the organized scene dirs."""
    print("\n[Phase 2] Running batch routing diagnosis...")
    diag_script = ROOT / "scripts" / "diagnose_mot_routing.py"
    project_dir = output_dir / "routing_diagnosis"

    for heatmap_value in ("top1_share", "mean_weight"):
        cmd = [
            sys.executable, str(diag_script),
            "--model", str(model_path),
            "--image-dir", str(scenes_dir),
            "--device", device,
            "--imgsz", str(imgsz),
            "--batch", str(batch),
            "--max-images", str(max_images),
            "--project", str(project_dir),
            "--heatmap-value", heatmap_value,
            "--permutations", "5000",
            "--bootstrap-samples", "5000",
            "--alpha", "0.05",
        ]
        print(f"  Running with --heatmap-value {heatmap_value}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  WARNING: diagnose script failed:\n{result.stderr[:500]}")
        else:
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    print(f"    {line}")


def load_yolo_model(checkpoint, device):
    """Load YOLO model for interpreter use."""
    import torch
    from ultralytics import YOLO
    model = YOLO(str(checkpoint)).model
    model.to(torch.device(device)).eval()
    return model


def preprocess_image(image_path, imgsz=640):
    """Load and preprocess a single image for inference."""
    import torch
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize((imgsz, imgsz))
    tensor = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float().div(255.0)
    return tensor.unsqueeze(0)


def visualize_exemplar_images(model, scenes, output_dir, device, imgsz=640):
    """Generate per-expert routing heatmaps for exemplar images."""
    import torch
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    print("\n[Phase 3] Visualizing exemplar images...")
    interpreter = RoutingInterpreter(model)
    heatmaps_dir = output_dir / "exemplar_heatmaps"

    for scene_name, paths in scenes.items():
        exemplars = paths[:EXEMPLARS_PER_SCENE]
        for img_path in exemplars:
            img_stem = img_path.stem
            out_dir = heatmaps_dir / scene_name / img_stem
            batch = preprocess_image(img_path, imgsz).to(torch.device(device))
            try:
                interpreter.visualize_routing(
                    batch,
                    output_dir=out_dir,
                    input_image=img_path,
                )
                print(f"  {scene_name}/{img_stem}: OK")
            except Exception as e:
                print(f"  {scene_name}/{img_stem}: FAILED - {e}")

    torch.cuda.empty_cache()


def run_collapse_detection(model, images_dir, output_dir, device, num_samples=500, imgsz=640, batch_size=8):
    """Detect routing collapse across the val set."""
    import torch
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    print("\n[Phase 4] Running collapse detection...")
    interpreter = RoutingInterpreter(model)

    all_images = sorted(images_dir.glob("*.jpg"))
    random.seed(42)
    sample = random.sample(all_images, min(num_samples, len(all_images)))

    all_heatmaps = {}
    processed = 0
    for i in range(0, len(sample), batch_size):
        batch_paths = sample[i:i + batch_size]
        tensors = [preprocess_image(p, imgsz) for p in batch_paths]
        batch = torch.cat(tensors, dim=0).to(torch.device(device))
        try:
            heatmaps = interpreter.capture_routing(batch)
            for name, hm in heatmaps.items():
                if name not in all_heatmaps:
                    all_heatmaps[name] = hm
                else:
                    existing = all_heatmaps[name]
                    all_heatmaps[name] = type(hm)(
                        layer_name=hm.layer_name,
                        module_type=hm.module_type,
                        probabilities=torch.cat([existing.probabilities, hm.probabilities], dim=0),
                        assignments=torch.cat([existing.assignments, hm.assignments], dim=0),
                    )
        except Exception as e:
            print(f"  Batch {i} failed: {e}")
            continue
        processed += len(batch_paths)
        if processed % 100 == 0:
            print(f"  Processed {processed}/{len(sample)} images...")

    collapse_reports = interpreter.detect_routing_collapse(heatmaps=all_heatmaps)

    report_data = {}
    for layer_name, report in collapse_reports.items():
        report_data[layer_name] = {
            "expert_usage": [float(x) for x in report.expert_usage],
            "dominant_expert": int(report.dominant_expert),
            "dominant_share": float(report.dominant_share),
            "normalized_gini": float(report.normalized_gini),
            "normalized_entropy": float(report.normalized_entropy),
            "dead_experts": list(report.dead_experts),
            "collapsed": bool(report.collapsed),
        }

    out_path = output_dir / "collapse_report.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Wrote {out_path}")

    for layer, data in report_data.items():
        status = "COLLAPSED" if data["collapsed"] else "OK"
        print(f"  {layer}: gini={data['normalized_gini']:.3f} entropy={data['normalized_entropy']:.3f} "
              f"usage={[f'{u:.2f}' for u in data['expert_usage']]} [{status}]")

    torch.cuda.empty_cache()
    return report_data


def run_specialization_analysis(model, images_dir, output_dir, device, num_samples=1000, imgsz=640, batch_size=8):
    """Analyze expert specialization over the val set."""
    import torch
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    print("\n[Phase 5] Running expert specialization analysis...")
    interpreter = RoutingInterpreter(model)

    all_images = sorted(images_dir.glob("*.jpg"))
    random.seed(123)
    sample = random.sample(all_images, min(num_samples, len(all_images)))

    def dataset_iter():
        for i in range(0, len(sample), batch_size):
            batch_paths = sample[i:i + batch_size]
            tensors = [preprocess_image(p, imgsz) for p in batch_paths]
            yield torch.cat(tensors, dim=0).to(torch.device(device))

    try:
        reports = interpreter.analyze_expert_specialization(
            dataset_iter(),
            num_samples=num_samples,
            max_batches=num_samples // batch_size + 1,
        )
        report_data = {}
        for layer_name, report in reports.items():
            report_data[layer_name] = {
                "module_type": report.module_type,
                "num_experts": report.num_experts,
                "num_samples": report.num_samples,
                "mean_usage": [float(x) for x in report.mean_usage] if hasattr(report.mean_usage, '__iter__') else [],
                "dominant_samples": [int(x) for x in report.dominant_samples] if hasattr(report.dominant_samples, '__iter__') else [],
            }
            if hasattr(report, 'feature_signatures') and report.feature_signatures:
                sigs = {}
                for k, v in report.feature_signatures.items():
                    if hasattr(v, '__iter__'):
                        sigs[k] = [float(x) for x in v]
                    else:
                        sigs[k] = float(v)
                report_data[layer_name]["feature_signatures"] = sigs

        out_path = output_dir / "specialization_report.json"
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"  Wrote {out_path}")

        for layer, data in report_data.items():
            usage = data.get("mean_usage", [])
            print(f"  {layer}: mean_usage={[f'{u:.3f}' for u in usage]}")

    except Exception as e:
        print(f"  Specialization analysis failed: {e}")
        report_data = {"error": str(e)}
        out_path = output_dir / "specialization_report.json"
        with open(out_path, "w") as f:
            json.dump(report_data, f, indent=2)

    torch.cuda.empty_cache()
    return report_data


def run_causal_analysis(model, scenes, output_dir, device, imgsz=640):
    """Force routing to each expert and measure output divergence."""
    import torch
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    print("\n[Phase 6] Running causal analysis...")
    interpreter = RoutingInterpreter(model)

    target_layers = []
    from ultralytics.nn.modules.mot import MoTBlock
    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            target_layers.append(name)
    if not target_layers:
        print("  No MoTBlock layers found, skipping.")
        return {}

    layers_to_test = [target_layers[0]]
    if len(target_layers) >= 3:
        layers_to_test.append(target_layers[len(target_layers) // 2])
    if len(target_layers) >= 2:
        layers_to_test.append(target_layers[-1])
    layers_to_test = list(dict.fromkeys(layers_to_test))

    exemplar_paths = []
    for scene_name, paths in scenes.items():
        if paths:
            exemplar_paths.append(paths[0])
    exemplar_paths = exemplar_paths[:4]

    tensors = [preprocess_image(p, imgsz) for p in exemplar_paths]
    batch = torch.cat(tensors, dim=0).to(torch.device(device))

    report_data = {}
    for layer_name in layers_to_test:
        report_data[layer_name] = {}
        for expert_idx in range(3):
            try:
                result = interpreter.routing_causal_analysis(batch, layer_name, expert_idx)
                report_data[layer_name][f"expert_{expert_idx}"] = {
                    "expert_name": EXPERT_NAMES[expert_idx],
                    "mean_absolute_difference": float(result.mean_absolute_difference),
                    "root_mean_square_difference": float(result.root_mean_square_difference),
                    "max_absolute_difference": float(result.max_absolute_difference),
                    "cosine_similarity": float(result.cosine_similarity),
                }
            except Exception as e:
                report_data[layer_name][f"expert_{expert_idx}"] = {"error": str(e)}
        cos_sims = []
        for v in report_data[layer_name].values():
            if "cosine_similarity" in v:
                cos_sims.append(v["cosine_similarity"])
        print(f"  {layer_name}: cosine_similarity per expert = {[f'{c:.4f}' for c in cos_sims]}")

    out_path = output_dir / "causal_report.json"
    with open(out_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"  Wrote {out_path}")

    torch.cuda.empty_cache()
    return report_data


def generate_report(output_dir, scenes, collapse_data, specialization_data, causal_data):
    """Generate a comprehensive markdown summary report."""
    print("\n[Phase 7] Generating summary report...")

    lines = [
        "# MoT Routing Behavior Interpretability Analysis Report",
        "",
        "## Experiment Setup",
        "",
        "| Item | Value |",
        "|------|-------|",
        f"| Model | YOLO-Master-v0.10-MoT-N |",
        f"| Checkpoint | `{CHECKPOINT}` |",
        "| Dataset | COCO 2017 val (5000 images) |",
        "| Experts | LocalConvTransformer, WindowTransformer, DeformableTransformer |",
        "| Top-K | 2 (P3/P4 stages), 1 (P5 stage) |",
        "",
        "## Phase 1: Scene Categorization",
        "",
        "| Scene | Count | Description |",
        "|-------|-------|-------------|",
        f"| dense_small | {len(scenes.get('dense_small', []))} | >10 small objects (area<1024) |",
        f"| sparse_large | {len(scenes.get('sparse_large', []))} | <=3 objects, at least 1 large (area>9216) |",
        f"| occluded_irregular | {len(scenes.get('occluded_irregular', []))} | iscrowd=1 or high pairwise IoU |",
        f"| regular | {len(scenes.get('regular', []))} | 4-10 medium objects, well-separated |",
        "",
    ]

    # Routing diagnosis results
    diag_dir = output_dir / "routing_diagnosis"
    scenario_csv = diag_dir / "mot_routing_scenarios.csv"
    deform_csv = diag_dir / "mot_deformable_activation_check.csv"

    if scenario_csv.exists():
        lines.append("## Phase 2: Scene-Based Routing Statistics")
        lines.append("")
        lines.append("### Expert Activation by Scene (top1_share)")
        lines.append("")
        import csv
        with open(scenario_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            lines.append("| Scene | Expert | top1_share | mean_weight |")
            lines.append("|-------|--------|-----------|-------------|")
            for row in rows:
                lines.append(f"| {row.get('scene','')} | {row.get('expert','')} | "
                           f"{float(row.get('top1_share_mean', 0)):.4f} | "
                           f"{float(row.get('mean_weight', 0)):.4f} |")
        lines.append("")

    if deform_csv.exists():
        lines.append("### DeformableTransformer Activation Test")
        lines.append("")
        lines.append("H1: DeformableTransformer activates more in occluded/irregular scenes")
        lines.append("")
        with open(deform_csv) as f:
            rows = list(csv.DictReader(f))
        if rows:
            lines.append("| Metric | Baseline | Irregular Mean | Baseline Mean | Diff | p-value | Significant |")
            lines.append("|--------|----------|---------------|---------------|------|---------|-------------|")
            for row in rows:
                lines.append(
                    f"| {row.get('metric','')} | {row.get('baseline','')} | "
                    f"{row.get('irregular_mean','')} | {row.get('baseline_mean','')} | "
                    f"{row.get('mean_diff','')} | {row.get('permutation_p_value_one_sided','')} | "
                    f"{row.get('deformable_significantly_higher','')} |"
                )
        lines.append("")

    # Collapse detection
    if collapse_data and "error" not in collapse_data:
        lines.append("## Phase 4: Routing Collapse Detection")
        lines.append("")
        lines.append("| Layer | Gini | Entropy | Expert Usage | Dead | Collapsed |")
        lines.append("|-------|------|---------|-------------|------|-----------|")
        for layer, data in collapse_data.items():
            usage_str = ", ".join(f"{u:.3f}" for u in data["expert_usage"])
            lines.append(
                f"| {layer} | {data['normalized_gini']:.3f} | "
                f"{data['normalized_entropy']:.3f} | [{usage_str}] | "
                f"{data['dead_experts']} | {'YES' if data['collapsed'] else 'No'} |"
            )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("- Gini close to 0 = uniform usage, close to 1 = one expert dominates")
        lines.append("- Entropy close to 1 = uniform, close to 0 = collapsed")
        lines.append("- Dead experts = usage < 1%")
        lines.append("")

    # Specialization
    if specialization_data and "error" not in specialization_data:
        lines.append("## Phase 5: Expert Specialization")
        lines.append("")
        lines.append("| Layer | Expert 0 (LocalConv) | Expert 1 (Window) | Expert 2 (Deformable) |")
        lines.append("|-------|---------------------|-------------------|----------------------|")
        for layer, data in specialization_data.items():
            usage = data.get("mean_usage", [0, 0, 0])
            if len(usage) >= 3:
                lines.append(f"| {layer} | {usage[0]:.3f} | {usage[1]:.3f} | {usage[2]:.3f} |")
        lines.append("")

    # Causal analysis
    if causal_data and "error" not in causal_data:
        lines.append("## Phase 6: Causal Analysis (Forced Routing)")
        lines.append("")
        lines.append("Cosine similarity between natural output and forced-single-expert output:")
        lines.append("- 1.0 = expert has no unique contribution (outputs identical)")
        lines.append("- <0.9 = expert produces meaningfully different features")
        lines.append("")
        lines.append("| Layer | LocalConv cos | Window cos | Deformable cos |")
        lines.append("|-------|--------------|------------|----------------|")
        for layer, experts in causal_data.items():
            cos_vals = []
            for i in range(3):
                e = experts.get(f"expert_{i}", {})
                cos_vals.append(f"{e.get('cosine_similarity', 'N/A'):.4f}" if "cosine_similarity" in e else "ERR")
            lines.append(f"| {layer} | {cos_vals[0]} | {cos_vals[1]} | {cos_vals[2]} |")
        lines.append("")

    # Conclusions
    lines.append("## Key Findings")
    lines.append("")

    any_collapsed = any(d.get("collapsed", False) for d in collapse_data.values()) if collapse_data else False
    if any_collapsed:
        lines.append("1. **Routing collapse detected**: One or more layers show expert usage collapse, "
                   "indicating the router failed to learn meaningful differentiation.")
    else:
        lines.append("1. **No routing collapse**: All layers show reasonable expert utilization.")

    lines.append("")
    lines.append("2. See `routing_diagnosis/mot_deformable_activation_check.csv` for statistical "
               "validation of the DeformableTransformer specialization hypothesis.")
    lines.append("")
    lines.append("3. See `exemplar_heatmaps/` for spatial routing visualizations per scene type.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by run_mot_interpretability.py*")

    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"  Wrote {report_path}")


def main():
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    imgsz = 640

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1
    scenes = categorize_coco_scenes(COCO_ANNOTATIONS, COCO_IMAGES, OUTPUT_DIR, MAX_PER_SCENE)

    # Phase 2
    scenes_dir = OUTPUT_DIR / "scenes"
    run_diagnose_routing(CHECKPOINT, scenes_dir, OUTPUT_DIR, device=device, imgsz=imgsz, batch=4, max_images=MAX_PER_SCENE)

    # Phase 3-6 need the model loaded
    print("\n  Loading model...")
    model = load_yolo_model(CHECKPOINT, device)

    # Phase 3
    visualize_exemplar_images(model, scenes, OUTPUT_DIR, device, imgsz)

    # Phase 4
    collapse_data = run_collapse_detection(model, COCO_IMAGES, OUTPUT_DIR, device, num_samples=500, imgsz=imgsz)

    # Phase 5
    specialization_data = run_specialization_analysis(model, COCO_IMAGES, OUTPUT_DIR, device, num_samples=1000, imgsz=imgsz)

    # Phase 6
    causal_data = run_causal_analysis(model, scenes, OUTPUT_DIR, device, imgsz)

    # Phase 7
    generate_report(OUTPUT_DIR, scenes, collapse_data, specialization_data, causal_data)

    print("\n" + "=" * 60)
    print("DONE! All results saved to:")
    print(f"  {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
