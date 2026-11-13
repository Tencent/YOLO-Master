#!/usr/bin/env python3
"""Unit tests for the OV-Distill pipeline: GAM parsing, IoU voting, negative extraction, format conversion, resume."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.annotate import load_done_keys, run_annotate  # noqa: E402
from scripts.ov_distill.clean import box_iou, cluster_boxes, fuse_cluster, run_clean  # noqa: E402
from scripts.ov_distill.concepts import (  # noqa: E402
    FileTagger,
    MockTagger,
    build_tagger,
    run_concepts,
    sample_negatives,
)
from scripts.ov_distill.convert import run_convert  # noqa: E402
from scripts.ov_distill.smoke import run_smoke  # noqa: E402
from scripts.ov_distill.teachers import (  # noqa: E402
    GroundingPIClient,
    MockTeacher,
    WeVisOneClient,
    build_teacher,
    parse_gam_multi,
    parse_gam_output,
)


class TestGAMParsing:
    """Parsing of the GAM spatial-token protocol."""

    def test_parse_boxes(self):
        text = "<|object_ref_start|>car<|object_ref_end|><|box_start|><0><0><999><999><|box_end|>"
        res = parse_gam_output(text, "car", width=999, height=999)
        assert not res.absent
        assert len(res.boxes) == 1
        assert res.boxes[0] == pytest.approx([0.0, 0.0, 999.0, 999.0])

    def test_parse_none_as_absent(self):
        text = "<|object_ref_start|>ufo<|object_ref_end|><|box_start|>None<|box_end|>"
        res = parse_gam_output(text, "ufo", width=100, height=100)
        assert res.absent
        assert res.boxes == []

    def test_coordinate_scaling(self):
        text = "<|object_ref_start|>car<|object_ref_end|><|box_start|><0><0><499><999><|box_end|>"
        res = parse_gam_output(text, "car", width=1000, height=500)
        assert res.boxes[0] == pytest.approx([0.0, 0.0, 499 / 999 * 1000, 500.0])


class TestWeVisOneParsing:
    """Parsing of the WeVisOne-VL <ref>/<box> protocol (coordinates normalized to 0-1000)."""

    def test_parse_detection_boxes(self):
        text = "<ref>cat</ref><box><100><200><300><400></box><ref>dog</ref><box><0><0><500><999></box>"
        parsed = WeVisOneClient.parse_output(text, width=1000, height=1000)
        assert parsed["cat"] == [pytest.approx([100.0, 200.0, 300.0, 400.0])]
        assert parsed["dog"] == [pytest.approx([0.0, 0.0, 500.0, 999.0])]

    def test_parse_none_rejection(self):
        assert WeVisOneClient.parse_output("None", 100, 100) == {}
        assert WeVisOneClient.parse_output("", 100, 100) == {}

    def test_coordinate_mapping_to_pixels(self):
        text = "<ref>cat</ref><box><0><0><1000><1000></box>"
        parsed = WeVisOneClient.parse_output(text, width=640, height=480)
        assert parsed["cat"][0] == pytest.approx([0.0, 0.0, 640.0, 480.0])

    def test_truncated_tail_tolerated(self):
        text = "<ref>cat</ref><box><100><100><200><200></box><ref>dog</ref><box><10><10>"
        parsed = WeVisOneClient.parse_output(text, width=100, height=100)
        assert "cat" in parsed and "dog" not in parsed  # truncation only drops the incomplete tail segment

    def test_build_prompt_matches_official(self):
        prompt = WeVisOneClient.build_prompt(["cat", "dog"])
        assert prompt == "Locate all the instances that match the following description: cat. dog."


class TestGroundingPIParsing:
    """Parsing of the GroundingPI GAM dense-bbox protocol and its prompt template."""

    def test_parse_multi_labels_and_boxes(self):
        text = (
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|><100><200><300><400><|box_end|>, "
            "<|object_ref_start|>dog<|object_ref_end|><|box_start|><0><0><999><999>,<10><10><50><60><|box_end|>"
        )
        parsed = parse_gam_multi(text, width=999, height=999)
        assert parsed["cat"] == [pytest.approx([100.0, 200.0, 300.0, 400.0])]
        assert len(parsed["dog"]) == 2
        assert parsed["dog"][0] == pytest.approx([0.0, 0.0, 999.0, 999.0])

    def test_parse_none_per_label(self):
        text = (
            "<|object_ref_start|>cat<|object_ref_end|><|box_start|>None<|box_end|>, "
            "<|object_ref_start|>dog<|object_ref_end|><|box_start|><1><2><3><4><|box_end|>"
        )
        parsed = parse_gam_multi(text, width=100, height=100)
        assert parsed["cat"] == []
        assert len(parsed["dog"]) == 1

    def test_build_prompt_uses_c_separator(self):
        prompt = GroundingPIClient.build_prompt(["cat", "dog"])
        assert prompt == "Locate all the instances that match the following categories: cat</c>dog."


class TestTeacherFactory:
    """Building teachers from spec strings."""

    def test_mock(self):
        t = build_teacher("mock:7")
        assert isinstance(t, MockTeacher)
        assert t.name == "mock-7"

    def test_gam_with_model(self):
        t = build_teacher("gam:http://127.0.0.1:8000/v1:groundingpi")
        assert t.base_url == "http://127.0.0.1:8000/v1"
        assert t.model == "groundingpi"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="unknown teacher spec"):
            build_teacher("dino:xxx")


class TestVoting:
    """IoU clustering and fusion."""

    def test_box_iou(self):
        assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
        assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_cluster_aligns_cross_teacher(self):
        members = [("t1", [0, 0, 10, 10]), ("t2", [1, 1, 11, 11]), ("t3", [90, 90, 100, 100])]
        clusters = cluster_boxes(members, iou_thr=0.5)
        sizes = sorted(len(c) for c in clusters)
        assert sizes == [1, 2]

    def test_same_teacher_not_double_counted(self):
        members = [("t1", [0, 0, 10, 10]), ("t1", [1, 1, 11, 11])]
        clusters = cluster_boxes(members, iou_thr=0.5)
        assert max(len({t for t, _ in c}) for c in clusters) == 1

    def test_fuse_cluster_mean(self):
        box, mean_iou = fuse_cluster([("t1", [0, 0, 10, 10]), ("t2", [2, 0, 12, 10])])
        assert box == pytest.approx([1, 0, 11, 10])
        assert 0 < mean_iou < 1


class TestPipeline:
    """annotate -> clean -> convert integration behavior."""

    def _annotate(self, tmp_path: Path) -> Path:
        img_dir = tmp_path / "imgs"
        img_dir.mkdir()
        from PIL import Image

        for i in range(2):
            Image.new("RGB", (160, 120), (10, 20, 30)).save(img_dir / f"a{i}.jpg")
        manifest = tmp_path / "manifest.jsonl"
        with manifest.open("w", encoding="utf-8") as f:
            for img in sorted(img_dir.glob("*.jpg")):
                f.write(json.dumps({"image": str(img), "queries": ["cat", "zz_absent_ufo"]}) + "\n")
        raw = tmp_path / "raw.jsonl"
        run_annotate(manifest=manifest, teacher_specs=["mock:0", "mock:1"], out=raw)
        return raw

    def test_annotate_resume_skips_done(self, tmp_path):
        raw = self._annotate(tmp_path)
        done = load_done_keys(raw)
        assert len(done) == 2 * 2 * 2  # 2 images x 2 teachers x 2 queries
        stats = run_annotate(manifest=tmp_path / "manifest.jsonl", teacher_specs=["mock:0", "mock:1"], out=raw)
        assert stats["skipped"] == 8
        assert stats["records"] == 0

    def test_clean_votes_and_negatives(self, tmp_path):
        raw = self._annotate(tmp_path)
        stats = run_clean(raw=raw, out=tmp_path / "cleaned.jsonl", iou_thr=0.5, min_votes=2)
        assert stats["kept_boxes"] > 0
        assert stats["negatives"] == 2  # one zz_absent_ufo per image
        with (tmp_path / "cleaned.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                assert rec["votes"] == 2
                assert rec["bucket"] == "high"

    def test_clean_tolerates_non_abstaining_teacher(self, tmp_path):
        """One of three teachers never abstains (its hallucinated box is voted out): 2 absent votes must still record a negative."""
        img = tmp_path / "x.jpg"
        from PIL import Image

        Image.new("RGB", (100, 100)).save(img)
        raw = tmp_path / "raw.jsonl"
        rows = [
            {"image": str(img), "teacher": "a", "query": "ufo", "boxes": [], "absent": True, "ok": True},
            {"image": str(img), "teacher": "b", "query": "ufo", "boxes": [], "absent": True, "ok": True},
            {"image": str(img), "teacher": "c", "query": "ufo", "boxes": [[1, 1, 50, 50]], "absent": False, "ok": True},
        ]
        with raw.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        stats = run_clean(raw=raw, out=tmp_path / "cleaned.jsonl", iou_thr=0.5, min_votes=2)
        assert stats["kept_boxes"] == 0  # the lone single-teacher box is voted out
        assert stats["negatives"] == 1  # absent 2/3 >= n-1 -> recorded as a negative
        # strict mode (neg_min_absent=3) records nothing
        stats_strict = run_clean(raw=raw, out=tmp_path / "cleaned2.jsonl", iou_thr=0.5, min_votes=2, neg_min_absent=3)
        assert stats_strict["negatives"] == 0

    def test_convert_outputs_yolo_format(self, tmp_path):
        raw = self._annotate(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        stats_clean = run_clean(raw=raw, out=cleaned, iou_thr=0.5, min_votes=2)
        out = tmp_path / "dataset"
        stats = run_convert(cleaned=cleaned, out=out)
        assert stats["boxes"] == stats_clean["kept_boxes"]
        cats = json.loads((out / "categories.json").read_text(encoding="utf-8"))
        assert {c["name"] for c in cats} == {"cat"}  # zz_absent_ufo went to the negatives
        label_files = list((out / "labels").glob("*.txt"))
        assert label_files
        for line in label_files[0].read_text(encoding="utf-8").splitlines():
            parts = line.split()
            assert len(parts) == 5
            assert all(0.0 <= float(v) <= 1.0 for v in parts[1:])

    def test_convert_outputs_grounding_json(self, tmp_path):
        """grounding format: GoldG single JSON; tokens_positive spans must slice back to the class name."""
        raw = self._annotate(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        stats_clean = run_clean(raw=raw, out=cleaned, iou_thr=0.5, min_votes=2)
        out = tmp_path / "ds_grounding"
        stats = run_convert(cleaned=cleaned, out=out, fmt="grounding")
        assert stats["boxes"] == stats_clean["kept_boxes"]
        payload = json.loads((out / "grounding_train.json").read_text(encoding="utf-8"))
        assert payload["images"] and payload["annotations"]
        images = {im["id"]: im for im in payload["images"]}
        for ann in payload["annotations"]:
            img = images[ann["image_id"]]
            x, y, w, h = ann["bbox"]
            assert w > 0 and h > 0
            assert 0 <= x < img["width"] and 0 <= y < img["height"]
            assert ann["iscrowd"] == 0
            for s, e in ann["tokens_positive"]:
                # matches how GroundingDataset parses caption[t0:t1]
                assert img["caption"][s:e].strip() == "cat"
        # exactly one caption per image, with no duplicated class names
        for im in payload["images"]:
            assert im["caption"].count("cat") == 1

    def test_convert_grounding_bucket_filter(self, tmp_path):
        """bucket filtering: excluding the medium bucket must reduce (or zero out) the box count."""
        raw = self._annotate(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        run_clean(raw=raw, out=cleaned, iou_thr=0.5, min_votes=2)
        stats_all = run_convert(cleaned=cleaned, out=tmp_path / "a", fmt="grounding")
        stats_high = run_convert(cleaned=cleaned, out=tmp_path / "h", fmt="grounding", buckets={"high"})
        assert stats_all["boxes"] >= stats_high["boxes"] > 0

    def test_convert_unknown_format_rejected(self, tmp_path):
        raw = self._annotate(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        run_clean(raw=raw, out=cleaned, iou_thr=0.5, min_votes=2)
        with pytest.raises(ValueError, match="unknown output format"):
            run_convert(cleaned=cleaned, out=tmp_path / "x", fmt="coco")

    def test_smoke_end_to_end(self, tmp_path):
        stats = run_smoke(out=tmp_path / "smoke", num_images=3)
        assert stats["negatives"] == 3
        assert stats["clean"]["kept_boxes"] > 0


class TestConcepts:
    """Concept discovery: taggers, negative sampling, manifest generation."""

    def _make_images(self, tmp_path: Path, n: int = 3) -> Path:
        from PIL import Image

        img_dir = tmp_path / "pool"
        img_dir.mkdir()
        for i in range(n):
            Image.new("RGB", (80, 60), (i * 40, 100, 150)).save(img_dir / f"img{i}.jpg")
        return img_dir

    def test_mock_tagger_deterministic(self, tmp_path):
        img_dir = self._make_images(tmp_path)
        t1, t2 = MockTagger(seed=0), MockTagger(seed=0)
        img = sorted(img_dir.glob("*.jpg"))[0]
        assert t1.tag(img) == t2.tag(img)
        assert 2 <= len(t1.tag(img)) <= 4

    def test_file_tagger(self, tmp_path):
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("Person\nCAR\nperson\n\n dog \n", encoding="utf-8")
        tagger = FileTagger(vocab)
        assert tagger.tag(tmp_path / "x.jpg") == ["person", "car", "dog"]

    def test_build_tagger_unknown(self):
        with pytest.raises(ValueError, match="unknown tagger spec"):
            build_tagger("clip:xxx")

    def test_sample_negatives_excludes_positives(self):
        rng = random.Random(0)
        negs = sample_negatives(["person", "car"], ["person", "car", "dog", "bird"], 5, rng)
        assert set(negs) <= {"dog", "bird"}
        assert len(negs) == 2
        assert sample_negatives(["person"], ["person"], 3, rng) == []

    def test_run_concepts_manifest(self, tmp_path):
        img_dir = self._make_images(tmp_path)
        vocab = tmp_path / "vocab.txt"
        vocab.write_text("person\ncar\ndog\nbicycle\n", encoding="utf-8")
        out = tmp_path / "manifest.jsonl"
        stats = run_concepts(
            images=img_dir,
            tagger_spec="mock:0",
            out=out,
            global_vocab=vocab,
            neg_per_image=1,
            max_queries=32,
            seed=0,
        )
        assert stats["images"] == 3
        assert stats["negatives"] == 3
        with out.open("r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
        assert len(lines) == 3
        for rec in lines:
            assert Path(rec["image"]).is_absolute()
            assert 3 <= len(rec["queries"]) <= 5  # 2-4 positives + 1 negative

    def test_concepts_to_annotate_chain(self, tmp_path):
        """Manifests produced by concepts are directly consumable by annotate."""
        img_dir = self._make_images(tmp_path, n=2)
        manifest = tmp_path / "manifest.jsonl"
        run_concepts(images=img_dir, tagger_spec="mock:0", out=manifest)
        raw = tmp_path / "raw.jsonl"
        stats = run_annotate(manifest=manifest, teacher_specs=["mock:0", "mock:1"], out=raw)
        assert stats["failed"] == 0
        assert stats["records"] > 0


class TestDrive20k:
    """20K campaign driver: next-assignment selection."""

    def _make_pool(self, tmp_path: Path) -> Path:
        pool = tmp_path / "pool"
        pool.mkdir()
        for s in (1, 2):
            with (pool / f"manifest_s{s:02d}.jsonl").open("w", encoding="utf-8") as f:
                f.write(json.dumps({"image": f"/tmp/img{s}a.jpg", "queries": ["cat", "dog"]}) + "\n")
                f.write(json.dumps({"image": f"/tmp/img{s}b.jpg", "queries": ["car"]}) + "\n")
        return pool

    def test_pick_first_unfinished(self, tmp_path):
        from scripts.ov_distill.drive_20k import pick_next

        pool = self._make_pool(tmp_path)
        nxt = pick_next(pool)
        assert nxt["slice"] == 1 and nxt["teacher"] == "wevisone"
        assert nxt["done"] == 0 and nxt["total"] == 3

    def test_done_keys_advance_rotation(self, tmp_path):
        from scripts.ov_distill.drive_20k import TEACHER_NAMES, pick_next

        pool = self._make_pool(tmp_path)
        # wevisone complete on slice 1 (3 keys), rexomni half done (1 of 3)
        rows = []
        for img, q in [("/tmp/img1a.jpg", "cat"), ("/tmp/img1a.jpg", "dog"), ("/tmp/img1b.jpg", "car")]:
            rows.append({"image": img, "teacher": TEACHER_NAMES["wevisone"], "query": q, "boxes": [], "ok": True})
        rows.append({"image": "/tmp/img1a.jpg", "teacher": TEACHER_NAMES["rexomni"], "query": "cat", "boxes": [], "ok": True})
        with (pool / "raw.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        nxt = pick_next(pool)
        assert nxt["slice"] == 1 and nxt["teacher"] == "rexomni"
        assert nxt["done"] == 1 and nxt["total"] == 3

    def test_all_done(self, tmp_path):
        from scripts.ov_distill.drive_20k import TEACHER_NAMES, TEACHER_ORDER, pick_next

        pool = self._make_pool(tmp_path)
        with (pool / "raw.jsonl").open("w", encoding="utf-8") as f:
            for s in (1, 2):
                for teacher in TEACHER_ORDER:
                    for img, qs in ((f"/tmp/img{s}a.jpg", ["cat", "dog"]), (f"/tmp/img{s}b.jpg", ["car"])):
                        for q in qs:
                            f.write(
                                json.dumps({"image": img, "teacher": TEACHER_NAMES[teacher], "query": q, "boxes": [], "ok": True})
                                + "\n"
                            )
        assert pick_next(pool)["teacher"] is None


from scripts.ov_distill.make_coco128_manifest import COCO_NAMES  # noqa: E402


class TestClipVerify:
    """CLIP re-verification: cache keys, threshold calibration, end-to-end filtering (fake encoder)."""

    def _setup_records(self, tmp_path: Path):
        from PIL import Image

        img = tmp_path / "a.jpg"
        Image.new("RGB", (100, 100), (30, 60, 90)).save(img)
        recs = [
            {"image": str(img), "image_path": str(img), "query": "cat", "box": [0, 0, 50, 50], "bucket": "medium"},
            {"image": str(img), "image_path": str(img), "query": "dog", "box": [50, 50, 99, 99], "bucket": "medium"},
            {"image": str(img), "image_path": str(img), "query": "cat", "box": [10, 10, 60, 60], "bucket": "high"},
        ]
        return img, recs

    def test_cache_key_tolerates_float_jitter(self):
        from scripts.ov_distill.clip_verify import _cache_key

        r1 = {"image": "a.jpg", "query": "cat", "box": [1.0, 2.0, 3.0, 4.0]}
        r2 = {"image": "a.jpg", "query": "cat", "box": [1.04, 2.04, 3.04, 4.04]}
        r3 = {"image": "a.jpg", "query": "cat", "box": [1.2, 2.0, 3.0, 4.0]}
        assert _cache_key(r1) == _cache_key(r2)
        assert _cache_key(r1) != _cache_key(r3)

    def test_calibrate_threshold_quantile(self, tmp_path):
        from scripts.ov_distill.clip_verify import _cache_key, calibrate_threshold

        img, recs = self._setup_records(tmp_path)
        # GT: cat has one box exactly overlapping recs[0]
        gt_dir = tmp_path / "gt"
        gt_dir.mkdir()
        cat_id = COCO_NAMES.index("cat")
        (gt_dir / "a.txt").write_text(f"{cat_id} 0.25 0.25 0.5 0.5\n")
        scores = {_cache_key(recs[0]): 80.0, _cache_key(recs[1]): 20.0}
        thr, stats = calibrate_threshold(recs[:2], scores, gt_dir, keep=0.95)
        assert stats["n_true"] == 1 and stats["n_false"] == 1
        assert thr == pytest.approx(80.0)  # a single true box: the quantile is its own score
        assert stats["false_score_mean"] == 20.0

    def test_run_clip_verify_filters_and_caches(self, tmp_path, monkeypatch):
        import numpy as np

        import scripts.ov_distill.clip_verify as cv

        img, recs = self._setup_records(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        with cleaned.open("w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        class FakeEncoder:
            calls = 0

            def __init__(self, model_dir):
                pass

            def encode_texts(self, texts):
                return np.array([[1.0, 0.0, 0.0, 0.0]] * len(texts))

            def encode_images(self, crops):
                # the first crop scores high, the rest score low
                FakeEncoder.calls += 1
                return np.array([[1.0, 0, 0, 0]] + [[0.0, 1, 0, 0]] * (len(crops) - 1))

        monkeypatch.setattr(cv, "WClipEncoder", FakeEncoder)
        out = tmp_path / "verified.jsonl"
        stats = cv.run_clip_verify(cleaned=cleaned, out=out, threshold=50.0)
        # two medium boxes: the first (high score) is kept, the second (low score) dropped;
        # the high bucket is out of scope and passed through untouched
        assert stats["kept"] == 1
        assert stats["dropped_low_score"] == 1
        assert stats["out_of_scope"] == 1
        rows = [json.loads(line) for line in out.open()]
        assert len(rows) == 2
        kept_medium = [r for r in rows if r.get("bucket") == "medium"]
        assert kept_medium[0]["clip_score"] == pytest.approx(100.0)

        # rerun: cache hit, encoder never called again
        calls_before = FakeEncoder.calls
        stats2 = cv.run_clip_verify(cleaned=cleaned, out=out, threshold=50.0)
        assert stats2["kept"] == 1
        assert FakeEncoder.calls == calls_before

    def test_threshold_required_without_calibration(self, tmp_path, monkeypatch):
        import scripts.ov_distill.clip_verify as cv

        import numpy as np

        class FakeEncoder:
            def __init__(self, model_dir):
                pass

            def encode_texts(self, texts):
                return np.ones((len(texts), 4))

            def encode_images(self, crops):
                return np.ones((len(crops), 4))

        monkeypatch.setattr(cv, "WClipEncoder", FakeEncoder)
        img, recs = self._setup_records(tmp_path)
        cleaned = tmp_path / "cleaned.jsonl"
        cleaned.write_text(json.dumps(recs[0]) + "\n")
        with pytest.raises(ValueError, match="calibrat"):
            cv.run_clip_verify(cleaned=cleaned, out=tmp_path / "o.jsonl", threshold=None)


class TestOvHoldout:
    """seen/unseen class split and unseen validation-set construction."""

    def test_split_deterministic_and_disjoint(self):
        from scripts.ov_distill.eval_ov_holdout import split_classes

        seen1, unseen1 = split_classes(0.25, seed=0)
        seen2, unseen2 = split_classes(0.25, seed=0)
        assert unseen1 == unseen2  # deterministic
        assert len(unseen1) == 20 and len(seen1) == 60
        assert not set(seen1) & set(unseen1)
        assert set(seen1) | set(unseen1) == set(COCO_NAMES)
        _, unseen3 = split_classes(0.25, seed=1)
        assert unseen3 != unseen1  # different seed -> different split

    def test_build_unseen_val_remaps_ids(self, tmp_path):
        from scripts.ov_distill.eval_ov_holdout import build_unseen_val, split_classes

        from PIL import Image

        _, unseen = split_classes(0.25, seed=0)
        labels = tmp_path / "labels"
        images = tmp_path / "images"
        labels.mkdir()
        images.mkdir()
        Image.new("RGB", (64, 64)).save(images / "im1.jpg")
        # im1: one unseen-class box + one seen-class box
        unseen_id = COCO_NAMES.index(unseen[0])
        seen_names = [n for n in COCO_NAMES if n not in set(unseen)]
        seen_id = COCO_NAMES.index(seen_names[0])
        (labels / "im1.txt").write_text(f"{unseen_id} 0.5 0.5 0.2 0.2\n{seen_id} 0.1 0.1 0.1 0.1\n")
        # im2: seen classes only; must be excluded
        (labels / "im2.txt").write_text(f"{seen_id} 0.5 0.5 0.2 0.2\n")

        out = tmp_path / "holdout"
        yaml_path = build_unseen_val(labels, images, out, unseen)
        new_lab = (out / "val" / "labels" / "im1.txt").read_text().strip()
        assert new_lab == "0 0.5 0.5 0.2 0.2"  # unseen[0] remapped to 0; the seen box removed
        assert not (out / "val" / "labels" / "im2.txt").exists()
        text = yaml_path.read_text()
        assert f"  0: {unseen[0]}" in text
        assert (out / "val" / "images" / "im1.jpg").is_symlink()
