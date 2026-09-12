"""Gated, serial fixed-budget Frozen/Scratch comparison on externally supplied data."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from scripts.d1.artifacts import digest, encoded, file_sha, immutable, write_json
from scripts.d1.runtime import TRAINING_PRECISION, VALIDATION_PRECISION
from ultralytics.nn.foundation.npy_cache import open_feature_cache
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / "ultralytics/cfg/experiments/d1/paired-comparison.yaml"
RECIPE = ROOT / "ultralytics/cfg/experiments/d1/cached-detection.yaml"


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def resource_snapshot():
    """Record visible contention; never kill foreign work or force memory pressure."""
    memory = Path("/sys/fs/cgroup/memory")
    if (memory / "memory.usage_in_bytes").is_file():
        used = int((memory / "memory.usage_in_bytes").read_text())
        limit = int((memory / "memory.limit_in_bytes").read_text())
    else:
        used = int(Path("/sys/fs/cgroup/memory.current").read_text())
        limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True,
    )
    processes = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            arguments = (path / "cmdline").read_bytes().decode(errors="replace").lower().split("\0")
            script = next((value for value in arguments[1:3] if value.endswith(".py")), "")
            working_dir = str((path / "cwd").resolve()).lower()
            workload = any(
                tag in arguments[0] or tag in script or (script and tag in working_dir)
                for tag in ("finssim", "fins_sim")
            )
            name = (path / "comm").read_text().strip()
            if workload and name not in {
                "git",
                "git-remote-http",
                "bash",
                "sshd",
            }:
                processes.append({"pid": int(path.name), "name": name})
        except OSError:
            continue
    return {"memory_bytes": used, "memory_limit_bytes": limit, "gpu": gpu, "fins_processes": processes}


def inspect_paths(args, contract):
    data = {}
    for dataset, spec in contract["datasets"].items():
        root = getattr(args, dataset + "_root").resolve(strict=True)
        cache = getattr(args, dataset + "_cache").resolve(strict=True)
        mount = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE,FSTYPE", "-T", str(root)], text=True)
        cache_mount = subprocess.check_output(["findmnt", "-n", "-o", "SOURCE,FSTYPE", "-T", str(cache)], text=True)
        if not mount.startswith("/dev/nvme") or not cache_mount.startswith("/dev/nvme"):
            raise ValueError("Both RGB and features must be on NVMe")
        names = (
            YAML.load(ROOT / "ultralytics/cfg/datasets/coco.yaml")["names"]
            if dataset == "coco"
            else YAML.load(root / "dataset.yaml")["names"]
        )
        item = {"root": str(root), "cache_root": str(cache), "names": names, "splits": {}}
        all_ids = set()
        for split in ("train", "val"):
            key = split + "2017" if dataset == "coco" else "visdrone-" + split
            reader = open_feature_cache(cache / key)
            records = sorted(reader.records.values(), key=lambda row: row["sample_id"])
            paths_digest = digest(("\n".join(row["image_path"] for row in records) + "\n").encode())
            if paths_digest != spec[split + "_paths_sha256"]:
                raise ValueError("Cache does not match the approved official split")
            if reader.index["contract_sha256"] != contract["cache_contract_sha256"]:
                raise ValueError("Cache preprocessing contract is not the approved contract")
            if reader.contract["teacher_weights_sha256"] != contract["teacher_weights_sha256"]:
                raise ValueError("Cache teacher differs from the approved ViT-S weights")
            ids = {row["sample_id"].split("/")[-1] for row in records}
            if len(records) != spec[split + "_images"] or len(ids) != len(records) or ids & all_ids:
                raise ValueError("Incomplete or overlapping official split")
            all_ids.update(ids)
            paths = [root / row["image_path"] for row in records]
            if any(not p.is_file() for p in paths):
                raise FileNotFoundError("RGB paths do not cover the feature cache")
            for index in sorted({0, len(paths) // 2, len(paths) - 1}):
                if file_sha(paths[index]) != records[index]["image_sha256"]:
                    raise ValueError("RGB content differs from cached source")
                reader.verify_sample(records[index]["sample_id"])
            item["splits"][split] = {
                "paths": [str(p) for p in paths],
                "cache": str(cache / key),
                "contract": reader.contract,
                "index_sha256": file_sha(reader.index_path),
                "index_path": str(reader.index_path),
                "paths_sha256": paths_digest,
                "labels_sha256": labels_digest(root, paths),
                "count": len(paths),
            }
            reader.close()
        if item["splits"]["train"]["contract"] != item["splits"]["val"]["contract"]:
            raise ValueError("Feature preprocessing/teacher contracts differ")
        item["annotations"] = {}
        if dataset == "coco":
            for split in ("train2017", "val2017"):
                path = root / "annotations" / ("instances_" + split + ".json")
                item["annotations"][str(path)] = file_sha(path)
        else:
            for name in ("manifest.json", "samples.jsonl"):
                path = root / name
                item["annotations"][str(path)] = file_sha(path)
        data[dataset] = item
    return data


def labels_digest(root, paths):
    value = hashlib.sha256()
    for image in paths:
        relative = Path(image).relative_to(root).relative_to("images").with_suffix(".txt")
        label = Path(root) / "labels" / relative
        value.update((relative.as_posix() + ":" + (file_sha(label) if label.is_file() else "missing") + "\n").encode())
    return value.hexdigest()


def verify_inputs(plan, *, labels=False):
    if file_sha(RECIPE) != plan["recipe_sha256"] or file_sha(CONTRACT) != plan["contract_sha256"]:
        raise RuntimeError("Experiment configuration changed")
    for dataset in plan["data"].values():
        for split in dataset["splits"].values():
            if file_sha(split["index_path"]) != split["index_sha256"]:
                raise RuntimeError("Feature cache index changed")
            if labels and labels_digest(Path(dataset["root"]), split["paths"]) != split["labels_sha256"]:
                raise RuntimeError("Detection labels changed after preparation")
        if labels:
            for path, expected in dataset["annotations"].items():
                if file_sha(Path(path)) != expected:
                    raise RuntimeError("Official annotations or prepared data manifest changed")


def write_data(output, dataset, data, *, subset=False, batch=1):
    folder = output / "inputs" / (dataset + ("-gate" if subset else "-full"))
    names = {}
    for split in ("train", "val"):
        paths = data["splits"][split]["paths"]
        if subset:
            paths = paths[: 2 * batch if split == "train" else 24]
        path = folder / (split + ".txt")
        immutable(path, ("\n".join(paths) + "\n").encode())
        names[split] = str(path)
    config = folder / "dataset.yaml"
    immutable(config, encoded({"path": data["root"], **names, "names": data["names"]}))
    return config


def training_command(plan, dataset, variant, output, data_yaml, *, window=None, resume=False):
    spec = plan["contract"]["datasets"][dataset]
    data = plan["data"][dataset]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc_per_node=6",
        "-m",
        "scripts.d1.train",
        "train",
        "--variant",
        variant,
        "--dataset",
        dataset,
        "--data",
        str(data_yaml),
        "--output",
        str(output),
        "--device",
        plan["device"],
        "--batch",
        str(spec["global_batch"]),
        "--epochs",
        str(spec["epochs"]),
        "--workers",
        "4",
        "--seed",
        "0",
        "--recipe",
        str(RECIPE),
        "--telemetry",
        "--approved",
    ]
    if variant != "SCRATCH":
        cmd += [
            "--train-cache",
            data["splits"]["train"]["cache"],
            "--val-cache",
            data["splits"]["val"]["cache"],
            "--trusted-cache",
        ]
    if window is not None:
        cmd += ["--window", str(window)]
    if resume:
        cmd += ["--resume-snapshot", str(output / "resume.pt")]
    return cmd


def run_job(root, plan, name, command):
    if git("rev-parse", "HEAD") != plan["code_commit"] or git("status", "--porcelain"):
        raise RuntimeError("Source changed after the queue contract was locked")
    verify_inputs(plan, labels="-seed" in name and "-evaluate-" not in name)
    selected = {int(index) for index in plan["device"].split(",")}
    for attempt in range(10):
        resources = resource_snapshot()
        rows = [tuple(int(value.strip()) for value in row.split(",")) for row in resources["gpu"].splitlines()]
        busy = any(index in selected and (memory > 1024 or utilization > 25) for index, memory, utilization in rows)
        if not busy:
            break
        time.sleep(2)
    if busy:
        raise RuntimeError("GPU contention detected before launch; no foreign process was stopped")
    if resources["fins_processes"]:
        raise RuntimeError("FinsSim contention detected; queue stopped without killing it")
    if resources["memory_bytes"] > resources["memory_limit_bytes"] * plan["contract"]["maximum_memory_fraction"]:
        # File-backed pages are reclaimable; only enforce against the anonymous working set.
        stat_path = Path("/sys/fs/cgroup/memory/memory.stat")
        stats = dict(line.split() for line in stat_path.read_text().splitlines()) if stat_path.exists() else {}
        if int(stats.get("total_rss", resources["memory_bytes"])) > resources["memory_limit_bytes"] * 0.6:
            raise RuntimeError("Insufficient container working-set headroom")
    import shutil

    if shutil.disk_usage(root).free < plan["contract"]["minimum_free_gib"] * 2**30:
        raise RuntimeError("Run filesystem is below the contracted free-space reserve")
    log_path = root / "logs" / (name + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    env = os.environ.copy()
    env.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONUNBUFFERED="1",
        YOLO_AUTOINSTALL="false",
    )
    with log_path.open("xb") as log:
        child = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        record = {
            "status": "running",
            "name": name,
            "pid": child.pid,
            "started_unix": started,
            "command": command,
            "resources_before": resources,
            "log": str(log_path),
        }
        try:
            while child.poll() is None:
                write_json(
                    root / "status.json",
                    {**record, "updated_unix": time.time(), "elapsed_seconds": time.time() - started},
                )
                time.sleep(10)
        except BaseException:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=60)
            raise
    record.update(
        status="completed" if child.returncode == 0 else "failed",
        returncode=child.returncode,
        elapsed_seconds=time.time() - started,
        finished_unix=time.time(),
    )
    write_json(root / "jobs" / (name + ".json"), record)
    write_json(root / "status.json", record)
    if child.returncode:
        raise RuntimeError(f"{name} failed; see {log_path}")
    return record


def evaluation_command(plan, dataset, variant, checkpoint, output, data_yaml):
    command = [
        sys.executable,
        "-m",
        "scripts.d1.train",
        "evaluate",
        "--variant",
        variant,
        "--dataset",
        dataset,
        "--data",
        str(data_yaml),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--device",
        plan["device"].split(",")[0],
        "--batch",
        "16",
        "--workers",
        "4",
    ]
    if variant != "SCRATCH":
        command += ["--val-cache", plan["data"][dataset]["splits"]["val"]["cache"], "--trusted-cache"]
    if dataset == "coco":
        command += ["--annotations", str(Path(plan["data"][dataset]["root"]) / "annotations/instances_val2017.json")]
    return command


def evaluate_run(root, plan, dataset, variant, name, output, data_yaml):
    checkpoints = sorted((output / "weights").glob("epoch-*.pt"))
    expected = plan["contract"]["datasets"][dataset]["epochs"] // plan["contract"]["evaluation_period"]
    if len(checkpoints) != expected:
        raise RuntimeError("Periodic checkpoint coverage is incomplete")
    for checkpoint in [*checkpoints, output / "weights/last.pt", output / "weights/best.pt"]:
        target = output / "evaluations" / checkpoint.stem
        run_job(
            root,
            plan,
            name + "-evaluate-" + checkpoint.stem,
            evaluation_command(plan, dataset, variant, checkpoint, target, data_yaml),
        )
        report = json.loads((target / "evaluation.json").read_text())
        if report["images"] != plan["contract"]["datasets"][dataset]["val_images"] or not report["strict_reload"]:
            raise RuntimeError("Independent evaluation coverage/strict reload failed")
    reports = []
    for checkpoint in checkpoints:
        target = output / "evaluations" / checkpoint.stem
        report = json.loads((target / "evaluation.json").read_text())
        if dataset == "coco":
            reports.append(
                {
                    "epoch": report["checkpoint_epoch_zero_based"] + 1,
                    "AP": report["official"]["metrics"]["AP"],
                    "checkpoint": str(checkpoint),
                    "sha256": file_sha(checkpoint),
                }
            )
    if reports:
        best = min(reports, key=lambda item: (-item["AP"], item["epoch"]))
        write_json(output / "standard-best.json", best)
    else:
        write_json(
            output / "official-scoring-pending.json",
            {
                "status": "matlab_scoring_pending",
                "periodic_checkpoints": len(checkpoints),
                "last_and_internal_best_exported": True,
                "tool": "official-VisDrone-MATLAB-DET",
            },
        )


def compare_states(first, second):
    """Require exact state equivalence; identities and wall-clock metadata may differ."""
    import numpy as np

    differences = []

    def walk(a, b, path):
        if isinstance(a, torch.Tensor):
            ok = isinstance(b, torch.Tensor) and a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            ok = isinstance(b, np.ndarray) and np.array_equal(a, b)
        elif isinstance(a, dict):
            ok = isinstance(b, dict) and a.keys() == b.keys()
            if ok:
                for key in a:
                    walk(a[key], b[key], path + "/" + str(key))
                return
        elif isinstance(a, (tuple, list)):
            ok = isinstance(b, type(a)) and len(a) == len(b)
            if ok:
                for i, (x, y) in enumerate(zip(a, b)):
                    walk(x, y, path + "/" + str(i))
                return
        else:
            ok = a == b
        if not ok:
            differences.append(path)

    for key in (
        "model",
        "ema",
        "optimizer",
        "scaler",
        "scheduler",
        "criterion",
        "optimizer_steps",
        "ema_updates",
        "ranks",
    ):
        if key not in first or key not in second:
            differences.append(key + "/missing")
        else:
            walk(first[key], second[key], key)
    return differences


def benchmark_eta(root, plan, times, evaluation_times):
    seconds = 0.0
    for dataset, spec in plan["contract"]["datasets"].items():
        for variant in ("BN64", "SCRATCH"):
            values = times.get(dataset + "-" + variant)
            if values is None:
                return None
            seconds += 3 * spec["epochs"] * statistics.median(values[-2:])
    evaluation_seconds = sum(
        3 * (spec["epochs"] // plan["contract"]["evaluation_period"] + 2) * evaluation_times[dataset + "-" + variant]
        for dataset, spec in plan["contract"]["datasets"].items()
        for variant in ("BN64", "SCRATCH")
    )
    result = {
        "status": "estimated",
        "train_val_hours": seconds / 3600,
        "train_val_gpu_hours": seconds * 6 / 3600,
        "seconds_per_epoch": times,
        "independent_prediction_hours": evaluation_seconds / 3600,
        "train_and_prediction_hours": (seconds + evaluation_seconds) / 3600,
        "excludes": ["visdrone_official_matlab_scoring_and_transfer", "one_time_teacher_extraction"],
        "all_12_runs_train_val_range_hours": [seconds / 3600 * 0.9, seconds / 3600 * 1.25],
        "note": "Full-epoch measured extrapolation, not a completion guarantee.",
    }
    write_json(root / "eta.json", result)
    return result


def run_queue(args):
    if not args.approved:
        raise ValueError("Explicit --approved is required")
    root = args.output.resolve()
    if root.is_relative_to(ROOT):
        raise ValueError("Artifacts must remain outside the source tree")
    root.mkdir(parents=True, exist_ok=False)
    with (root / "controller.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if git("status", "--porcelain"):
                raise RuntimeError("Commit the tested code before starting")
            contract = YAML.load(CONTRACT)
            if contract.get("validation_precision") != VALIDATION_PRECISION:
                raise ValueError("Comparison validation precision differs from the runtime policy")
            if contract.get("training_precision") != TRAINING_PRECISION:
                raise ValueError("Comparison training precision differs from the runtime policy")
            devices = [int(value) for value in args.device.split(",")]
            available = {int(row.split(",")[0]) for row in resource_snapshot()["gpu"].splitlines()}
            if (
                len(devices) != contract["world_size"]
                or len(set(devices)) != len(devices)
                or not set(devices) <= available
            ):
                raise ValueError("The paired contract requires six GPUs")
            plan = {
                "code_commit": git("rev-parse", "HEAD"),
                "contract": contract,
                "contract_sha256": file_sha(CONTRACT),
                "recipe_sha256": file_sha(RECIPE),
                "device": args.device,
                "data": inspect_paths(args, contract),
            }
            write_json(root / "plan.json", plan)
            full_data = {d: write_data(root, d, v) for d, v in plan["data"].items()}
            times = {}
            evaluation_times = {}
            for dataset in ("visdrone", "coco"):
                batch = contract["datasets"][dataset]["global_batch"]
                gate_data = write_data(root, dataset, plan["data"][dataset], subset=True, batch=batch)
                for variant in ("BN64", "SCRATCH"):
                    key = dataset + "-" + variant
                    continuous = root / "gates" / (key + "-continuous")
                    resumed = root / "gates" / (key + "-resumed")
                    run_job(
                        root,
                        plan,
                        key + "-gate-continuous",
                        training_command(plan, dataset, variant, continuous, gate_data, window=2),
                    )
                    run_job(
                        root,
                        plan,
                        key + "-gate-initial",
                        training_command(plan, dataset, variant, resumed, gate_data, window=1),
                    )
                    run_job(
                        root,
                        plan,
                        key + "-gate-resume",
                        training_command(plan, dataset, variant, resumed, gate_data, window=2, resume=True),
                    )
                    a = torch.load(continuous / "resume.pt", map_location="cpu", weights_only=False)
                    b = torch.load(resumed / "resume.pt", map_location="cpu", weights_only=False)
                    differences = compare_states(a, b)
                    write_json(
                        root / "gates" / (key + "-equivalence.json"),
                        {"status": "passed" if not differences else "failed", "differences": differences[:100]},
                    )
                    del a, b
                    if differences:
                        raise RuntimeError(f"{key} uninterrupted/resume states differ: {differences[:5]}")
                    benchmark = root / "benchmarks" / key
                    run_job(
                        root,
                        plan,
                        key + "-benchmark",
                        training_command(
                            plan, dataset, variant, benchmark, full_data[dataset], window=contract["benchmark_epochs"]
                        ),
                    )
                    rows = [
                        json.loads(path.read_text()) for path in sorted((benchmark / "validation").glob("epoch-*.json"))
                    ]
                    if len(rows) != contract["benchmark_epochs"]:
                        raise RuntimeError("Benchmark did not produce the contracted complete epoch timings")
                    times[key] = [row["epoch_wall_seconds"] for row in rows]
                    evaluated = root / "benchmarks" / (key + "-evaluation")
                    evaluation_job = run_job(
                        root,
                        plan,
                        key + "-benchmark-evaluate",
                        evaluation_command(
                            plan, dataset, variant, benchmark / "weights/last.pt", evaluated, full_data[dataset]
                        ),
                    )
                    report = json.loads((evaluated / "evaluation.json").read_text())
                    if report["images"] != contract["datasets"][dataset]["val_images"] or not report["strict_reload"]:
                        raise RuntimeError("Benchmark independent validation failed")
                    evaluation_times[key] = evaluation_job["elapsed_seconds"]
                    write_json(root / "benchmark-progress.json", times)
                    benchmark_eta(root, plan, times, evaluation_times)
            for seed in contract["seeds"]:
                for dataset in ("visdrone", "coco"):
                    variants = ("BN64", "SCRATCH") if seed % 2 == 0 else ("SCRATCH", "BN64")
                    for variant in variants:
                        name = f"{dataset}-{variant}-seed{seed}"
                        output = root / "runs" / name
                        command = training_command(plan, dataset, variant, output, full_data[dataset])
                        command[command.index("--seed") + 1] = str(seed)
                        run_job(root, plan, name, command)
                        evaluate_run(root, plan, dataset, variant, name, output, full_data[dataset])
            write_json(
                root / "status.json", {"status": "training_complete_evaluation_pending", "updated_unix": time.time()}
            )
        except BaseException as exc:
            previous = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {}
            write_json(
                root / "status.json", {**previous, "status": "failed", "error": str(exc), "updated_unix": time.time()}
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    for dataset in ("coco", "visdrone"):
        for kind in ("root", "cache"):
            parser.add_argument(f"--{dataset}-{kind}", type=Path, required=True)
    parser.add_argument("--device", default="0,1,2,3,4,5")
    parser.add_argument("--approved", action="store_true")
    run_queue(parser.parse_args())


if __name__ == "__main__":
    main()
