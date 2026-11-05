"""Fit the adaptive budget on COCO-Stuff and benchmark frozen SED on ADE20K-150."""

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
from time import perf_counter
from zipfile import ZipFile

from filelock import FileLock
import numpy as np
import torch

from .datasets import ade150, coco_stuff, sample_labeled
from .metrics import class_iou, confusion_matrix, mean_iou, pixel_counts
from .model import SedInference, build_sed
from .policy import (
    BUDGETS, BUDGET_NAMES, FEATURE_NAMES, BudgetHead, fit_head, heuristic_budget,
    heuristic_from_features, oracle_index,
)


BUDGET_NAME = dict(zip(BUDGETS, BUDGET_NAMES))
MODES = (*BUDGET_NAMES, "heuristic", "learned")


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def write_json(path, data):
    pending = path.with_suffix(".partial")
    pending.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def resume_rows(path, image_ids, calculate, label):
    """Resume a flushed JSONL log and only rerun its uncommitted final row."""
    rows = []
    if path.exists():
        with path.open("rb+") as existing:
            for line in existing:
                if not line.endswith(b"\n"):
                    existing.truncate(existing.tell() - len(line))
                    break
                rows.append(json.loads(line))
        if [row["id"] for row in rows] != image_ids[:len(rows)]:
            raise ValueError(f"Saved {label} rows differ from the run's split manifest.")
    with path.open("a", encoding="utf-8", newline="\n") as output:
        for index, image_id in enumerate(image_ids[len(rows):], len(rows) + 1):
            start = perf_counter()
            row = {"id": image_id, **calculate(image_id)}
            row["wall_seconds"] = perf_counter() - start
            output.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
            rows.append(row)
            if index % 25 == 0 or index == len(image_ids):
                print(f"{label} {index}/{len(image_ids)}", flush=True)
    return rows


def collect_one(engine, dataset, image_id, tolerance):
    item, target = dataset.load(image_id)
    features, masks = engine.sweep(item)
    scores = [mean_iou(pixel_counts(confusion_matrix(target, mask, dataset.num_classes)))
              for mask, _ in masks.values()]
    return {"features": features.tolist(), "iou_by_budget": scores, "oracle": oracle_index(scores, tolerance)}


def evaluate_one(engine, dataset, image_id, controller, tolerance):
    item, target = dataset.load(image_id)
    features, masks = engine.sweep(item)
    ground_truth = np.unique(target[target != 255])
    modes = {}
    for name, (prediction, kept) in masks.items():
        counts = pixel_counts(confusion_matrix(target, prediction, dataset.num_classes))
        modes[name] = {
            "predicted": counts[1].tolist(), "correct": counts[2].tolist(),
            "image_iou": mean_iou(counts), "retained_classes": int(kept.sum()),
            "gt_classes_retained": int(kept[ground_truth].sum()),
        }
    scores = [modes[name]["image_iou"] for name in BUDGET_NAMES]
    return {
        "target": counts[0].tolist(),
        "oracle": oracle_index(scores, tolerance) if ground_truth.size else None,
        "choices": {"heuristic": BUDGET_NAME[heuristic_from_features(features)],
                    "learned": BUDGET_NAME[controller.budget_from_features(features)]},
        "fixed": modes,
    }


def time_one(engine, item, controller, order):
    policies = {"heuristic": heuristic_budget, "learned": controller.budget}
    modes = dict(zip(BUDGET_NAMES, BUDGETS))
    result = {}
    for name in order:
        _, _, timing = engine.predict(item, modes.get(name), policies.get(name), timed=True)
        result[name] = timing
    return {"modes": result}


def verify_sweep(engine, item, controller=None):
    features, cached = engine.sweep(item)
    for name, budget in zip(BUDGET_NAMES, BUDGETS):
        direct, kept, _ = engine.predict(item, budget)
        if not np.array_equal(direct, cached[name][0]) or not np.array_equal(kept, cached[name][1]):
            raise RuntimeError(f"Shared-prefix {name} mask differs from standalone SED.")
    if controller is not None:
        for from_features, policy in (
            (heuristic_from_features, heuristic_budget),
            (controller.budget_from_features, controller.budget),
        ):
            name = BUDGET_NAME[from_features(features)]
            direct, kept, _ = engine.predict(item, policy=policy)
            if not np.array_equal(direct, cached[name][0]) or not np.array_equal(kept, cached[name][1]):
                raise RuntimeError(f"Policy mask differs from its cached {name} branch.")


def fit_or_load(args, fit_rows, dev_rows):
    path = args.output / "head.json"
    if path.exists():
        saved = load_json(path)
        controller = BudgetHead()
        controller.load_state_dict({k: torch.tensor(v) for k, v in saved["state"].items()})
        return controller.eval().to("cuda"), saved["fit"]

    def tensors(rows):
        return torch.tensor([row["features"] for row in rows]), torch.tensor([row["oracle"] for row in rows])

    start = perf_counter()
    controller, stats = fit_head(*tensors(fit_rows), *tensors(dev_rows), seed=args.seed)
    stats["seconds"] = perf_counter() - start
    most_frequent = Counter(row["oracle"] for row in fit_rows).most_common(1)[0][0]
    stats["majority_dev_accuracy"] = sum(row["oracle"] == most_frequent for row in dev_rows) / len(dev_rows)
    write_json(path, {"state": {k: v.tolist() for k, v in controller.state_dict().items()}, "fit": stats})
    print(f"Head fit in {stats['seconds']:.1f} s. Dev accuracy {stats['dev_accuracy']:.1%}.", flush=True)
    return controller.to("cuda"), stats


def distribution(rows):
    return dict(Counter(BUDGET_NAMES[r["oracle"]] for r in rows if r["oracle"] is not None))


def summarize(rows, timed, crowded_classes):
    total = {name: np.zeros((3, 150), dtype=np.int64) for name in MODES}
    crowded = {name: matrix.copy() for name, matrix in total.items()}
    selected = {name: Counter() for name in MODES}
    retained = {name: [] for name in MODES}
    retained_truth = Counter()
    target_class_total = crowded_n = 0
    for row in rows:
        target = np.asarray(row["target"])
        present = np.count_nonzero(target)
        is_crowded = present >= crowded_classes
        target_class_total += present
        crowded_n += is_crowded
        for name in MODES:
            choice = row["choices"].get(name, name)
            result = row["fixed"][choice]
            counts = np.stack((target, result["predicted"], result["correct"]))
            total[name] += counts
            if is_crowded:
                crowded[name] += counts
            selected[name][choice] += 1
            retained[name].append(result["retained_classes"])
            retained_truth[name] += result["gt_classes_retained"]
    results = {}
    for name in MODES:
        timings = [r["modes"][name] for r in timed]
        latency = [r["ms"] for r in timings]
        crowded_iou = mean_iou(crowded[name])
        results[name] = {
            "mIoU_percent": 100 * mean_iou(total[name]),
            "crowded_mIoU_percent": None if crowded_iou is None else 100 * crowded_iou,
            "gt_class_retention_percent": 100 * retained_truth[name] / target_class_total,
            "median_retained_classes": float(np.median(retained[name])),
            "k_counts": dict(selected[name]),
            "p50_ms": float(np.median(latency)), "p95_ms": float(np.percentile(latency, 95)),
            "peak_allocated_mib": max(r["peak_mib"] for r in timings),
        }
    return results, total, int(crowded_n)


def make_manifest(args):
    file_args = ("weights", "openclip_weights", "ade_zip", "coco_images", "coco_masks")
    project = Path(__file__).resolve().parents[2]
    return {
        "dataset": "ADE20K A-150 validation", "calibration": "COCO-Stuff val2017",
        "fit_n": args.fit, "dev_n": args.dev, "eval_n": args.eval_images, "timing_n": args.timing_images,
        "seed": args.seed, "tolerance": args.tolerance, "amp": not args.fp32, "crowded_classes": 10,
        "os": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
        "gpu_total_mib": torch.cuda.get_device_properties(0).total_memory / 1024**2,
        "sha256": {k: sha256(getattr(args, k)) for k in file_args},
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "sed_patch_sha256": sha256(project / "patches" / "adaptive_cer.patch"),
    }


def run(args):
    out = args.output
    manifest = make_manifest(args)
    manifest_path = out / "manifest.json"
    if manifest_path.exists() and load_json(manifest_path) != manifest:
        raise ValueError("This run directory has different settings, inputs, or code. Choose a new --output.")
    write_json(manifest_path, manifest)
    if (out / "summary.json").exists():
        print(f"Completed run at {out / 'summary.json'}", flush=True)
        return

    print(f"Building SED ConvNeXt-B on {manifest['gpu']}", flush=True)
    model, resize, model_config = build_sed(args)
    engine = SedInference(model, not args.fp32)
    (out / "sed_config.yaml").write_text(model_config, encoding="utf-8")
    with ZipFile(args.coco_images) as coco_images, ZipFile(args.coco_masks) as coco_masks, ZipFile(args.ade_zip) as ade_zip:
        coco = coco_stuff(coco_images, coco_masks, args.sed_root, resize)
        ade = ade150(ade_zip, resize)
        rng = random.Random(args.seed)
        splits_path = out / "splits.json"
        if splits_path.exists():
            ids = load_json(splits_path)
        else:
            source_ids = sample_labeled(coco, args.fit + args.dev + 8, rng)
            validation_ids = rng.sample(ade.ids, args.eval_images)
            ids = {"fit": source_ids[:args.fit], "dev": source_ids[args.fit:args.fit + args.dev],
                   "warmup": source_ids[-8:], "validation": validation_ids,
                   "timing": rng.sample(validation_ids, args.timing_images)}
            write_json(splits_path, ids)

        engine.set_vocabulary("coco")
        if not (out / "calibration.jsonl").exists():
            verify_sweep(engine, coco.load(ids["warmup"][0])[0])
            print("COCO cached and standalone masks match.", flush=True)
        calibration = resume_rows(out / "calibration.jsonl", ids["fit"] + ids["dev"],
                                  lambda image_id: collect_one(engine, coco, image_id, args.tolerance),
                                  "Calibration")
        fit_rows, dev_rows = calibration[:args.fit], calibration[args.fit:]
        controller, fit_stats = fit_or_load(args, fit_rows, dev_rows)

        engine.set_vocabulary("ade150")
        if not (out / "validation.jsonl").exists():
            for image_id in ids["validation"][:3]:
                verify_sweep(engine, ade.load(image_id)[0], controller)
            print("ADE cached and standalone masks match for all seven methods.", flush=True)
        rows = resume_rows(out / "validation.jsonl", ids["validation"],
                           lambda image_id: evaluate_one(engine, ade, image_id, controller, args.tolerance),
                           "ADE20K validation")

        # Time complete forwards on a fixed subset. No shared work is used in these measurements.
        print("Warming up all methods before independent latency measurements.", flush=True)
        engine.set_vocabulary("ade150")
        for image_id in ids["warmup"]:
            time_one(engine, coco.load(image_id)[0], controller, MODES)
        timed = resume_rows(out / "timing.jsonl", ids["timing"],
                            lambda image_id: time_one(engine, ade.load(image_id)[0], controller,
                                                      random.Random(f"{args.seed}:{image_id}").sample(MODES, len(MODES))),
                            "Timing")

    results, totals, crowded_n = summarize(rows, timed, manifest["crowded_classes"])
    names = load_json(args.sed_root / "datasets" / "ade150.json")
    with (out / "per_class_iou.csv").open("w", encoding="utf-8", newline="") as csv_file:
        csv_rows = csv.writer(csv_file)
        csv_rows.writerow(("id", "class", "ground_truth_pixels", *MODES))
        class_scores = {mode: class_iou(totals[mode]) * 100 for mode in MODES}
        for class_id, name in enumerate(names):
            csv_rows.writerow((class_id, name, totals["all"][0, class_id],
                               *(f"{class_scores[mode][class_id]:.6f}" for mode in MODES)))
    summary = {
        **manifest, "feature_names": FEATURE_NAMES, "fit": fit_stats,
        "validation_class_coverage": int(np.count_nonzero(totals["all"][0])),
        "crowded_n": crowded_n, "oracle_fit_counts": distribution(fit_rows),
        "oracle_dev_counts": distribution(dev_rows), "oracle_val_counts": distribution(rows),
        "phase_seconds": {"calibration": sum(r["wall_seconds"] for r in calibration),
                          "validation": sum(r["wall_seconds"] for r in rows),
                          "timing": sum(r["wall_seconds"] for r in timed)},
        "results": results,
    }
    write_json(out / "summary.json", summary)
    print(f"Completed run at {out / 'summary.json'}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("sed-root", "weights", "openclip-weights", "ade-zip", "coco-images", "coco-masks"):
        parser.add_argument("--" + name, type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--output", type=lambda p: Path(p).resolve(), default=Path("runs/ade150"))
    parser.add_argument("--fit", type=int, default=640)
    parser.add_argument("--dev", type=int, default=160)
    parser.add_argument("--eval-images", type=int, default=2000)
    parser.add_argument("--timing-images", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--fp32", action="store_true")
    args = parser.parse_args()
    if min(args.fit, args.dev, args.timing_images) < 1 or not args.timing_images <= args.eval_images <= 2000:
        parser.error("Positive split sizes are required. Timing images must fit within the evaluation split.")
    if not 0 <= args.tolerance <= 1:
        parser.error("The oracle tolerance must be between 0 and 1.")
    return args


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This benchmark needs CUDA PyTorch and an NVIDIA GPU.")
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    args.output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.output / ".run.lock"), timeout=0):
        run(args)


if __name__ == "__main__":
    main()

