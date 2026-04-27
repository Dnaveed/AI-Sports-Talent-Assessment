"""Train simple per-exercise rep-count calibration from manual labels.

The base landmark counters are rule-based. This script learns small linear
corrections from manually counted videos so evaluation can better match human
labels.

Example:
  python tools/train_rep_calibration.py --dataset-dir "archive (3)"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_landmark_dataset import (  # noqa: E402
    DATASET_TO_APP_EXERCISE,
    analyze_sequence,
    iter_video_rows,
    load_labels,
    load_manual_validation,
    parse_optional_int,
)


def fit_linear(samples: List[Tuple[int, int]]) -> Dict[str, float]:
    if not samples:
        return {"slope": 1.0, "intercept": 0.0, "samples": 0}
    if len(samples) == 1:
        x, y = samples[0]
        return {"slope": 1.0, "intercept": float(y - x), "samples": 1}

    xs = [float(x) for x, _ in samples]
    ys = [float(y) for _, y in samples]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return {"slope": 0.0, "intercept": round(mean_y, 6), "samples": len(samples)}

    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    intercept = mean_y - slope * mean_x
    return {"slope": round(slope, 6), "intercept": round(intercept, 6), "samples": len(samples)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="archive (3)")
    parser.add_argument("--validation-labels", default="archive (3)/validation_rep_labels.csv")
    parser.add_argument("--out", default="archive (3)/rep_calibration.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels = load_labels(dataset_dir / "labels.csv")
    manual = load_manual_validation(Path(args.validation_labels))

    total_samples = defaultdict(list)
    valid_samples = defaultdict(list)
    sample_count = 0

    for vid_id, rows in iter_video_rows(dataset_dir / "landmarks.csv"):
        actual = manual.get(vid_id)
        if not actual:
            continue

        exercise = DATASET_TO_APP_EXERCISE.get(labels.get(vid_id, ""))
        if not exercise:
            continue

        actual_total = parse_optional_int(actual.get("actual_total_reps"))
        actual_valid = parse_optional_int(actual.get("actual_valid_reps"))
        if actual_total is None and actual_valid is None:
            continue

        result = analyze_sequence(rows, exercise)
        if actual_total is not None:
            total_samples[exercise].append((int(result["predicted_total_reps"]), actual_total))
        if actual_valid is not None:
            valid_samples[exercise].append((int(result["predicted_valid_reps"]), actual_valid))
        sample_count += 1

    calibration = {
        "type": "linear_rep_count_calibration",
        "source": str(Path(args.validation_labels)),
        "exercises": {},
    }
    for exercise in sorted(set(total_samples) | set(valid_samples)):
        calibration["exercises"][exercise] = {
            "total": fit_linear(total_samples[exercise]),
            "valid": fit_linear(valid_samples[exercise]),
        }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    print(f"Wrote calibration model: {out}")
    print(f"Trained from {sample_count} manually labeled videos")
    for exercise, model in calibration["exercises"].items():
        print(
            f"  {exercise}: total n={model['total']['samples']}, "
            f"valid n={model['valid']['samples']}"
        )


if __name__ == "__main__":
    main()
