"""Evaluate rep-counting analyzers on landmark time-series datasets.

This script is designed for the Kaggle exercise-recognition time-series export
that contains:
  - labels.csv with vid_id,class
  - landmarks.csv with one row per frame and x_/y_/z_ landmark columns

It skips unsupported classes such as pull_up, runs the existing project
analyzers on supported exercises, writes predicted counts, and optionally
compares those predictions with a small manually filled validation CSV.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DATASET_TO_APP_EXERCISE = {
    "jumping_jack": "jumping_jack",
    "push_up": "pushup",
    "situp": "situp",
    "squat": "squat",
}

ANGLE_RULES = {
    "pushup": {
        "triples": [("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist")],
        "down": 95.0,
        "up": 160.0,
        "min_rom": 42.0,
        "valid_rom": 0.75,
    },
    "squat": {
        "triples": [("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")],
        "down": 100.0,
        "up": 160.0,
        "min_rom": 38.0,
        "valid_rom": 0.75,
    },
    "situp": {
        "triples": [("left_shoulder", "left_hip", "left_knee"), ("right_shoulder", "right_hip", "right_knee")],
        "down": 105.0,
        "up": 155.0,
        "min_rom": 28.0,
        "valid_rom": 0.72,
    },
}

LANDMARK_NAMES = [
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
]


def load_labels(path: Path) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["vid_id"])] = row["class"]
    return labels


def row_to_keypoints(row: dict) -> dict:
    keypoints = {}
    for name in LANDMARK_NAMES:
        try:
            x = float(row[f"x_{name}"])
            y = float(row[f"y_{name}"])
            z = float(row[f"z_{name}"])
        except (KeyError, TypeError, ValueError):
            continue
        keypoints[name] = {
            "x": x,
            "y": y,
            "z": z,
            "visibility": 1.0,
            "px": int(x),
            "py": int(y),
        }
    return keypoints


def angle3(a: dict, b: dict, c: dict) -> Optional[float]:
    abx = a["x"] - b["x"]
    aby = a["y"] - b["y"]
    cbx = c["x"] - b["x"]
    cby = c["y"] - b["y"]
    denom = math.hypot(abx, aby) * math.hypot(cbx, cby)
    if denom == 0:
        return None
    cos = max(-1.0, min(1.0, (abx * cbx + aby * cby) / denom))
    return math.degrees(math.acos(cos))


def mean_angle(kp: dict, triples: List[Tuple[str, str, str]]) -> Optional[float]:
    values = []
    for a, b, c in triples:
        if a in kp and b in kp and c in kp:
            value = angle3(kp[a], kp[b], kp[c])
            if value is not None:
                values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int((p / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


class AngleRepCounter:
    def __init__(self, rule: dict):
        self.rule = rule
        self.phase = "up"
        self.rep_count = 0
        self.rep_breakdown: List[dict] = []
        self.min_angle = 180.0
        self.max_angle = 0.0
        self.low_frames = 0
        self.high_frames = 0
        self.angle_history: List[float] = []

    def analyze(self, kp: dict) -> dict:
        angle = mean_angle(kp, self.rule["triples"])
        if angle is None:
            return {"phase": self.phase}

        self.angle_history.append(angle)
        if len(self.angle_history) > 180:
            self.angle_history = self.angle_history[-180:]

        low = percentile(self.angle_history, 5) or angle
        high = percentile(self.angle_history, 95) or angle
        observed_rom = high - low
        down = self.rule["down"]
        up = self.rule["up"]
        if observed_rom >= 16:
            down = low + max(5.0, 0.25 * observed_rom)
            up = low + max(12.0, 0.68 * observed_rom)
            if up <= down + 8:
                up = down + 8

        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

        if angle < down:
            self.low_frames += 1
            self.high_frames = 0
        elif angle > up:
            self.high_frames += 1
            self.low_frames = 0
        else:
            self.low_frames = max(0, self.low_frames - 1)
            self.high_frames = max(0, self.high_frames - 1)

        if self.phase == "up" and angle < down + 12:
            self.phase = "downward"
        if self.phase == "downward" and self.low_frames >= 2:
            self.phase = "bottom"
        if self.phase == "bottom" and angle > down + 12:
            self.phase = "upward"

        if self.phase == "upward" and self.high_frames >= 2:
            self.phase = "up"
            self.rep_count += 1
            rom_score = max(0.0, min(1.0, (up - self.min_angle) / max(1.0, up - down)))
            faults = []
            if (self.max_angle - self.min_angle) < self.rule["min_rom"] or rom_score < self.rule["valid_rom"]:
                faults.append("limited_range")
            if self.min_angle > down + 10:
                faults.append("insufficient_depth")
            if self.max_angle < up - 8:
                faults.append("incomplete_lockout")
            quality = max(0.0, min(100.0, 100.0 * (0.75 * rom_score + 0.25) - 8.0 * len(faults)))
            self.rep_breakdown.append(
                {
                    "rep": self.rep_count,
                    "quality_score": round(quality, 1),
                    "rom_percent": round(rom_score * 100, 1),
                    "faults": faults,
                }
            )
            self.min_angle = angle
            self.max_angle = angle

        return {"phase": self.phase}


class JumpingJackCounter:
    def __init__(self):
        self.phase = "closed"
        self.rep_count = 0
        self.rep_breakdown: List[dict] = []
        self.current_open: Optional[dict] = None

    def analyze(self, kp: dict) -> dict:
        required = ("left_wrist", "right_wrist", "left_ankle", "right_ankle", "left_shoulder", "right_shoulder")
        if any(name not in kp for name in required):
            return {"phase": self.phase}
        shoulder_w = max(1e-5, abs(kp["left_shoulder"]["x"] - kp["right_shoulder"]["x"]))
        hand_span = abs(kp["left_wrist"]["x"] - kp["right_wrist"]["x"]) / shoulder_w
        foot_span = abs(kp["left_ankle"]["x"] - kp["right_ankle"]["x"]) / shoulder_w
        open_pose = hand_span > 1.85 and foot_span > 1.45
        closed_pose = hand_span < 1.45 and foot_span < 1.25

        if self.phase == "closed" and open_pose:
            self.phase = "open"
            self.current_open = {"hand_span": hand_span, "foot_span": foot_span}
        elif self.phase == "open" and self.current_open:
            self.current_open["hand_span"] = max(self.current_open["hand_span"], hand_span)
            self.current_open["foot_span"] = max(self.current_open["foot_span"], foot_span)

        if self.phase == "open" and closed_pose:
            self.phase = "closed"
            self.rep_count += 1
            peak = self.current_open or {"hand_span": hand_span, "foot_span": foot_span}
            faults = []
            if peak["hand_span"] < 2.2:
                faults.append("arm_range")
            if peak["foot_span"] < 1.75:
                faults.append("leg_range")
            quality = max(0.0, min(100.0, ((peak["hand_span"] / 2.4 + peak["foot_span"] / 1.9) / 2) * 100 - 8 * len(faults)))
            self.rep_breakdown.append(
                {
                    "rep": self.rep_count,
                    "quality_score": round(quality, 1),
                    "hand_span_ratio": round(peak["hand_span"], 2),
                    "foot_span_ratio": round(peak["foot_span"], 2),
                    "faults": faults,
                }
            )
            self.current_open = None

        return {"phase": self.phase}


def make_counter(exercise: str):
    if exercise == "jumping_jack":
        return JumpingJackCounter()
    return AngleRepCounter(ANGLE_RULES[exercise])


def iter_video_rows(landmarks_path: Path) -> Iterator[Tuple[int, List[dict]]]:
    current_vid: Optional[int] = None
    rows: List[dict] = []
    with landmarks_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = int(row["vid_id"])
            if current_vid is None:
                current_vid = vid
            if vid != current_vid:
                yield current_vid, rows
                current_vid = vid
                rows = []
            rows.append(row)
    if current_vid is not None:
        yield current_vid, rows


def analyze_sequence(rows: Iterable[dict], exercise: str, frame_stride: int = 1) -> dict:
    analyzer = make_counter(exercise)
    frame_count = 0
    analyzed_frames = 0
    last_phase = getattr(analyzer, "phase", "unknown")

    for frame_count, row in enumerate(rows, start=1):
        if frame_stride > 1 and (frame_count - 1) % frame_stride != 0:
            continue
        kp = row_to_keypoints(row)
        if not kp:
            continue
        result = analyzer.analyze(kp)
        analyzed_frames += 1
        last_phase = result.get("phase", last_phase)

    rep_breakdown = getattr(analyzer, "rep_breakdown", [])
    strict_valid_reps = sum(
        1
        for rep in rep_breakdown
        if not rep.get("faults") and float(rep.get("quality_score", 0) or 0) >= 82
    )
    avg_quality = (
        sum(float(rep.get("quality_score", 0) or 0) for rep in rep_breakdown) / len(rep_breakdown)
        if rep_breakdown
        else 0.0
    )

    return {
        "predicted_total_reps": getattr(analyzer, "rep_count", 0),
        "predicted_valid_reps": strict_valid_reps,
        "predicted_avg_rep_quality": round(avg_quality, 1),
        "frames": frame_count,
        "analyzed_frames": analyzed_frames,
        "last_phase": last_phase,
        "rep_breakdown": rep_breakdown,
    }


def load_calibration(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibration(result: dict, exercise: str, calibration: dict) -> dict:
    exercise_model = calibration.get("exercises", {}).get(exercise)
    if not exercise_model:
        return result

    calibrated = dict(result)
    for key, model_key in (
        ("predicted_total_reps", "total"),
        ("predicted_valid_reps", "valid"),
    ):
        model = exercise_model.get(model_key)
        if not model:
            continue
        raw = float(result[key])
        calibrated[key] = max(0, int(round(model.get("slope", 1.0) * raw + model.get("intercept", 0.0))))

    calibrated["predicted_valid_reps"] = min(
        calibrated["predicted_valid_reps"],
        calibrated["predicted_total_reps"],
    )
    return calibrated


def load_manual_validation(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    manual = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vid_id = int(row["vid_id"])
            except (TypeError, ValueError):
                continue
            manual[vid_id] = row
    return manual


def parse_optional_int(value: object) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return int(text)
    except ValueError:
        return None


def write_validation_template(path: Path, labels: Dict[int, str], per_class: int) -> None:
    selected: List[Tuple[int, str, str]] = []
    counts = Counter()
    for vid_id in sorted(labels):
        dataset_class = labels[vid_id]
        exercise = DATASET_TO_APP_EXERCISE.get(dataset_class)
        if not exercise:
            continue
        if counts[exercise] >= per_class:
            continue
        selected.append((vid_id, dataset_class, exercise))
        counts[exercise] += 1

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "vid_id",
                "dataset_class",
                "exercise",
                "actual_total_reps",
                "actual_valid_reps",
                "notes",
            ],
        )
        writer.writeheader()
        for vid_id, dataset_class, exercise in selected:
            writer.writerow(
                {
                    "vid_id": vid_id,
                    "dataset_class": dataset_class,
                    "exercise": exercise,
                    "actual_total_reps": "",
                    "actual_valid_reps": "",
                    "notes": "",
                }
            )


def evaluate(args: argparse.Namespace) -> None:
    dataset_dir = Path(args.dataset_dir)
    labels_path = dataset_dir / "labels.csv"
    landmarks_path = dataset_dir / "landmarks.csv"
    predictions_path = Path(args.predictions_out)
    validation_path = Path(args.validation_labels)

    labels = load_labels(labels_path)
    manual = load_manual_validation(validation_path)
    calibration = load_calibration(Path(args.calibration_model)) if args.use_calibration else {}

    if args.write_template or not validation_path.exists():
        write_validation_template(validation_path, labels, args.template_per_class)
        print(f"Wrote manual validation template: {validation_path}")

    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts = Counter(labels.values())
    processed = 0
    skipped = Counter()
    comparison_rows = []

    with predictions_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "vid_id",
            "dataset_class",
            "exercise",
            "predicted_total_reps",
            "predicted_valid_reps",
            "predicted_avg_rep_quality",
            "actual_total_reps",
            "actual_valid_reps",
            "total_rep_error",
            "valid_rep_error",
            "frames",
            "analyzed_frames",
            "last_phase",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for vid_id, rows in iter_video_rows(landmarks_path):
            dataset_class = labels.get(vid_id, "")
            exercise = DATASET_TO_APP_EXERCISE.get(dataset_class)
            if not exercise:
                skipped[dataset_class or "unlabeled"] += 1
                continue
            if args.limit and processed >= args.limit:
                break

            raw_result = analyze_sequence(rows, exercise, frame_stride=args.frame_stride)
            result = apply_calibration(raw_result, exercise, calibration)
            actual = manual.get(vid_id, {})
            actual_total = parse_optional_int(actual.get("actual_total_reps"))
            actual_valid = parse_optional_int(actual.get("actual_valid_reps"))
            total_error = (
                result["predicted_total_reps"] - actual_total
                if actual_total is not None
                else ""
            )
            valid_error = (
                result["predicted_valid_reps"] - actual_valid
                if actual_valid is not None
                else ""
            )

            out = {
                "vid_id": vid_id,
                "dataset_class": dataset_class,
                "exercise": exercise,
                "predicted_total_reps": result["predicted_total_reps"],
                "predicted_valid_reps": result["predicted_valid_reps"],
                "predicted_avg_rep_quality": result["predicted_avg_rep_quality"],
                "actual_total_reps": actual_total if actual_total is not None else "",
                "actual_valid_reps": actual_valid if actual_valid is not None else "",
                "total_rep_error": total_error,
                "valid_rep_error": valid_error,
                "frames": result["frames"],
                "analyzed_frames": result["analyzed_frames"],
                "last_phase": result["last_phase"],
                "notes": actual.get("notes", ""),
            }
            writer.writerow(out)
            if actual_total is not None or actual_valid is not None:
                comparison_rows.append(out)
            processed += 1

    print("Dataset class counts:")
    for label, count in class_counts.most_common():
        print(f"  {label}: {count}")
    print(f"Processed supported sequences: {processed}")
    if skipped:
        print("Skipped unsupported classes:")
        for label, count in skipped.most_common():
            print(f"  {label}: {count}")
    print(f"Wrote predictions: {predictions_path}")

    if comparison_rows:
        total_errors = [
            abs(int(row["total_rep_error"]))
            for row in comparison_rows
            if row["total_rep_error"] != ""
        ]
        valid_errors = [
            abs(int(row["valid_rep_error"]))
            for row in comparison_rows
            if row["valid_rep_error"] != ""
        ]
        if total_errors:
            print(f"Mean absolute total-rep error: {sum(total_errors) / len(total_errors):.2f}")
        if valid_errors:
            print(f"Mean absolute valid-rep error: {sum(valid_errors) / len(valid_errors):.2f}")
    else:
        print("No manual rep labels filled yet, so accuracy comparison was skipped.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="archive (3)", help="Folder containing labels.csv and landmarks.csv")
    parser.add_argument(
        "--validation-labels",
        default="archive (3)/validation_rep_labels.csv",
        help="Manual labels CSV with actual_total_reps and actual_valid_reps",
    )
    parser.add_argument(
        "--predictions-out",
        default="archive (3)/rep_count_predictions.csv",
        help="Where to write predicted rep counts",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit supported sequences processed; 0 means all")
    parser.add_argument("--frame-stride", type=int, default=1, help="Analyze every Nth frame")
    parser.add_argument("--write-template", action="store_true", help="Rewrite the manual validation template")
    parser.add_argument("--template-per-class", type=int, default=5, help="Rows per supported class in template")
    parser.add_argument(
        "--calibration-model",
        default="archive (3)/rep_calibration.json",
        help="Optional JSON calibration model trained from manual labels",
    )
    parser.add_argument(
        "--no-calibration",
        dest="use_calibration",
        action="store_false",
        help="Ignore the calibration model even if it exists",
    )
    parser.set_defaults(use_calibration=True)
    return parser


if __name__ == "__main__":
    evaluate(build_parser().parse_args())
