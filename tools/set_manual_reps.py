"""Set manual total/valid rep labels for a video in the archive CSVs.

Example:
  python tools/set_manual_reps.py --dataset-dir "archive (3)" --vid-id 309 --total 4 --valid 4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List


VALIDATION_FIELDS = ["vid_id", "dataset_class", "exercise", "actual_total_reps", "actual_valid_reps", "notes"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def labels_by_vid(path: Path) -> Dict[int, str]:
    labels = {}
    for row in read_rows(path):
        labels[int(row["vid_id"])] = row["class"]
    return labels


def sync_prediction_actuals(path: Path, vid_id: int, total: int, valid: int) -> None:
    rows = read_rows(path)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    for row in rows:
        if int(row["vid_id"]) == vid_id:
            row["actual_total_reps"] = str(total)
            row["actual_valid_reps"] = str(valid)
            if row.get("predicted_total_reps") not in ("", None):
                row["total_rep_error"] = str(abs(int(float(row["predicted_total_reps"])) - total))
            if row.get("predicted_valid_reps") not in ("", None):
                row["valid_rep_error"] = str(abs(int(float(row["predicted_valid_reps"])) - valid))
            row["notes"] = "manual label"
            break
    write_rows(path, fieldnames, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="archive (3)")
    parser.add_argument("--vid-id", type=int, required=True)
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--valid", type=int, required=True)
    parser.add_argument("--notes", default="manual label")
    args = parser.parse_args()

    if args.valid > args.total:
        raise SystemExit("--valid cannot be greater than --total")

    dataset_dir = Path(args.dataset_dir)
    labels = labels_by_vid(dataset_dir / "labels.csv")
    dataset_class = labels.get(args.vid_id, "unknown")
    exercise = {
        "jumping_jack": "jumping_jack",
        "push_up": "pushup",
        "situp": "situp",
        "squat": "squat",
    }.get(dataset_class, dataset_class)

    validation_path = dataset_dir / "validation_rep_labels.csv"
    rows = read_rows(validation_path)
    updated = False
    for row in rows:
        if int(row["vid_id"]) == args.vid_id:
            row.update(
                {
                    "dataset_class": dataset_class,
                    "exercise": exercise,
                    "actual_total_reps": str(args.total),
                    "actual_valid_reps": str(args.valid),
                    "notes": args.notes,
                }
            )
            updated = True
            break

    if not updated:
        rows.append(
            {
                "vid_id": str(args.vid_id),
                "dataset_class": dataset_class,
                "exercise": exercise,
                "actual_total_reps": str(args.total),
                "actual_valid_reps": str(args.valid),
                "notes": args.notes,
            }
        )

    rows.sort(key=lambda row: int(row["vid_id"]))
    write_rows(validation_path, VALIDATION_FIELDS, rows)
    sync_prediction_actuals(dataset_dir / "rep_count_predictions.csv", args.vid_id, args.total, args.valid)
    print(f"Saved manual reps for vid_id={args.vid_id}: total={args.total}, valid={args.valid}")


if __name__ == "__main__":
    main()
