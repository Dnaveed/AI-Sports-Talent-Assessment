"""View or export one landmark sequence from the exercise time-series dataset.

Examples:
  python tools/view_landmark_sequence.py --dataset-dir "archive (3)" --vid-id 309
  python tools/view_landmark_sequence.py --dataset-dir "archive (3)" --vid-id 309 --save outputs/vid_309.mp4
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


LANDMARKS = [
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
    "left_foot_index",
    "right_foot_index",
]

EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
]


def load_labels(path: Path) -> Dict[int, str]:
    labels = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["vid_id"])] = row["class"]
    return labels


def load_predictions(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["vid_id"])] = row
    return out


def load_sequence(path: Path, vid_id: int) -> List[dict]:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            current = int(row["vid_id"])
            if current == vid_id:
                rows.append(row)
            elif rows and current != vid_id:
                break
    if not rows:
        raise ValueError(f"No frames found for vid_id={vid_id}")
    return rows


def point(row: dict, name: str) -> Optional[Tuple[float, float]]:
    try:
        return float(row[f"x_{name}"]), float(row[f"y_{name}"])
    except (KeyError, TypeError, ValueError):
        return None


def frame_points(row: dict) -> Dict[str, Tuple[float, float]]:
    pts = {}
    for name in LANDMARKS:
        p = point(row, name)
        if p is not None:
            pts[name] = p
    return pts


def bounds(rows: List[dict]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for row in rows:
        for name in LANDMARKS:
            p = point(row, name)
            if p is None:
                continue
            xs.append(p[0])
            ys.append(p[1])
    if not xs or not ys:
        return -100, 100, -100, 100
    pad_x = max(8.0, (max(xs) - min(xs)) * 0.12)
    pad_y = max(8.0, (max(ys) - min(ys)) * 0.12)
    return min(xs) - pad_x, max(xs) + pad_x, min(ys) - pad_y, max(ys) + pad_y


def build_animation(rows: List[dict], vid_id: int, label: str, prediction: Optional[dict], interval_ms: int):
    xmin, xmax, ymin, ymax = bounds(rows)
    fig, ax = plt.subplots(figsize=(6, 7))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    pred_text = ""
    if prediction:
        pred_text = (
            f" | predicted total={prediction.get('predicted_total_reps', '?')}"
            f" valid={prediction.get('predicted_valid_reps', '?')}"
        )
    title = ax.set_title("")
    scat = ax.scatter([], [], s=28, color="#00a6d6")
    lines = [ax.plot([], [], color="#1f2937", linewidth=2)[0] for _ in EDGES]

    def update(i: int):
        row = rows[i]
        pts = frame_points(row)
        xy = list(pts.values())
        scat.set_offsets(xy if xy else [[0, 0]])
        for line, (a, b) in zip(lines, EDGES):
            if a in pts and b in pts:
                line.set_data([pts[a][0], pts[b][0]], [pts[a][1], pts[b][1]])
            else:
                line.set_data([], [])
        title.set_text(f"vid_id={vid_id} | {label} | frame {i + 1}/{len(rows)}{pred_text}")
        return [scat, title, *lines]

    anim = FuncAnimation(fig, update, frames=len(rows), interval=interval_ms, blit=False)
    return fig, anim


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="archive (3)")
    parser.add_argument("--vid-id", type=int, required=True)
    parser.add_argument("--interval-ms", type=int, default=45)
    parser.add_argument("--save", default="", help="Optional .mp4 or .gif output path")
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels = load_labels(dataset_dir / "labels.csv")
    predictions = load_predictions(dataset_dir / "rep_count_predictions.csv")
    rows = load_sequence(dataset_dir / "landmarks.csv", args.vid_id)
    label = labels.get(args.vid_id, "unknown")
    fig, anim = build_animation(rows, args.vid_id, label, predictions.get(args.vid_id), args.interval_ms)

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == ".gif":
            anim.save(out, writer=PillowWriter(fps=max(1, round(1000 / args.interval_ms))), dpi=args.dpi)
        else:
            anim.save(out, writer=FFMpegWriter(fps=max(1, round(1000 / args.interval_ms))), dpi=args.dpi)
        print(f"Saved animation: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
