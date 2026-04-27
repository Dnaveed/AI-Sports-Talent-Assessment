"""Export a landmark sequence as a self-contained HTML skeleton viewer.

This avoids matplotlib/video dependencies and works by drawing the sequence on
an HTML canvas in the browser.

Example:
  python tools/export_landmark_sequence_html.py --dataset-dir "archive (3)" --vid-id 309
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    predictions = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            predictions[int(row["vid_id"])] = row
    return predictions


def load_manual_labels(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    labels = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            labels[int(row["vid_id"])] = row
    return labels


def point(row: dict, name: str) -> Optional[Tuple[float, float]]:
    try:
        return float(row[f"x_{name}"]), float(row[f"y_{name}"])
    except (KeyError, TypeError, ValueError):
        return None


def load_sequence(path: Path, vid_id: int) -> List[dict]:
    frames = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            current = int(row["vid_id"])
            if current == vid_id:
                pts = {}
                for name in LANDMARKS:
                    p = point(row, name)
                    if p:
                        pts[name] = {"x": p[0], "y": p[1]}
                frames.append(pts)
            elif frames and current != vid_id:
                break
    if not frames:
        raise ValueError(f"No frames found for vid_id={vid_id}")
    return frames


def compute_bounds(frames: List[dict]) -> dict:
    xs = []
    ys = []
    for frame in frames:
        for p in frame.values():
            xs.append(p["x"])
            ys.append(p["y"])
    pad_x = max(8.0, (max(xs) - min(xs)) * 0.12)
    pad_y = max(8.0, (max(ys) - min(ys)) * 0.12)
    return {
        "minX": min(xs) - pad_x,
        "maxX": max(xs) + pad_x,
        "minY": min(ys) - pad_y,
        "maxY": max(ys) + pad_y,
    }


def render_html(payload: dict) -> str:
    data = json.dumps(payload)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Landmark Sequence Viewer</title>
<style>
  body {{ margin:0; background:#0f172a; color:#e5e7eb; font-family:Arial, sans-serif; }}
  .wrap {{ max-width:980px; margin:0 auto; padding:18px; }}
  .bar {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin-bottom:12px; }}
  button {{ background:#06b6d4; color:#001018; border:0; border-radius:6px; padding:8px 12px; font-weight:700; cursor:pointer; }}
  input[type=range] {{ flex:1; min-width:220px; }}
  canvas {{ width:100%; height:auto; background:#f8fafc; border-radius:8px; display:block; }}
  .meta {{ color:#94a3b8; font-size:14px; margin-bottom:10px; }}
  .count {{ font-family:Consolas, monospace; color:#67e8f9; }}
</style>
</head>
<body>
<div class="wrap">
  <h2>Landmark Sequence Viewer</h2>
  <div class="meta" id="meta"></div>
  <canvas id="canvas" width="820" height="760"></canvas>
  <div class="bar">
    <button id="play">Pause</button>
    <input id="scrub" type="range" min="0" value="0">
    <span class="count" id="counter"></span>
  </div>
</div>
<script>
const DATA = {data};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const scrub = document.getElementById('scrub');
const counter = document.getElementById('counter');
const playBtn = document.getElementById('play');
const meta = document.getElementById('meta');
const edges = DATA.edges;
let frame = 0;
let playing = true;
scrub.max = DATA.frames.length - 1;
const countLabel = DATA.manual ? 'manual' : 'predicted';
meta.textContent = `vid_id=${{DATA.vidId}} | class=${{DATA.label}} | ${{countLabel}} total=${{DATA.totalReps}} valid=${{DATA.validReps}}`;

function sx(x) {{
  return ((x - DATA.bounds.minX) / (DATA.bounds.maxX - DATA.bounds.minX)) * canvas.width;
}}
function sy(y) {{
  return ((y - DATA.bounds.minY) / (DATA.bounds.maxY - DATA.bounds.minY)) * canvas.height;
}}
function draw() {{
  const pts = DATA.frames[frame];
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 5;
  ctx.strokeStyle = '#1e293b';
  ctx.fillStyle = '#0891b2';
  for (const [a,b] of edges) {{
    if (!pts[a] || !pts[b]) continue;
    ctx.beginPath();
    ctx.moveTo(sx(pts[a].x), sy(pts[a].y));
    ctx.lineTo(sx(pts[b].x), sy(pts[b].y));
    ctx.stroke();
  }}
  for (const p of Object.values(pts)) {{
    ctx.beginPath();
    ctx.arc(sx(p.x), sy(p.y), 6, 0, Math.PI * 2);
    ctx.fill();
  }}
  counter.textContent = `frame ${{frame + 1}} / ${{DATA.frames.length}}`;
  scrub.value = frame;
}}
function tick() {{
  if (playing) {{
    frame = (frame + 1) % DATA.frames.length;
    draw();
  }}
  setTimeout(tick, DATA.intervalMs);
}}
playBtn.onclick = () => {{
  playing = !playing;
  playBtn.textContent = playing ? 'Pause' : 'Play';
}};
scrub.oninput = () => {{
  frame = Number(scrub.value);
  draw();
}};
draw();
tick();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="archive (3)")
    parser.add_argument("--vid-id", type=int, required=True)
    parser.add_argument("--out", default="")
    parser.add_argument("--interval-ms", type=int, default=45)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    labels = load_labels(dataset_dir / "labels.csv")
    predictions = load_predictions(dataset_dir / "rep_count_predictions.csv")
    manual_labels = load_manual_labels(dataset_dir / "validation_rep_labels.csv")
    frames = load_sequence(dataset_dir / "landmarks.csv", args.vid_id)
    pred = predictions.get(args.vid_id, {})
    manual = manual_labels.get(args.vid_id, {})
    total_reps = manual.get("actual_total_reps") or pred.get("predicted_total_reps", "?")
    valid_reps = manual.get("actual_valid_reps") or pred.get("predicted_valid_reps", "?")
    payload = {
        "vidId": args.vid_id,
        "label": labels.get(args.vid_id, "unknown"),
        "manual": bool(manual.get("actual_total_reps") or manual.get("actual_valid_reps")),
        "totalReps": total_reps,
        "validReps": valid_reps,
        "predictedTotal": pred.get("predicted_total_reps", "?"),
        "predictedValid": pred.get("predicted_valid_reps", "?"),
        "frames": frames,
        "bounds": compute_bounds(frames),
        "edges": EDGES,
        "intervalMs": args.interval_ms,
    }
    out = Path(args.out) if args.out else dataset_dir / f"landmark_viewer_vid_{args.vid_id}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(payload), encoding="utf-8")
    print(f"Wrote viewer: {out}")


if __name__ == "__main__":
    main()
