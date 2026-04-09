from typing import List, Dict, Optional, Tuple
import numpy as np
from .base import PoseAnalyzer


class LungeAnalyzer:
    def __init__(self):
        self.name = "lunge"
        self.rep_count = 0
        self.phase = "ready"
        self.down_thr = 96.0
        self.up_thr = 156.0
        self.joint_triples: List[Tuple[str, str, str]] = [
            ("left_hip", "left_knee", "left_ankle"),
            ("right_hip", "right_knee", "right_ankle"),
        ]
        self.min_angle = 180.0
        self.max_angle = 0.0
        self.rep_breakdown: List[dict] = []
        self.fault_counts: Dict[str, int] = {}
        self.phase_scores = {"setup": [], "eccentric": [], "bottom": [], "concentric": [], "finish": []}

    def _add_fault(self, label: str):
        self.fault_counts[label] = self.fault_counts.get(label, 0) + 1

    def _angle_avg(self, kp: dict) -> Optional[float]:
        values = []
        for a, b, c in self.joint_triples:
            if kp.get(a) and kp.get(b) and kp.get(c):
                values.append(PoseAnalyzer.calculate_angle(kp[a], kp[b], kp[c]))
        return float(np.mean(values)) if values else None

    def _body_line_error(self, kp: dict) -> Optional[float]:
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        lk, rk = kp.get("left_knee"), kp.get("right_knee")
        if not all([ls, rs, lh, rh, lk, rk]):
            return None
        ms = PoseAnalyzer.midpoint(ls, rs)
        mh = PoseAnalyzer.midpoint(lh, rh)
        mk = PoseAnalyzer.midpoint(lk, rk)
        return abs(mh["y"] - ((ms["y"] + mk["y"]) / 2))

    def analyze_frame(self, kp: dict) -> dict:
        angle = self._angle_avg(kp)
        if angle is None:
            return {"rep_count": self.rep_count, "phase": self.phase, "correctness": 0.0,
                    "issues": ["Key joints not visible"], "angles": {}}

        issues: List[str] = []
        score = 100.0
        body_err = self._body_line_error(kp)
        self.min_angle = min(self.min_angle, angle)
        self.max_angle = max(self.max_angle, angle)

        if self.phase in ("ready", "up") and angle < (self.down_thr + 12):
            self.phase = "eccentric"
        if self.phase == "eccentric" and angle < self.down_thr:
            self.phase = "bottom"
        if self.phase == "bottom" and angle > (self.down_thr + 12):
            self.phase = "concentric"

        rep_event = None
        if self.phase == "concentric" and angle > self.up_thr:
            self.phase = "finish"
            self.rep_count += 1
            rom_pct = max(0.0, min(100.0, ((self.up_thr - self.min_angle) / max(1.0, self.up_thr - self.down_thr)) * 100.0))
            rep_score = 100.0
            rep_faults = []
            if self.min_angle > self.down_thr + 10:
                rep_faults.append("insufficient_depth"); rep_score -= 18
            if body_err is not None and body_err > 0.055:
                rep_faults.append("core_alignment"); rep_score -= 14
            if self.max_angle < self.up_thr + 5:
                rep_faults.append("incomplete_lockout"); rep_score -= 10
            for f in rep_faults:
                self._add_fault(f)
            rep_event = {"rep": self.rep_count, "min_angle": round(self.min_angle, 1),
                         "max_angle": round(self.max_angle, 1), "rom_percent": round(rom_pct, 1),
                         "quality_score": round(max(0.0, rep_score), 1), "faults": rep_faults}
            self.rep_breakdown.append(rep_event)
            self.min_angle = angle; self.max_angle = angle
            self.phase = "up"

        if body_err is not None and body_err > 0.055:
            issues.append("Maintain straighter trunk alignment"); score -= 12
        if self.phase == "bottom" and angle > self.down_thr + 10:
            issues.append("Increase depth for full range of motion"); score -= 14
        if self.phase in ("concentric", "up") and angle < self.up_thr - 10:
            issues.append("Finish each rep with stronger lockout"); score -= 8

        phase_map = {"ready": "setup", "up": "finish", "eccentric": "eccentric",
                     "bottom": "bottom", "concentric": "concentric", "finish": "finish"}
        self.phase_scores[phase_map.get(self.phase, "setup")].append(max(0.0, score))

        return {"rep_count": self.rep_count, "phase": self.phase, "correctness": max(0.0, score),
                "issues": issues, "angles": {"avg": round(angle, 1)}, "rep_event": rep_event}
