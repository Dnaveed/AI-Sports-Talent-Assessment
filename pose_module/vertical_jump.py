from typing import List, Dict, Optional
import numpy as np


class VerticalJumpAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.phase = "ground"
        self.baseline_hip_y = None
        self.min_hip_y = None
        self.jump_heights: List[float] = []
        self.athlete_height_px = None
        self.rep_breakdown: List[dict] = []
        self.fault_counts: Dict[str, int] = {}
        self.phase_scores = {"setup": [], "eccentric": [], "bottom": [], "concentric": [], "finish": []}

    def _add_fault(self, label: str):
        self.fault_counts[label] = self.fault_counts.get(label, 0) + 1

    def _calibrate(self, kp: dict):
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        la, ra = kp.get("left_ankle"), kp.get("right_ankle")
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if lh and rh:
            self.baseline_hip_y = (lh["y"] + rh["y"]) / 2
        if la and ra and ls and rs:
            ankle_y = (la["py"] + ra["py"]) / 2
            shoulder_y = (ls["py"] + rs["py"]) / 2
            self.athlete_height_px = abs(ankle_y - shoulder_y) * 1.4

    def analyze_frame(self, kp: dict, frame_height: int) -> dict:
        lh, rh = kp.get("left_hip"), kp.get("right_hip")
        la, ra = kp.get("left_ankle"), kp.get("right_ankle")
        if not all([lh, rh, la, ra]):
            return {"rep_count": self.rep_count, "phase": self.phase, "correctness": 0.0,
                    "issues": ["Body not fully visible"], "jump_height_cm": None}

        if self.baseline_hip_y is None or self.athlete_height_px is None:
            self._calibrate(kp)

        hip_y = (lh["y"] + rh["y"]) / 2
        ankle_y = (la["y"] + ra["y"]) / 2
        airborne = ankle_y < 0.92 or (
            self.baseline_hip_y is not None and (self.baseline_hip_y - hip_y) > 0.05
        )

        score = 100.0
        issues: List[str] = []
        rep_event = None

        if self.phase == "ground" and airborne:
            self.phase = "airborne"
            self.min_hip_y = hip_y
            self.phase_scores["concentric"].append(96.0)
        elif self.phase == "airborne":
            if hip_y < (self.min_hip_y or hip_y):
                self.min_hip_y = hip_y
            if not airborne:
                self.phase = "ground"
                self.rep_count += 1
                jump_cm = None
                rep_faults = []
                if self.baseline_hip_y and self.min_hip_y and self.athlete_height_px:
                    delta = self.baseline_hip_y - self.min_hip_y
                    jump_cm = delta * frame_height / self.athlete_height_px * 175
                    self.jump_heights.append(jump_cm)
                    if jump_cm < 18:
                        rep_faults.append("low_explosiveness"); score -= 14
                if ankle_y > 0.96:
                    rep_faults.append("heavy_landing"); score -= 10
                for f in rep_faults:
                    self._add_fault(f)
                rep_event = {"rep": self.rep_count,
                             "jump_height_cm": round(jump_cm, 1) if jump_cm is not None else None,
                             "quality_score": round(max(0.0, score), 1), "faults": rep_faults}
                self.rep_breakdown.append(rep_event)
                self.phase_scores["finish"].append(max(0.0, score))

        if not airborne and self.phase == "ground" and self.baseline_hip_y and hip_y > self.baseline_hip_y + 0.03:
            issues.append("Use a quicker transition from dip to jump")
            score -= 8

        avg = round(float(np.mean(self.jump_heights)), 1) if self.jump_heights else None
        return {"rep_count": self.rep_count, "phase": self.phase, "correctness": max(0.0, score),
                "issues": issues, "jump_height_cm": avg, "rep_event": rep_event}
