from typing import List, Dict
import numpy as np


class JumpingJackAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.phase = "closed"
        self.rep_breakdown: List[dict] = []
        self.fault_counts: Dict[str, int] = {}
        self.phase_scores = {"setup": [], "eccentric": [], "bottom": [], "concentric": [], "finish": []}

    def _add_fault(self, label: str):
        self.fault_counts[label] = self.fault_counts.get(label, 0) + 1

    def analyze_frame(self, kp: dict) -> dict:
        lw, rw = kp.get("left_wrist"), kp.get("right_wrist")
        la, ra = kp.get("left_ankle"), kp.get("right_ankle")
        ls, rs = kp.get("left_shoulder"), kp.get("right_shoulder")
        if not all([lw, rw, la, ra, ls, rs]):
            return {"rep_count": self.rep_count, "phase": self.phase, "correctness": 0.0,
                    "issues": ["Limbs not fully visible"], "angles": {}}

        shoulder_w = max(1e-5, abs(ls["x"] - rs["x"]))
        hand_span = abs(lw["x"] - rw["x"]) / shoulder_w
        foot_span = abs(la["x"] - ra["x"]) / shoulder_w

        open_pose = hand_span > 2.2 and foot_span > 1.8
        closed_pose = hand_span < 1.2 and foot_span < 1.1

        score = 100.0
        issues: List[str] = []
        rep_event = None

        if self.phase == "closed" and open_pose:
            self.phase = "open"
            self.phase_scores["eccentric"].append(95.0)
        elif self.phase == "open" and closed_pose:
            self.phase = "closed"
            self.rep_count += 1
            rep_faults = []
            if hand_span < 2.4:
                rep_faults.append("arm_range"); score -= 12
            if foot_span < 1.9:
                rep_faults.append("leg_range"); score -= 12
            for f in rep_faults:
                self._add_fault(f)
            rep_event = {"rep": self.rep_count, "hand_span_ratio": round(hand_span, 2),
                         "foot_span_ratio": round(foot_span, 2),
                         "quality_score": round(max(0.0, score), 1), "faults": rep_faults}
            self.rep_breakdown.append(rep_event)
            self.phase_scores["finish"].append(max(0.0, score))

        if not open_pose and self.phase == "open":
            issues.append("Extend both arms and legs fully at the top")
            score -= 10

        return {"rep_count": self.rep_count, "phase": self.phase, "correctness": max(0.0, score),
                "issues": issues,
                "angles": {"hand_span": round(hand_span, 2), "foot_span": round(foot_span, 2)},
                "rep_event": rep_event}
