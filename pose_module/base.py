"""
Shared utilities, data classes, and core pose infrastructure used by all exercise analyzers.
"""

import cv2
import numpy as np
import urllib.request
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def ensure_model():
    if MODEL_PATH.exists():
        return
    print(f"[AthleteAI] Downloading pose model (~2 MB) -> {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[AthleteAI] Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download MediaPipe model: {e}\n"
            f"Please download manually from:\n  {MODEL_URL}\n"
            f"and place it at:  {MODEL_PATH}"
        )


LANDMARK_IDX = {
    "nose": 0, "left_eye_inner": 1, "left_eye": 2, "left_eye_outer": 3,
    "right_eye_inner": 4, "right_eye": 5, "right_eye_outer": 6,
    "left_ear": 7, "right_ear": 8, "mouth_left": 9, "mouth_right": 10,
    "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
    "left_wrist": 15, "right_wrist": 16, "left_pinky": 17, "right_pinky": 18,
    "left_index": 19, "right_index": 20, "left_thumb": 21, "right_thumb": 22,
    "left_hip": 23, "right_hip": 24, "left_knee": 25, "right_knee": 26,
    "left_ankle": 27, "right_ankle": 28, "left_heel": 29, "right_heel": 30,
    "left_foot_index": 31, "right_foot_index": 32,
}

KEY_NAMES = [
    "nose", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
]


class ExerciseType(str, Enum):
    PUSHUP = "pushup"
    SQUAT = "squat"
    JUMPING_JACK = "jumping_jack"
    VERTICAL_JUMP = "vertical_jump"
    SITUP = "situp"
    LUNGE = "lunge"


@dataclass
class FrameAnalysis:
    frame_number: int
    timestamp: float
    keypoints: dict
    rep_count: int
    phase: str
    correctness_score: float
    issues: list
    face_visible: bool
    confidence: float


@dataclass
class VideoAnalysisResult:
    exercise_type: str
    total_frames: int
    fps: float
    duration: float
    total_reps: int
    avg_correctness_score: float
    jump_height_cm: Optional[float]
    cheat_detected: bool
    cheat_reasons: list
    frame_analyses: list
    summary: dict
    performance_metrics: dict
    rule_score: float
    ml_score: float
    hybrid_form_score: float
    confidence_score: float
    analysis_version: str
    detailed_feedback: dict
    rep_breakdown: list


class PoseAnalyzer:
    def __init__(self):
        ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=str(MODEL_PATH))
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(opts)

    def process_frame(self, bgr_frame: np.ndarray, timestamp_ms: int):
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self._landmarker.close()

    @staticmethod
    def extract_keypoints(landmarks: list, frame_width: int, frame_height: int) -> dict:
        keypoints = {}
        for name in KEY_NAMES:
            idx = LANDMARK_IDX[name]
            lm = landmarks[idx]
            keypoints[name] = {
                "x": float(lm.x), "y": float(lm.y), "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
                "px": int(lm.x * frame_width), "py": int(lm.y * frame_height),
            }
        return keypoints

    @staticmethod
    def calculate_angle(a: dict, b: dict, c: dict) -> float:
        a_ = np.array([a["x"], a["y"]])
        b_ = np.array([b["x"], b["y"]])
        c_ = np.array([c["x"], c["y"]])
        ba, bc = a_ - b_, c_ - b_
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @staticmethod
    def midpoint(a: dict, b: dict) -> dict:
        return {
            "x": (a["x"] + b["x"]) / 2, "y": (a["y"] + b["y"]) / 2,
            "z": (a["z"] + b["z"]) / 2,
            "visibility": min(a["visibility"], b["visibility"]),
            "px": int((a.get("px", 0) + b.get("px", 0)) / 2),
            "py": int((a.get("py", 0) + b.get("py", 0)) / 2),
        }

    @staticmethod
    def check_face_visible(keypoints: dict, min_vis: float = 0.5) -> bool:
        return (
            sum(
                1 for p in ("nose", "left_ear", "right_ear")
                if keypoints.get(p, {}).get("visibility", 0) > min_vis
            ) >= 2
        )


class LandmarkSmoother:
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.prev = None

    def smooth(self, kp: dict) -> dict:
        if self.prev is None:
            self.prev = kp
            return kp
        out = {}
        for name, cur in kp.items():
            old = self.prev.get(name, cur)
            out[name] = {
                "x": old["x"] * (1 - self.alpha) + cur["x"] * self.alpha,
                "y": old["y"] * (1 - self.alpha) + cur["y"] * self.alpha,
                "z": old["z"] * (1 - self.alpha) + cur["z"] * self.alpha,
                "visibility": old["visibility"] * (1 - self.alpha) + cur["visibility"] * self.alpha,
                "px": int(old["px"] * (1 - self.alpha) + cur["px"] * self.alpha),
                "py": int(old["py"] * (1 - self.alpha) + cur["py"] * self.alpha),
            }
        self.prev = out
        return out


class CheatDetector:
    def __init__(self):
        self.face_vis_history = []
        self.conf_history = []
        self.speed_history = []
        self.motion_energy = []
        self._prev_kp = None

    def analyze(self, kp: dict, face_visible: bool, avg_conf: float):
        self.face_vis_history.append(face_visible)
        self.conf_history.append(avg_conf)

        if self._prev_kp and kp.get("left_hip") and self._prev_kp.get("left_hip"):
            dy = abs(kp["left_hip"]["y"] - self._prev_kp["left_hip"]["y"])
            self.speed_history.append(dy)

        if self._prev_kp:
            deltas = []
            for n in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
                if kp.get(n) and self._prev_kp.get(n):
                    deltas.append(
                        abs(kp[n]["x"] - self._prev_kp[n]["x"]) + abs(kp[n]["y"] - self._prev_kp[n]["y"])
                    )
            if deltas:
                self.motion_energy.append(float(np.mean(deltas)))

        self._prev_kp = kp

    def report(self) -> dict:
        reasons = []
        detected = False

        if len(self.face_vis_history) > 10:
            rate = sum(self.face_vis_history) / len(self.face_vis_history)
            if rate < 0.5:
                reasons.append(f"Face hidden in {round((1 - rate) * 100)}% of analyzed frames")
                detected = True

        if len(self.conf_history) > 10:
            arr = np.array(self.conf_history)
            drops = int(np.sum(np.diff(arr) < -0.35))
            if np.mean(arr) < 0.42:
                reasons.append("Very low pose confidence throughout video")
                detected = True
            if drops > max(2, int(len(arr) * 0.06)):
                reasons.append("Abrupt confidence drops suggest possible cuts")
                detected = True

        if len(self.speed_history) > 10:
            speeds = np.array(self.speed_history)
            if np.sum(speeds > 0.17) > 3:
                reasons.append("Unnaturally fast frame-to-frame body displacement")
                detected = True

        if len(self.motion_energy) > 20:
            m = np.array(self.motion_energy)
            if np.std(m) < 0.002:
                reasons.append("Motion pattern too static for natural movement")
                detected = True

        return {"detected": detected, "reasons": reasons}


class GenericAngleRepAnalyzer:
    def __init__(self, name: str, joint_triples: List[Tuple[str, str, str]], down_thr: float, up_thr: float):
        self.name = name
        self.rep_count = 0
        self.phase = "ready"
        self.down_thr = down_thr
        self.up_thr = up_thr
        self.joint_triples = joint_triples
        self.min_angle = 180.0
        self.max_angle = 0.0
        self.rep_breakdown: List[dict] = []
        self.fault_counts: Dict[str, int] = {}
        self.phase_scores = {"setup": [], "eccentric": [], "bottom": [], "concentric": [], "finish": []}
        self._angle_history: List[float] = []
        self._adaptive_ready = False

    def _add_fault(self, label: str):
        self.fault_counts[label] = self.fault_counts.get(label, 0) + 1

    def _maybe_adapt_situp_thresholds(self, angle: float) -> None:
        if self.name != "situp":
            return
        self._angle_history.append(angle)
        if len(self._angle_history) > 180:
            self._angle_history = self._angle_history[-180:]
        if self._adaptive_ready or len(self._angle_history) < 24:
            return
        p_low = float(np.percentile(self._angle_history, 5))
        p_high = float(np.percentile(self._angle_history, 95))
        rom = p_high - p_low
        if rom < 20:
            return
        down_thr = p_low + max(6.0, 0.20 * rom)
        up_thr = p_low + max(14.0, 0.68 * rom)
        down_thr = float(np.clip(down_thr, 95.0, 145.0))
        up_thr = float(np.clip(up_thr, down_thr + 20.0, 178.0))
        self.down_thr = down_thr
        self.up_thr = up_thr
        self._adaptive_ready = True

    def _angle_avg(self, kp: dict) -> Optional[float]:
        values = []
        for a, b, c in self.joint_triples:
            if kp.get(a) and kp.get(b) and kp.get(c):
                values.append(PoseAnalyzer.calculate_angle(kp[a], kp[b], kp[c]))
        if not values:
            return None
        return float(np.mean(values))

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
        self._maybe_adapt_situp_thresholds(angle)
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
                rep_faults.append("insufficient_depth")
                rep_score -= 18
            if body_err is not None and body_err > 0.055:
                rep_faults.append("core_alignment")
                rep_score -= 14
            if self.max_angle < self.up_thr + 5:
                rep_faults.append("incomplete_lockout")
                rep_score -= 10
            for f in rep_faults:
                self._add_fault(f)
            rep_event = {
                "rep": self.rep_count,
                "min_angle": round(self.min_angle, 1),
                "max_angle": round(self.max_angle, 1),
                "rom_percent": round(rom_pct, 1),
                "quality_score": round(max(0.0, rep_score), 1),
                "faults": rep_faults,
            }
            self.rep_breakdown.append(rep_event)
            self.min_angle = angle
            self.max_angle = angle
            self.phase = "up"

        if body_err is not None and body_err > 0.055:
            issues.append("Maintain straighter trunk alignment")
            score -= 12
        if self.phase == "bottom" and angle > self.down_thr + 10:
            issues.append("Increase depth for full range of motion")
            score -= 14
        if self.phase in ("concentric", "up") and angle < self.up_thr - 10:
            issues.append("Finish each rep with stronger lockout")
            score -= 8

        phase_map = {"ready": "setup", "up": "finish", "eccentric": "eccentric",
                     "bottom": "bottom", "concentric": "concentric", "finish": "finish"}
        self.phase_scores[phase_map.get(self.phase, "setup")].append(max(0.0, score))

        return {
            "rep_count": self.rep_count, "phase": self.phase,
            "correctness": max(0.0, score), "issues": issues,
            "angles": {"avg": round(angle, 1)}, "rep_event": rep_event,
        }
