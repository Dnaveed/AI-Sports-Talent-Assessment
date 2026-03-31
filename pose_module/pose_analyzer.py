"""
Pose Analysis Module - upgraded rule engine with:
- landmark smoothing and visibility gating
- exercise-specific phase state machines
- per-rep quality breakdown
- hybrid scoring (rule score + ML-style proxy score)
- confidence and detailed feedback payloads
"""

import cv2
import numpy as np
import json
import urllib.request
from dataclasses import dataclass, asdict
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
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

KEY_NAMES = [
    "nose",
    "left_ear",
    "right_ear",
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
                "x": float(lm.x),
                "y": float(lm.y),
                "z": float(lm.z),
                "visibility": float(getattr(lm, "visibility", 1.0)),
                "px": int(lm.x * frame_width),
                "py": int(lm.y * frame_height),
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
            "x": (a["x"] + b["x"]) / 2,
            "y": (a["y"] + b["y"]) / 2,
            "z": (a["z"] + b["z"]) / 2,
            "visibility": min(a["visibility"], b["visibility"]),
            "px": int((a.get("px", 0) + b.get("px", 0)) / 2),
            "py": int((a.get("py", 0) + b.get("py", 0)) / 2),
        }

    @staticmethod
    def check_face_visible(keypoints: dict, min_vis: float = 0.5) -> bool:
        return (
            sum(
                1
                for p in ("nose", "left_ear", "right_ear")
                if keypoints.get(p, {}).get("visibility", 0) > min_vis
            )
            >= 2
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
        self.current_issues: List[str] = []
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

        # Calibrate once enough motion is observed to handle side-view videos.
        if self._adaptive_ready or len(self._angle_history) < 24:
            return

        p_low = float(np.percentile(self._angle_history, 5))
        p_high = float(np.percentile(self._angle_history, 95))
        rom = p_high - p_low
        if rom < 20:
            return

        down_thr = p_low + max(6.0, 0.20 * rom)
        # Use a ROM-relative top target so partial-but-valid lockout still counts.
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
            return {
                "rep_count": self.rep_count,
                "phase": self.phase,
                "correctness": 0.0,
                "issues": ["Key joints not visible"],
                "angles": {},
            }

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

        phase_map = {
            "ready": "setup",
            "up": "finish",
            "eccentric": "eccentric",
            "bottom": "bottom",
            "concentric": "concentric",
            "finish": "finish",
        }
        p_name = phase_map.get(self.phase, "setup")
        self.phase_scores[p_name].append(max(0.0, score))

        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "correctness": max(0.0, score),
            "issues": issues,
            "angles": {"avg": round(angle, 1)},
            "rep_event": rep_event,
        }


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
            return {
                "rep_count": self.rep_count,
                "phase": self.phase,
                "correctness": 0.0,
                "issues": ["Limbs not fully visible"],
                "angles": {},
            }

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
                rep_faults.append("arm_range")
                score -= 12
            if foot_span < 1.9:
                rep_faults.append("leg_range")
                score -= 12
            for f in rep_faults:
                self._add_fault(f)
            rep_event = {
                "rep": self.rep_count,
                "hand_span_ratio": round(hand_span, 2),
                "foot_span_ratio": round(foot_span, 2),
                "quality_score": round(max(0.0, score), 1),
                "faults": rep_faults,
            }
            self.rep_breakdown.append(rep_event)
            self.phase_scores["finish"].append(max(0.0, score))

        if not open_pose and self.phase == "open":
            issues.append("Extend both arms and legs fully at the top")
            score -= 10

        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "correctness": max(0.0, score),
            "issues": issues,
            "angles": {"hand_span": round(hand_span, 2), "foot_span": round(foot_span, 2)},
            "rep_event": rep_event,
        }


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
            return {
                "rep_count": self.rep_count,
                "phase": self.phase,
                "correctness": 0.0,
                "issues": ["Body not fully visible"],
                "jump_height_cm": None,
            }

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
                        rep_faults.append("low_explosiveness")
                        score -= 14
                if ankle_y > 0.96:
                    rep_faults.append("heavy_landing")
                    score -= 10

                for f in rep_faults:
                    self._add_fault(f)

                rep_event = {
                    "rep": self.rep_count,
                    "jump_height_cm": round(jump_cm, 1) if jump_cm is not None else None,
                    "quality_score": round(max(0.0, score), 1),
                    "faults": rep_faults,
                }
                self.rep_breakdown.append(rep_event)
                self.phase_scores["finish"].append(max(0.0, score))

        if not airborne and self.phase == "ground" and self.baseline_hip_y and hip_y > self.baseline_hip_y + 0.03:
            issues.append("Use a quicker transition from dip to jump")
            score -= 8

        avg = round(float(np.mean(self.jump_heights)), 1) if self.jump_heights else None
        return {
            "rep_count": self.rep_count,
            "phase": self.phase,
            "correctness": max(0.0, score),
            "issues": issues,
            "jump_height_cm": avg,
            "rep_event": rep_event,
        }


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


class VideoProcessor:
    ANALYSIS_VERSION = "v3.0-hybrid-rules"

    @staticmethod
    def _required_joints_for_exercise(exercise_type: str) -> List[str]:
        base = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
        if exercise_type in (ExerciseType.SQUAT, ExerciseType.LUNGE, ExerciseType.VERTICAL_JUMP):
            return base + ["left_knee", "right_knee", "left_ankle", "right_ankle"]
        if exercise_type == ExerciseType.JUMPING_JACK:
            return base + ["left_wrist", "right_wrist", "left_ankle", "right_ankle"]
        if exercise_type == ExerciseType.PUSHUP:
            return base + ["left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        if exercise_type == ExerciseType.SITUP:
            return base + ["left_knee", "right_knee"]
        return base

    @staticmethod
    def _invalid_capture_thresholds(exercise_type: str) -> Tuple[float, float]:
        # Sit-up recordings are often side-angle; allow lower usable-frame ratio and
        # higher partial-joint occlusion before marking the whole attempt invalid.
        if exercise_type == ExerciseType.SITUP:
            return 0.25, 0.70
        return 0.35, 0.50

    def _exercise_joints_visible(self, keypoints: dict, exercise_type: str, min_vis: float = 0.45) -> bool:
        if exercise_type == ExerciseType.SITUP:
            left_chain = ("left_shoulder", "left_hip", "left_knee")
            right_chain = ("right_shoulder", "right_hip", "right_knee")
            left_ok = sum(1 for name in left_chain if keypoints.get(name, {}).get("visibility", 0) >= min_vis) >= 2
            right_ok = sum(1 for name in right_chain if keypoints.get(name, {}).get("visibility", 0) >= min_vis) >= 2
            core_ok = sum(
                1
                for name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                if keypoints.get(name, {}).get("visibility", 0) >= min_vis
            ) >= 2
            return core_ok and (left_ok or right_ok)

        required = self._required_joints_for_exercise(exercise_type)
        if not required:
            return True
        visible = sum(1 for name in required if keypoints.get(name, {}).get("visibility", 0) >= min_vis)
        return (visible / len(required)) >= 0.75

    def get_exercise_analyzer(self, exercise_type: str):
        et = ExerciseType(exercise_type)
        if et == ExerciseType.PUSHUP:
            return GenericAngleRepAnalyzer(
                name="pushup",
                joint_triples=[("left_shoulder", "left_elbow", "left_wrist"), ("right_shoulder", "right_elbow", "right_wrist")],
                down_thr=95,
                up_thr=160,
            )
        if et == ExerciseType.SQUAT:
            return GenericAngleRepAnalyzer(
                name="squat",
                joint_triples=[("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")],
                down_thr=100,
                up_thr=160,
            )
        if et == ExerciseType.SITUP:
            return GenericAngleRepAnalyzer(
                name="situp",
                joint_triples=[("left_shoulder", "left_hip", "left_knee"), ("right_shoulder", "right_hip", "right_knee")],
                down_thr=105,
                up_thr=155,
            )
        if et == ExerciseType.LUNGE:
            return GenericAngleRepAnalyzer(
                name="lunge",
                joint_triples=[("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")],
                down_thr=96,
                up_thr=156,
            )
        if et == ExerciseType.JUMPING_JACK:
            return JumpingJackAnalyzer()
        if et == ExerciseType.VERTICAL_JUMP:
            return VerticalJumpAnalyzer()
        return GenericAngleRepAnalyzer(
            name="default",
            joint_triples=[("left_hip", "left_knee", "left_ankle"), ("right_hip", "right_knee", "right_ankle")],
            down_thr=100,
            up_thr=160,
        )

    @staticmethod
    def _mean_or_zero(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def _ml_proxy_score(
        self,
        usable_rate: float,
        conf_vals: List[float],
        frame_scores: List[float],
        rep_breakdown: List[dict],
    ) -> float:
        conf_avg = self._mean_or_zero(conf_vals)
        frame_std = float(np.std(frame_scores)) if frame_scores else 30.0
        rep_scores = [r.get("quality_score", 0.0) for r in rep_breakdown]
        rep_std = float(np.std(rep_scores)) if rep_scores else 25.0
        rep_avg = self._mean_or_zero(rep_scores)

        # Proxy model score from temporal consistency + confidence calibration.
        ml = (
            30.0 * usable_rate
            + 35.0 * conf_avg
            + 0.25 * rep_avg
            + max(0.0, 20.0 - frame_std * 0.9)
            + max(0.0, 15.0 - rep_std * 1.2)
        )
        return max(0.0, min(100.0, ml))

    def _build_recommendations(self, fault_counts: Dict[str, int], exercise_type: str) -> List[str]:
        mapping = {
            "insufficient_depth": "Increase range of motion to hit proper depth each rep.",
            "core_alignment": "Brace your core and keep trunk alignment stable.",
            "incomplete_lockout": "Finish each rep with full extension at the top.",
            "arm_range": "Raise arms higher and wider at the top phase.",
            "leg_range": "Widen stance more during open phase.",
            "low_explosiveness": "Add a stronger countermovement and explosive takeoff.",
            "heavy_landing": "Land softer with better knee and ankle control.",
        }
        top_faults = sorted(fault_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        rec = [mapping.get(k, f"Improve {k.replace('_', ' ')}.") for k, _ in top_faults]
        if not rec:
            rec = [f"Good consistency in {exercise_type}; maintain this technical standard."]
        return rec

    def process_video(self, video_path: str, exercise_type: str, progress_callback=None) -> VideoAnalysisResult:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        # Some WebM clips report corrupt FPS (e.g., 1000) which under-samples frames.
        fps = raw_fps if 1.0 <= raw_fps <= 120.0 else 30.0
        raw_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        total_frames = raw_total_frames if raw_total_frames > 0 else 0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if total_frames > 0 and fps > 0 else 0.0

        analyzer = self.get_exercise_analyzer(exercise_type)
        cheat = CheatDetector()
        pose = PoseAnalyzer()
        smoother = LandmarkSmoother(alpha=0.35)

        frame_analyses = []
        frame_scores = []
        conf_vals = []
        usable_frames = 0
        low_joint_visibility_frames = 0

        sample_every = max(1, int(fps / 10))
        frame_number = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1
                if frame_number % sample_every != 0:
                    continue

                timestamp_ms = int((frame_number / max(1.0, fps)) * 1000)
                result = pose.process_frame(frame, timestamp_ms)

                if not result.pose_landmarks:
                    continue

                landmarks = result.pose_landmarks[0]
                kp_raw = PoseAnalyzer.extract_keypoints(landmarks, fw, fh)
                kp = smoother.smooth(kp_raw)
                face_vis = PoseAnalyzer.check_face_visible(kp)
                avg_conf = float(np.mean([k["visibility"] for k in kp.values()]))
                conf_vals.append(avg_conf)
                exercise_visible = self._exercise_joints_visible(kp, exercise_type)
                if not exercise_visible:
                    low_joint_visibility_frames += 1

                vis_gate = avg_conf >= 0.45 and exercise_visible
                if not vis_gate:
                    issue = (
                        "Required body joints are not visible for this exercise"
                        if not exercise_visible
                        else "Low pose confidence - improve camera angle/lighting"
                    )
                    analysis = {
                        "rep_count": analyzer.rep_count,
                        "phase": getattr(analyzer, "phase", "unknown"),
                        "correctness": 0.0,
                        "issues": [issue],
                    }
                else:
                    usable_frames += 1
                    if exercise_type == ExerciseType.VERTICAL_JUMP:
                        analysis = analyzer.analyze_frame(kp, fh)
                    else:
                        analysis = analyzer.analyze_frame(kp)

                cheat.analyze(kp, face_vis, avg_conf)
                score = float(analysis.get("correctness", 0.0))
                frame_scores.append(score)

                ts = frame_number / max(1.0, fps)
                frame_analyses.append(
                    asdict(
                        FrameAnalysis(
                            frame_number=frame_number,
                            timestamp=round(ts, 2),
                            keypoints={
                                name: {
                                    k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in vals.items()
                                    if k in ("x", "y", "visibility")
                                }
                                for name, vals in kp.items()
                            },
                            rep_count=analysis.get("rep_count", 0),
                            phase=analysis.get("phase", "unknown"),
                            correctness_score=round(score, 1),
                            issues=analysis.get("issues", []),
                            face_visible=face_vis,
                            confidence=round(avg_conf, 3),
                        )
                    )
                )

                if progress_callback and total_frames > 0:
                    progress_callback(frame_number / max(1, total_frames) * 100)
        finally:
            cap.release()
            pose.close()

        # Fall back to observed video length when container metadata is invalid.
        if duration <= 0.0 and frame_number > 0:
            duration = frame_number / max(1.0, fps)

        cheat_report = cheat.report()
        total_reps = getattr(analyzer, "rep_count", 0)
        rep_breakdown = getattr(analyzer, "rep_breakdown", [])
        fault_counts = getattr(analyzer, "fault_counts", {})
        phase_scores = getattr(analyzer, "phase_scores", {})

        jump_height = None
        if hasattr(analyzer, "jump_heights") and analyzer.jump_heights:
            jump_height = round(max(analyzer.jump_heights), 1)

        rule_score = self._mean_or_zero(frame_scores)
        usable_rate = usable_frames / max(1, len(frame_scores))
        ml_score = self._ml_proxy_score(usable_rate, conf_vals, frame_scores, rep_breakdown)
        hybrid = 0.6 * rule_score + 0.4 * ml_score

        disagreement = abs(rule_score - ml_score)
        confidence_score = max(
            0.0,
            min(
                100.0,
                100.0 * self._mean_or_zero(conf_vals) * usable_rate * (1.0 - min(1.0, disagreement / 100.0)),
            ),
        )

        invalid_reasons = []
        no_rep_ex = exercise_type in (
            ExerciseType.PUSHUP,
            ExerciseType.SQUAT,
            ExerciseType.JUMPING_JACK,
            ExerciseType.SITUP,
            ExerciseType.LUNGE,
            ExerciseType.VERTICAL_JUMP,
        )
        if len(frame_scores) < 8:
            invalid_reasons.append("insufficient_pose_samples")
        min_usable_rate, max_low_joint_fraction = self._invalid_capture_thresholds(exercise_type)
        if usable_rate < min_usable_rate:
            invalid_reasons.append("insufficient_usable_frames")
        if len(frame_scores) > 0 and (low_joint_visibility_frames / len(frame_scores)) > max_low_joint_fraction:
            invalid_reasons.append("exercise_joints_not_visible")
        if no_rep_ex and total_reps == 0 and usable_rate < 0.6:
            invalid_reasons.append("no_detectable_exercise_motion")

        invalid_attempt = len(invalid_reasons) > 0
        if invalid_attempt:
            rule_score = min(rule_score, 5.0)
            ml_score = min(ml_score, 5.0)
            hybrid = min(hybrid, 5.0)
            confidence_score = min(confidence_score, 15.0)

        quality_flags = []
        if usable_rate < 0.7:
            quality_flags.append("low_usable_frame_rate")
        if self._mean_or_zero(conf_vals) < 0.55:
            quality_flags.append("low_landmark_confidence")
        if disagreement > 18:
            quality_flags.append("rule_model_disagreement")
        if invalid_attempt:
            quality_flags.append("invalid_assessment_capture")
            for reason in invalid_reasons:
                if reason not in quality_flags:
                    quality_flags.append(reason)

        if invalid_attempt:
            cheat_report["detected"] = True
            reason_text = "Invalid capture for selected exercise"
            if reason_text not in cheat_report["reasons"]:
                cheat_report["reasons"].append(reason_text)
            if "exercise_joints_not_visible" in invalid_reasons:
                cheat_report["reasons"].append("Lower-body joints were not visible for most analyzed frames")
            if "no_detectable_exercise_motion" in invalid_reasons:
                cheat_report["reasons"].append("No detectable rep motion for the selected exercise")

        top_faults = [
            {"type": k, "count": v, "frequency_percent": round(v / max(1, total_reps) * 100, 1)}
            for k, v in sorted(fault_counts.items(), key=lambda x: x[1], reverse=True)
        ]

        detailed_feedback = {
            "top_faults": top_faults[:5],
            "phase_scores": {
                k: round(self._mean_or_zero(v), 1) if isinstance(v, list) else 0.0
                for k, v in phase_scores.items()
            },
            "recommendations": (
                [
                    "Re-record with full body visible (head to feet) and stable side/front camera angle.",
                    "Keep required exercise joints in frame for the full set.",
                    "Repeat the test while clearly performing the selected exercise reps.",
                ]
                if invalid_attempt
                else self._build_recommendations(fault_counts, exercise_type)
            ),
            "quality_flags": quality_flags,
            "usable_frame_rate": round(usable_rate, 3),
            "invalid_attempt": invalid_attempt,
            "invalid_reasons": invalid_reasons,
        }

        summary = {
            "total_reps": total_reps,
            "avg_correctness_score": round(hybrid, 1),
            "rule_score": round(rule_score, 1),
            "ml_score": round(ml_score, 1),
            "confidence_score": round(confidence_score, 1),
            "duration_seconds": round(duration, 1),
            "reps_per_minute": round(total_reps / (duration / 60), 1) if duration > 0 else 0,
            "exercise_type": exercise_type,
            "video_fps": round(fps, 1),
            "frames_analyzed": len(frame_analyses),
            "analysis_version": self.ANALYSIS_VERSION,
            "quality_flags": quality_flags,
        }

        perf = self._performance_metrics(exercise_type, total_reps, hybrid, jump_height)

        return VideoAnalysisResult(
            exercise_type=exercise_type,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            total_reps=total_reps,
            avg_correctness_score=round(hybrid, 1),
            jump_height_cm=jump_height,
            cheat_detected=cheat_report["detected"],
            cheat_reasons=cheat_report["reasons"],
            frame_analyses=frame_analyses,
            summary=summary,
            performance_metrics=perf,
            rule_score=round(rule_score, 1),
            ml_score=round(ml_score, 1),
            hybrid_form_score=round(hybrid, 1),
            confidence_score=round(confidence_score, 1),
            analysis_version=self.ANALYSIS_VERSION,
            detailed_feedback=detailed_feedback,
            rep_breakdown=rep_breakdown,
        )

    @staticmethod
    def _performance_metrics(exercise_type, reps, correctness, jump_height):
        bm = {
            "pushup": {"beginner": 10, "intermediate": 25, "advanced": 40},
            "squat": {"beginner": 15, "intermediate": 30, "advanced": 50},
            "jumping_jack": {"beginner": 20, "intermediate": 50, "advanced": 80},
            "vertical_jump": {"beginner": 25, "intermediate": 45, "advanced": 65},
            "situp": {"beginner": 15, "intermediate": 30, "advanced": 50},
            "lunge": {"beginner": 10, "intermediate": 24, "advanced": 40},
        }.get(exercise_type, {"beginner": 10, "intermediate": 25, "advanced": 40})

        metric = jump_height if (exercise_type == "vertical_jump" and jump_height) else reps
        if metric >= bm["advanced"]:
            level, pct = "Advanced", 90
        elif metric >= bm["intermediate"]:
            level, pct = "Intermediate", 60
        elif metric >= bm["beginner"]:
            level, pct = "Beginner", 30
        else:
            level, pct = "Needs Work", 10

        grade = (
            "A"
            if correctness >= 90
            else "B"
            if correctness >= 80
            else "C"
            if correctness >= 70
            else "D"
            if correctness >= 60
            else "F"
        )

        return {
            "fitness_level": level,
            "estimated_percentile": pct,
            "form_grade": grade,
            "benchmarks": bm,
            "metric_value": metric,
            "metric_unit": "cm" if exercise_type == "vertical_jump" else "reps",
        }


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        result = VideoProcessor().process_video(sys.argv[1], sys.argv[2])
        print(json.dumps(result.__dict__, indent=2, default=str))
    else:
        print("Usage: python pose_analyzer.py <video_path> <exercise_type>")
        print("Exercise types: pushup, squat, vertical_jump, jumping_jack, situp, lunge")
