"""
Pose Analysis Module - AI-powered movement detection for athlete fitness evaluation.
Compatible with MediaPipe 0.10+ (new Tasks API — mp.solutions removed).

Uses: mediapipe.tasks.python.vision.PoseLandmarker
Falls back gracefully with clear error messages.
"""

import cv2
import numpy as np
import json
import os
import urllib.request
from dataclasses import dataclass, asdict
from typing import Optional
from enum import Enum
from pathlib import Path

# ── MediaPipe new Tasks API ────────────────────────────────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark


# ── Model file (downloaded once, cached locally) ──────────────────────────────
MODEL_PATH = Path(__file__).parent / "pose_landmarker_lite.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_lite/float16/latest/"
    "pose_landmarker_lite.task"
)


def ensure_model():
    """Download the pose landmarker model file if not already present."""
    if MODEL_PATH.exists():
        return
    print(f"[AthleteAI] Downloading pose model (~2 MB) → {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[AthleteAI] Model downloaded successfully.")
    except Exception as e:
        raise RuntimeError(
            f"Could not download MediaPipe model: {e}\n"
            f"Please download manually from:\n  {MODEL_URL}\n"
            f"and place it at:  {MODEL_PATH}"
        )


# ── Landmark index map (same indices as before, now accessed differently) ──────
LANDMARK_IDX = {
    'nose': 0,
    'left_eye_inner': 1, 'left_eye': 2, 'left_eye_outer': 3,
    'right_eye_inner': 4, 'right_eye': 5, 'right_eye_outer': 6,
    'left_ear': 7, 'right_ear': 8,
    'mouth_left': 9, 'mouth_right': 10,
    'left_shoulder': 11, 'right_shoulder': 12,
    'left_elbow': 13, 'right_elbow': 14,
    'left_wrist': 15, 'right_wrist': 16,
    'left_pinky': 17, 'right_pinky': 18,
    'left_index': 19, 'right_index': 20,
    'left_thumb': 21, 'right_thumb': 22,
    'left_hip': 23, 'right_hip': 24,
    'left_knee': 25, 'right_knee': 26,
    'left_ankle': 27, 'right_ankle': 28,
    'left_heel': 29, 'right_heel': 30,
    'left_foot_index': 31, 'right_foot_index': 32,
}

KEY_NAMES = [
    'nose', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
    'left_heel', 'right_heel',
]


class ExerciseType(str, Enum):
    PUSHUP       = "pushup"
    SQUAT        = "squat"
    JUMPING_JACK = "jumping_jack"
    VERTICAL_JUMP = "vertical_jump"
    SITUP        = "situp"
    LUNGE        = "lunge"


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


# ══════════════════════════════════════════════════════════════════════════════
# PoseAnalyzer  — wraps the new MediaPipe Tasks API
# ══════════════════════════════════════════════════════════════════════════════

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
        """Run pose detection on a single BGR frame. Returns raw landmarker result."""
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self):
        self._landmarker.close()

    # ── Keypoint extraction ──────────────────────────────────────────────────

    @staticmethod
    def extract_keypoints(landmarks: list, frame_width: int, frame_height: int) -> dict:
        """
        landmarks: list[NormalizedLandmark]  (result.pose_landmarks[0])
        Returns dict keyed by landmark name with x, y, z, visibility, px, py.
        """
        keypoints = {}
        for name in KEY_NAMES:
            idx = LANDMARK_IDX[name]
            lm  = landmarks[idx]
            keypoints[name] = {
                'x':          float(lm.x),
                'y':          float(lm.y),
                'z':          float(lm.z),
                'visibility': float(getattr(lm, 'visibility', 1.0)),
                'px':         int(lm.x * frame_width),
                'py':         int(lm.y * frame_height),
            }
        return keypoints

    @staticmethod
    def calculate_angle(a: dict, b: dict, c: dict) -> float:
        a_ = np.array([a['x'], a['y']])
        b_ = np.array([b['x'], b['y']])
        c_ = np.array([c['x'], c['y']])
        ba, bc = a_ - b_, c_ - b_
        cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    @staticmethod
    def midpoint(a: dict, b: dict) -> dict:
        return {
            'x': (a['x'] + b['x']) / 2,
            'y': (a['y'] + b['y']) / 2,
            'z': (a['z'] + b['z']) / 2,
            'visibility': min(a['visibility'], b['visibility']),
        }

    @staticmethod
    def check_face_visible(keypoints: dict, min_vis: float = 0.5) -> bool:
        return sum(
            1 for p in ('nose', 'left_ear', 'right_ear')
            if keypoints.get(p, {}).get('visibility', 0) > min_vis
        ) >= 2


# ══════════════════════════════════════════════════════════════════════════════
# Exercise Analyzers
# ══════════════════════════════════════════════════════════════════════════════

class PushupAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.phase     = "up"

    def analyze_frame(self, kp: dict) -> dict:
        issues, score = [], 100.0
        ls, rs = kp.get('left_shoulder'), kp.get('right_shoulder')
        le, re = kp.get('left_elbow'),    kp.get('right_elbow')
        lw, rw = kp.get('left_wrist'),    kp.get('right_wrist')
        lh, rh = kp.get('left_hip'),      kp.get('right_hip')
        lk, rk = kp.get('left_knee'),     kp.get('right_knee')

        if not all([ls, rs, le, re, lw, rw, lh, rh]):
            return {'rep_count': self.rep_count, 'phase': self.phase,
                    'correctness': 0, 'issues': ['Cannot detect upper body'], 'angles': {}}

        la = PoseAnalyzer.calculate_angle(ls, le, lw)
        ra = PoseAnalyzer.calculate_angle(rs, re, rw)
        avg = (la + ra) / 2

        if self.phase == "up"   and avg < 90:   self.phase = "down"
        elif self.phase == "down" and avg > 155: self.phase = "up"; self.rep_count += 1

        if lk and rk:
            mh = PoseAnalyzer.midpoint(lh, rh)
            mk = PoseAnalyzer.midpoint(lk, rk)
            ms = PoseAnalyzer.midpoint(ls, rs)
            if abs(mh['y'] - (ms['y'] + mk['y']) / 2) > 0.06:
                issues.append("Keep body straight — hips misaligned"); score -= 20

        if self.phase == "down" and avg > 115:
            issues.append("Go lower for full range of motion"); score -= 15

        left_flare = PoseAnalyzer.calculate_angle(ls, le, lh)
        if left_flare < 45:
            issues.append("Tuck elbows closer to body"); score -= 10

        return {'rep_count': self.rep_count, 'phase': self.phase,
                'correctness': max(0, score), 'issues': issues,
                'angles': {'left_elbow': round(la, 1), 'right_elbow': round(ra, 1), 'avg': round(avg, 1)}}


class SquatAnalyzer:
    def __init__(self):
        self.rep_count = 0
        self.phase     = "up"

    def analyze_frame(self, kp: dict) -> dict:
        issues, score = [], 100.0
        lh, rh = kp.get('left_hip'),   kp.get('right_hip')
        lk, rk = kp.get('left_knee'),  kp.get('right_knee')
        la, ra = kp.get('left_ankle'), kp.get('right_ankle')
        ls, rs = kp.get('left_shoulder'), kp.get('right_shoulder')

        if not all([lh, rh, lk, rk, la, ra]):
            return {'rep_count': self.rep_count, 'phase': self.phase,
                    'correctness': 0, 'issues': ['Cannot detect lower body'], 'angles': {}}

        lka = PoseAnalyzer.calculate_angle(lh, lk, la)
        rka = PoseAnalyzer.calculate_angle(rh, rk, ra)
        avg = (lka + rka) / 2

        if self.phase == "up"   and avg < 100: self.phase = "down"
        elif self.phase == "down" and avg > 155: self.phase = "up"; self.rep_count += 1

        if lk['px'] > la['px'] + 20: issues.append("Left knee too far forward over toes"); score -= 15
        if rk['px'] < ra['px'] - 20: issues.append("Right knee too far forward over toes"); score -= 15
        if self.phase == "down" and avg > 115: issues.append("Squat deeper for full ROM"); score -= 20

        if ls and rs:
            ms, mh = PoseAnalyzer.midpoint(ls, rs), PoseAnalyzer.midpoint(lh, rh)
            if abs(ms['x'] - mh['x']) > 0.08: issues.append("Keep back more upright"); score -= 10

        return {'rep_count': self.rep_count, 'phase': self.phase,
                'correctness': max(0, score), 'issues': issues,
                'angles': {'left_knee': round(lka, 1), 'right_knee': round(rka, 1), 'avg': round(avg, 1)}}


class JumpAnalyzer:
    def __init__(self):
        self.rep_count          = 0
        self.phase              = "ground"
        self.baseline_hip_y     = None
        self.min_hip_y          = None
        self.jump_heights: list = []
        self.athlete_height_px  = None

    def _calibrate(self, kp: dict, frame_height: int):
        lh, rh = kp.get('left_hip'), kp.get('right_hip')
        if lh and rh:
            self.baseline_hip_y = (lh['y'] + rh['y']) / 2
        la, ra = kp.get('left_ankle'), kp.get('right_ankle')
        ls, rs = kp.get('left_shoulder'), kp.get('right_shoulder')
        if la and ra and ls and rs:
            ankle_y    = (la['py'] + ra['py']) / 2
            shoulder_y = (ls['py'] + rs['py']) / 2
            self.athlete_height_px = abs(ankle_y - shoulder_y) * 1.4

    def analyze_frame(self, kp: dict, frame_height: int) -> dict:
        lh, rh = kp.get('left_hip'), kp.get('right_hip')
        la, ra = kp.get('left_ankle'), kp.get('right_ankle')
        if not all([lh, rh, la, ra]):
            return {'rep_count': self.rep_count, 'phase': self.phase,
                    'correctness': 0, 'issues': ['Cannot detect body'], 'jump_height_cm': None}

        if self.baseline_hip_y is None:
            self._calibrate(kp, frame_height)

        hip_y    = (lh['y'] + rh['y']) / 2
        ankle_y  = (la['y'] + ra['y']) / 2
        airborne = ankle_y < 0.92 or (self.baseline_hip_y and (self.baseline_hip_y - hip_y) > 0.05)

        if self.phase == "ground" and airborne:
            self.phase   = "airborne"
            self.min_hip_y = hip_y
        elif self.phase == "airborne":
            if hip_y < (self.min_hip_y or hip_y):
                self.min_hip_y = hip_y
            if not airborne:
                self.phase = "ground"
                self.rep_count += 1
                if self.baseline_hip_y and self.min_hip_y and self.athlete_height_px:
                    delta = self.baseline_hip_y - self.min_hip_y
                    h_cm  = delta * frame_height / self.athlete_height_px * 175
                    self.jump_heights.append(h_cm)

        avg = round(float(np.mean(self.jump_heights)), 1) if self.jump_heights else None
        return {'rep_count': self.rep_count, 'phase': self.phase,
                'correctness': 100.0, 'issues': [],
                'jump_height_cm': avg,
                'all_jump_heights': [round(h, 1) for h in self.jump_heights]}


# ══════════════════════════════════════════════════════════════════════════════
# Cheat Detector
# ══════════════════════════════════════════════════════════════════════════════

class CheatDetector:
    def __init__(self):
        self.face_vis_history  = []
        self.conf_history      = []
        self.speed_history     = []
        self._prev_kp          = None

    def analyze(self, kp: dict, face_visible: bool, avg_conf: float):
        self.face_vis_history.append(face_visible)
        self.conf_history.append(avg_conf)
        if self._prev_kp and kp.get('left_hip') and self._prev_kp.get('left_hip'):
            self.speed_history.append(abs(kp['left_hip']['y'] - self._prev_kp['left_hip']['y']))
        self._prev_kp = kp

    def report(self) -> dict:
        reasons, detected = [], False
        if len(self.face_vis_history) > 10:
            rate = sum(self.face_vis_history) / len(self.face_vis_history)
            if rate < 0.5:
                reasons.append(f"Face hidden in {round((1-rate)*100)}% of frames"); detected = True
        if len(self.conf_history) > 10:
            arr   = np.array(self.conf_history)
            drops = np.sum(np.diff(arr) < -0.3)
            if np.mean(arr) < 0.4:
                reasons.append("Very low pose confidence throughout video"); detected = True
            if drops > len(arr) * 0.05:
                reasons.append("Sudden confidence drops — possible frame cuts"); detected = True
        if len(self.speed_history) > 10:
            if np.sum(np.array(self.speed_history) > 0.15) > 3:
                reasons.append("Abnormally fast movement in some frames"); detected = True
        return {'detected': detected, 'reasons': reasons}


# ══════════════════════════════════════════════════════════════════════════════
# VideoProcessor  — main pipeline
# ══════════════════════════════════════════════════════════════════════════════

class VideoProcessor:

    def get_exercise_analyzer(self, exercise_type: str):
        et = ExerciseType(exercise_type)
        if et == ExerciseType.PUSHUP:                            return PushupAnalyzer()
        if et == ExerciseType.SQUAT:                             return SquatAnalyzer()
        if et in (ExerciseType.VERTICAL_JUMP, ExerciseType.JUMPING_JACK): return JumpAnalyzer()
        return SquatAnalyzer()

    def process_video(self, video_path: str, exercise_type: str,
                      progress_callback=None) -> VideoAnalysisResult:

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fw           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration     = total_frames / fps

        analyzer      = self.get_exercise_analyzer(exercise_type)
        cheat         = CheatDetector()
        pose          = PoseAnalyzer()           # one PoseLandmarker instance
        frame_analyses, correctness_scores = [], []

        sample_every = max(1, int(fps / 10))     # analyse ~10 fps
        frame_number = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_number += 1
                if frame_number % sample_every != 0:
                    continue

                timestamp_ms = int((frame_number / fps) * 1000)
                result       = pose.process_frame(frame, timestamp_ms)

                if not result.pose_landmarks:
                    continue

                landmarks = result.pose_landmarks[0]       # first (only) pose
                kp        = PoseAnalyzer.extract_keypoints(landmarks, fw, fh)
                face_vis  = PoseAnalyzer.check_face_visible(kp)
                avg_conf  = float(np.mean([k['visibility'] for k in kp.values()]))

                if exercise_type == ExerciseType.VERTICAL_JUMP:
                    analysis = analyzer.analyze_frame(kp, fh)
                else:
                    analysis = analyzer.analyze_frame(kp)

                cheat.analyze(kp, face_vis, avg_conf)
                correctness_scores.append(analysis.get('correctness', 0))

                ts = frame_number / fps
                frame_analyses.append(asdict(FrameAnalysis(
                    frame_number=frame_number,
                    timestamp=round(ts, 2),
                    keypoints={
                        name: {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in vals.items() if k in ('x', 'y', 'visibility')}
                        for name, vals in kp.items()
                    },
                    rep_count=analysis.get('rep_count', 0),
                    phase=analysis.get('phase', 'unknown'),
                    correctness_score=round(analysis.get('correctness', 0), 1),
                    issues=analysis.get('issues', []),
                    face_visible=face_vis,
                    confidence=round(avg_conf, 3),
                )))

                if progress_callback:
                    progress_callback(frame_number / total_frames * 100)
        finally:
            cap.release()
            pose.close()

        cheat_report    = cheat.report()
        total_reps      = analyzer.rep_count
        avg_correctness = float(np.mean(correctness_scores)) if correctness_scores else 0.0
        jump_height     = None
        if hasattr(analyzer, 'jump_heights') and analyzer.jump_heights:
            jump_height = round(max(analyzer.jump_heights), 1)

        summary = {
            'total_reps':       total_reps,
            'avg_correctness_score': round(avg_correctness, 1),
            'duration_seconds': round(duration, 1),
            'reps_per_minute':  round(total_reps / (duration / 60), 1) if duration > 0 else 0,
            'exercise_type':    exercise_type,
            'video_fps':        round(fps, 1),
            'frames_analyzed':  len(frame_analyses),
        }
        perf = self._performance_metrics(exercise_type, total_reps, avg_correctness, jump_height)

        return VideoAnalysisResult(
            exercise_type=exercise_type,
            total_frames=total_frames,
            fps=fps,
            duration=duration,
            total_reps=total_reps,
            avg_correctness_score=round(avg_correctness, 1),
            jump_height_cm=jump_height,
            cheat_detected=cheat_report['detected'],
            cheat_reasons=cheat_report['reasons'],
            frame_analyses=frame_analyses,
            summary=summary,
            performance_metrics=perf,
        )

    @staticmethod
    def _performance_metrics(exercise_type, reps, correctness, jump_height):
        bm = {
            'pushup':        {'beginner': 10, 'intermediate': 25, 'advanced': 40},
            'squat':         {'beginner': 15, 'intermediate': 30, 'advanced': 50},
            'jumping_jack':  {'beginner': 20, 'intermediate': 50, 'advanced': 80},
            'vertical_jump': {'beginner': 25, 'intermediate': 45, 'advanced': 65},
            'situp':         {'beginner': 15, 'intermediate': 30, 'advanced': 50},
        }.get(exercise_type, {'beginner': 10, 'intermediate': 25, 'advanced': 40})

        metric = jump_height if (exercise_type == 'vertical_jump' and jump_height) else reps
        if   metric >= bm['advanced']:     level, pct = 'Advanced',     90
        elif metric >= bm['intermediate']: level, pct = 'Intermediate',  60
        elif metric >= bm['beginner']:     level, pct = 'Beginner',      30
        else:                              level, pct = 'Needs Work',    10

        grade = 'A' if correctness >= 90 else 'B' if correctness >= 80 else \
                'C' if correctness >= 70 else 'D' if correctness >= 60 else 'F'

        return {
            'fitness_level':        level,
            'estimated_percentile': pct,
            'form_grade':           grade,
            'benchmarks':           bm,
            'metric_value':         metric,
            'metric_unit':          'cm' if exercise_type == 'vertical_jump' else 'reps',
        }


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        result = VideoProcessor().process_video(sys.argv[1], sys.argv[2])
        print(json.dumps(result.__dict__, indent=2, default=str))
    else:
        print("Usage: python pose_analyzer.py <video_path> <exercise_type>")
        print("Exercise types: pushup, squat, vertical_jump, jumping_jack, situp, lunge")