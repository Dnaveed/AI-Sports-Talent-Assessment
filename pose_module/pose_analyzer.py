"""
Pose Analysis Module - orchestrates per-exercise analyzers with:
- landmark smoothing and visibility gating
- per-rep quality breakdown
- hybrid scoring (rule score + ML-style proxy score)
- confidence and detailed feedback payloads
"""

import cv2
import numpy as np
import json
from dataclasses import asdict
from typing import Dict, List, Tuple

from .base import (
    ExerciseType, FrameAnalysis, VideoAnalysisResult,
    PoseAnalyzer, LandmarkSmoother, CheatDetector,
)
from .pushup import PushupAnalyzer
from .squat import SquatAnalyzer
from .situp import SitupAnalyzer
from .lunge import LungeAnalyzer
from .jumping_jack import JumpingJackAnalyzer
from .vertical_jump import VerticalJumpAnalyzer


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
        if exercise_type == ExerciseType.SITUP:
            return 0.25, 0.70
        return 0.35, 0.50

    def _exercise_joints_visible(self, keypoints: dict, exercise_type: str, min_vis: float = 0.45) -> bool:
        if exercise_type == ExerciseType.SITUP:
            left_chain = ("left_shoulder", "left_hip", "left_knee")
            right_chain = ("right_shoulder", "right_hip", "right_knee")
            left_ok = sum(1 for n in left_chain if keypoints.get(n, {}).get("visibility", 0) >= min_vis) >= 2
            right_ok = sum(1 for n in right_chain if keypoints.get(n, {}).get("visibility", 0) >= min_vis) >= 2
            core_ok = sum(
                1 for n in ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
                if keypoints.get(n, {}).get("visibility", 0) >= min_vis
            ) >= 2
            return core_ok and (left_ok or right_ok)

        required = self._required_joints_for_exercise(exercise_type)
        if not required:
            return True
        visible = sum(1 for n in required if keypoints.get(n, {}).get("visibility", 0) >= min_vis)
        return (visible / len(required)) >= 0.75

    def get_exercise_analyzer(self, exercise_type: str):
        et = ExerciseType(exercise_type)
        return {
            ExerciseType.PUSHUP: PushupAnalyzer,
            ExerciseType.SQUAT: SquatAnalyzer,
            ExerciseType.SITUP: SitupAnalyzer,
            ExerciseType.LUNGE: LungeAnalyzer,
            ExerciseType.JUMPING_JACK: JumpingJackAnalyzer,
            ExerciseType.VERTICAL_JUMP: VerticalJumpAnalyzer,
        }.get(et, SquatAnalyzer)()

    @staticmethod
    def _mean_or_zero(vals: List[float]) -> float:
        return float(np.mean(vals)) if vals else 0.0

    def _ml_proxy_score(self, usable_rate: float, conf_vals: List[float],
                        frame_scores: List[float], rep_breakdown: List[dict]) -> float:
        conf_avg = self._mean_or_zero(conf_vals)
        frame_std = float(np.std(frame_scores)) if frame_scores else 30.0
        rep_scores = [r.get("quality_score", 0.0) for r in rep_breakdown]
        rep_std = float(np.std(rep_scores)) if rep_scores else 25.0
        rep_avg = self._mean_or_zero(rep_scores)
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
                    analysis = {"rep_count": analyzer.rep_count,
                                "phase": getattr(analyzer, "phase", "unknown"),
                                "correctness": 0.0, "issues": [issue]}
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
                    asdict(FrameAnalysis(
                        frame_number=frame_number,
                        timestamp=round(ts, 2),
                        keypoints={
                            name: {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in vals.items() if k in ("x", "y", "visibility")}
                            for name, vals in kp.items()
                        },
                        rep_count=analysis.get("rep_count", 0),
                        phase=analysis.get("phase", "unknown"),
                        correctness_score=round(score, 1),
                        issues=analysis.get("issues", []),
                        face_visible=face_vis,
                        confidence=round(avg_conf, 3),
                    ))
                )

                if progress_callback and total_frames > 0:
                    progress_callback(frame_number / max(1, total_frames) * 100)
        finally:
            cap.release()
            pose.close()

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
        confidence_score = max(0.0, min(100.0,
            100.0 * self._mean_or_zero(conf_vals) * usable_rate * (1.0 - min(1.0, disagreement / 100.0))
        ))

        invalid_reasons = []
        no_rep_ex = exercise_type in (
            ExerciseType.PUSHUP, ExerciseType.SQUAT, ExerciseType.JUMPING_JACK,
            ExerciseType.SITUP, ExerciseType.LUNGE, ExerciseType.VERTICAL_JUMP,
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

        grade = "A" if correctness >= 90 else "B" if correctness >= 80 else "C" if correctness >= 70 else "D" if correctness >= 60 else "F"

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
