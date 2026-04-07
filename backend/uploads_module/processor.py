"""Video processing logic."""
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from database import get_db
from config import SCORING_MODE


async def process_video_job(jid, sid, vpath, exercise_type, user_id):
    """Process uploaded video and generate analysis."""
    db = get_db()

    async def prog(p):
        await db.processing_jobs.update_one(
            {"_id": jid},
            {"$set": {"progress": round(p, 1), "updated_at": datetime.utcnow()}}
        )

    await db.processing_jobs.update_one({"_id": jid}, {"$set": {"status": "processing"}})

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "pose_module"))
        from pose_analyzer import VideoProcessor
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: VideoProcessor().process_video(vpath, exercise_type))
        rd = result.__dict__
        
        session_doc = await db.test_sessions.find_one({"_id": sid}, {"live_pose_input": 1})
        live_pose_input = (session_doc or {}).get("live_pose_input") or {}

        await prog(90)
        rid = str(uuid4())
        pm = rd.get("performance_metrics", {})

        # Build athlete-specific baseline
        recent_same = await db.analysis_results.find(
            {"user_id": user_id, "exercise_type": exercise_type},
            {"avg_correctness_score": 1, "total_reps": 1, "jump_height_cm": 1, "created_at": 1}
        ).sort("created_at", -1).limit(12).to_list(12)

        baseline_form = round(
            sum(d.get("avg_correctness_score", 0) for d in recent_same) / len(recent_same), 1
        ) if recent_same else None
        baseline_reps = round(
            sum(d.get("total_reps", 0) for d in recent_same) / len(recent_same), 1
        ) if recent_same else None
        
        jump_vals = [d.get("jump_height_cm") for d in recent_same if d.get("jump_height_cm") is not None]
        baseline_jump = round(sum(jump_vals) / len(jump_vals), 1) if jump_vals else None

        rule_score_val = rd.get("rule_score", rd.get("avg_correctness_score", 0))
        hybrid_score_val = rd.get("hybrid_form_score") or rd.get("avg_correctness_score", 0)

        # Scoring mode selection
        if SCORING_MODE == "shadow":
            current_form = rule_score_val
        elif SCORING_MODE == "rule_only":
            current_form = rule_score_val
        else:
            current_form = hybrid_score_val

        form_delta = round(current_form - baseline_form, 1) if baseline_form is not None else None
        personalized_form_index = round(max(0.0, min(100.0, current_form + (form_delta or 0) * 0.2)), 1)

        detailed_feedback = rd.get("detailed_feedback", {})
        top_faults = detailed_feedback.get("top_faults", []) if isinstance(detailed_feedback, dict) else []

        try:
            duration_seconds_val = float(rd.get("duration", 0) or 0)
        except (TypeError, ValueError):
            duration_seconds_val = 0.0
        if duration_seconds_val < 0 or duration_seconds_val > 10 * 60 * 60:
            duration_seconds_val = 0.0

        # Insert analysis result
        await db.analysis_results.insert_one({
            "_id": rid,
            "session_id": sid,
            "user_id": user_id,
            "exercise_type": exercise_type,
            "total_reps": rd.get("total_reps", 0),
            "avg_correctness_score": current_form,
            "jump_height_cm": rd.get("jump_height_cm"),
            "duration_seconds": round(duration_seconds_val, 1),
            "reps_per_minute": rd.get("summary", {}).get("reps_per_minute", 0),
            "fitness_level": pm.get("fitness_level", "Unknown"),
            "form_grade": pm.get("form_grade", "N/A"),
            "estimated_percentile": pm.get("estimated_percentile", 0),
            "rule_score": rule_score_val,
            "ml_score": rd.get("ml_score", rd.get("avg_correctness_score", 0)),
            "hybrid_form_score": hybrid_score_val,
            "confidence_score": rd.get("confidence_score", 0),
            "analysis_version": rd.get("analysis_version", "v2"),
            "scoring_mode": SCORING_MODE,
            "shadow_scores": {
                "rule_score": rule_score_val,
                "hybrid_score": hybrid_score_val,
            },
            "cheat_detected": bool(rd.get("cheat_detected", False)),
            "cheat_reasons": rd.get("cheat_reasons", []),
            "frame_analyses": rd.get("frame_analyses", [])[:100],
            "rep_breakdown": rd.get("rep_breakdown", [])[:60],
            "detailed_feedback": detailed_feedback,
            "top_faults": top_faults,
            "personalization": {
                "baseline_form_score": baseline_form,
                "baseline_reps": baseline_reps,
                "baseline_jump_height_cm": baseline_jump,
                "current_form_delta": form_delta,
                "personalized_form_index": personalized_form_index,
                "history_window": len(recent_same),
            },
            "performance_metrics": pm,
            "live_pose_input": {
                "total_reps": int(live_pose_input.get("total_reps") or 0),
                "valid_reps": int(live_pose_input.get("valid_reps") or 0),
                "form_accuracy": live_pose_input.get("form_accuracy"),
                "feedback": live_pose_input.get("feedback", ""),
            },
            "live_vs_backend": {
                "rep_delta": int(rd.get("total_reps", 0) or 0) - int(live_pose_input.get("total_reps") or 0),
                "valid_rep_delta": int(rd.get("total_reps", 0) or 0) - int(live_pose_input.get("valid_reps") or 0),
                "form_delta": (
                    round(float(current_form) - float(live_pose_input.get("form_accuracy")), 1)
                    if live_pose_input.get("form_accuracy") is not None else None
                ),
            },
            "created_at": datetime.utcnow(),
        })

        await db.test_sessions.update_one(
            {"_id": sid},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}}
        )
        await db.processing_jobs.update_one(
            {"_id": jid},
            {"$set": {"status": "completed", "progress": 100.0, "updated_at": datetime.utcnow()}}
        )

    except Exception as e:
        await db.processing_jobs.update_one(
            {"_id": jid},
            {"$set": {"status": "failed", "error_message": str(e), "updated_at": datetime.utcnow()}}
        )
        await db.test_sessions.update_one({"_id": sid}, {"$set": {"status": "failed"}})
