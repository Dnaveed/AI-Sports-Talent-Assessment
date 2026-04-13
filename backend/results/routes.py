"""Results routes - retrieve and export analysis results."""
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import StreamingResponse
from datetime import datetime
import csv
import io
from typing import Optional

from database import get_db
from dependencies import get_current_user
from results.utils import serialize, sanitize_analysis_doc, _compute_progress_summary, _build_notifications

router = APIRouter(prefix="/api", tags=["results"])


@router.get("/results")
async def my_results(user=Depends(get_current_user)):
    """Get user's analysis results."""
    db = get_db()
    docs = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).limit(50).to_list(50)
    return [serialize(sanitize_analysis_doc(d)) for d in docs]


@router.get("/results/export")
async def export_results(
    format: str = Query("csv", pattern="^(csv|pdf)$"),
    exercise_type: Optional[str] = None,
    fitness_level: Optional[str] = None,
    user=Depends(get_current_user),
):
    """Export athlete results as CSV or plain text."""
    db = get_db()
    query: dict = {"user_id": user["user_id"]}
    if exercise_type:
        query["exercise_type"] = exercise_type
    if fitness_level:
        query["fitness_level"] = fitness_level

    docs = await db.analysis_results.find(query, {"frame_analyses": 0}).sort("created_at", -1).to_list(500)
    results = [sanitize_analysis_doc(d) for d in docs]

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "Date", "Exercise", "Reps", "Form Score", "Grade",
            "Level", "Duration (s)", "Reps/Min", "Cheat Detected",
            "Rule Score", "ML Score", "Hybrid Score", "Confidence",
        ])
        for r in results:
            created = r.get("created_at")
            if isinstance(created, datetime):
                created = created.isoformat()
            writer.writerow([
                created or "",
                r.get("exercise_type", ""),
                r.get("total_reps", 0),
                r.get("avg_correctness_score", 0),
                r.get("form_grade", ""),
                r.get("fitness_level", ""),
                r.get("duration_seconds", 0),
                r.get("reps_per_minute", 0),
                "Yes" if r.get("cheat_detected") else "No",
                r.get("rule_score", 0),
                r.get("ml_score", 0),
                r.get("hybrid_form_score", 0),
                r.get("confidence_score", 0),
            ])
        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=athlete_results.csv"},
        )

    lines = ["AthleteAI — My Results Export", "=" * 40, ""]
    for r in results:
        created = r.get("created_at")
        if isinstance(created, datetime):
            created = created.isoformat()
        lines.append(
            f"{created or 'N/A'} | {r.get('exercise_type','')} | "
            f"Reps: {r.get('total_reps',0)} | Score: {r.get('avg_correctness_score',0):.1f}% | "
            f"Grade: {r.get('form_grade','')} | Level: {r.get('fitness_level','')}"
        )
    content = "\n".join(lines)
    return StreamingResponse(
        iter([content]),
        media_type="text/plain",
        headers={"Content-Disposition": "attachment; filename=athlete_results.txt"},
    )


@router.get("/results/{result_id}")
async def get_result(result_id: str, user=Depends(get_current_user)):
    """Get specific analysis result."""
    db = get_db()
    doc = await db.analysis_results.find_one({"_id": result_id})
    if not doc:
        raise HTTPException(404, "Not found")
    if doc["user_id"] != user["user_id"] and user.get("role") != "admin":
        raise HTTPException(403, "Access denied")
    return serialize(sanitize_analysis_doc(doc))


@router.get("/results/{result_id}/exercise-metrics")
async def result_exercise_metrics(result_id: str, user=Depends(get_current_user)):
    """Get detailed metrics for a result."""
    db = get_db()
    result = await db.analysis_results.find_one({"_id": result_id})
    if not result:
        raise HTTPException(404, "Not found")
    if user.get("role") == "athlete" and result.get("user_id") != user["user_id"]:
        raise HTTPException(403, "Access denied")

    return {
        "exercise_type": result.get("exercise_type"),
        "total_reps": result.get("total_reps", 0),
        "avg_correctness_score": result.get("avg_correctness_score", 0),
        "form_grade": result.get("form_grade", "N/A"),
        "fitness_level": result.get("fitness_level", "Unknown"),
        "jump_height_cm": result.get("jump_height_cm"),
        "duration_seconds": result.get("duration_seconds", 0),
        "reps_per_minute": result.get("reps_per_minute", 0),
        "cheat_detected": result.get("cheat_detected", False),
        "rule_score": result.get("rule_score", 0),
        "ml_score": result.get("ml_score", 0),
        "hybrid_form_score": result.get("hybrid_form_score", 0),
        "confidence_score": result.get("confidence_score", 0),
        "estimated_percentile": result.get("estimated_percentile", 0),
        "created_at": result.get("created_at").isoformat() if isinstance(result.get("created_at"), datetime) else "",
    }


@router.get("/sessions/{session_id}/result")
async def session_result(session_id: str, user=Depends(get_current_user)):
    """Get result for a specific session."""
    db = get_db()
    doc = await db.analysis_results.find_one({"session_id": session_id})
    if not doc:
        raise HTTPException(404, "No result yet")
    return serialize(sanitize_analysis_doc(doc))


@router.get("/progress")
async def athlete_progress(user=Depends(get_current_user)):
    """Get athlete progress summary."""
    db = get_db()
    results = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).to_list(200)
    user_doc = await db.users.find_one({"_id": user["user_id"]}, {"password_hash": 0})
    summary = _compute_progress_summary([sanitize_analysis_doc(d) for d in results], user_doc)
    return summary


@router.get("/notifications")
async def athlete_notifications(user=Depends(get_current_user)):
    """Get athlete notifications."""
    db = get_db()
    results = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).to_list(200)
    registrations = await db.test_registrations.find({"user_id": user["user_id"]}).to_list(200)
    test_ids = [r.get("test_id") for r in registrations if r.get("test_id")]
    tests = await db.tests.find({"_id": {"$in": test_ids}}).to_list(200) if test_ids else []
    notifications = _build_notifications([sanitize_analysis_doc(d) for d in results], registrations, tests)
    return {"items": notifications, "count": len(notifications)}
