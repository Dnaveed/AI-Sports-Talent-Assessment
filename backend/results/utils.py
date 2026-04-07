"""Results utilities - serialization, sanitization, and analysis."""
from datetime import datetime, timedelta
from typing import Optional, List


def serialize(doc: dict) -> dict:
    """Convert MongoDB document to JSON-serializable format."""
    if doc is None:
        return None
    out = {}
    for k, v in doc.items():
        key = "id" if k == "_id" else k
        if isinstance(v, datetime):
            out[key] = v.isoformat()
        elif isinstance(v, list):
            out[key] = [serialize(i) if isinstance(i, dict) else i for i in v]
        elif isinstance(v, dict):
            out[key] = serialize(v)
        else:
            out[key] = str(v) if k == "_id" else v
    return out


def sanitize_analysis_doc(doc: dict) -> dict:
    """Clamp/normalize analysis numerics to prevent corrupt values."""
    if not doc:
        return doc

    out = dict(doc)

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    # Duration validation
    duration = safe_float(out.get("duration_seconds", 0.0), 0.0)
    if duration < 0 or duration > 10 * 60 * 60:
        duration = 0.0
    out["duration_seconds"] = round(duration, 1)

    # Reps validation
    total_reps = safe_float(out.get("total_reps", 0.0), 0.0)
    out["total_reps"] = int(max(0, min(total_reps, 10000)))

    # Score fields validation (0-100)
    for key in ("avg_correctness_score", "rule_score", "ml_score", "hybrid_form_score", "confidence_score"):
        if key in out:
            out[key] = round(max(0.0, min(100.0, safe_float(out.get(key, 0.0), 0.0))), 1)

    return out


def _compute_badges(results: List[dict]) -> List[dict]:
    """Compute achievement badges based on results."""
    badges = []
    total = len(results)
    
    if total >= 5:
        badges.append({"id": "starter", "title": "Getting Started", "description": "Completed 5 or more tests."})
    if total >= 10:
        badges.append({"id": "consistent", "title": "Consistency", "description": "Completed 10 or more tests."})
    
    avg_score = round(sum(float(r.get("avg_correctness_score", 0) or 0) for r in results) / total, 1) if total else 0.0
    if avg_score >= 75:
        badges.append({"id": "form_focus", "title": "Form Focus", "description": f"Average form score reached {avg_score:.1f}% or higher."})
    if avg_score >= 85:
        badges.append({"id": "elite", "title": "Elite Form", "description": f"Average form score reached {avg_score:.1f}% or higher."})
    
    exercise_counts = {}
    for r in results:
        ex = r.get("exercise_type", "unknown")
        exercise_counts[ex] = exercise_counts.get(ex, 0) + 1
    
    for ex, count in exercise_counts.items():
        if count >= 5:
            badges.append({
                "id": f"{ex}-focus",
                "title": f"{ex.replace('_', ' ').title()} Specialist",
                "description": f"Logged 5 or more {ex.replace('_', ' ')} sessions."
            })
    
    return badges


def _compute_progress_summary(results: List[dict], user_doc: Optional[dict] = None) -> dict:
    """Compute comprehensive progress summary for athlete."""
    ordered = sorted(results, key=lambda d: d.get("created_at") or "", reverse=True)
    total_tests = len(ordered)
    avg_score = round(sum(float(r.get("avg_correctness_score", 0) or 0) for r in ordered) / total_tests, 1) if total_tests else 0.0
    best_score = round(max((float(r.get("avg_correctness_score", 0) or 0) for r in ordered), default=0.0), 1)
    best_reps = max((int(r.get("total_reps", 0) or 0) for r in ordered), default=0)

    exercise_best = {}
    for r in ordered:
        ex = r.get("exercise_type", "unknown")
        score = float(r.get("avg_correctness_score", 0) or 0)
        if ex not in exercise_best or score > exercise_best[ex]["score"]:
            exercise_best[ex] = {
                "exercise_type": ex,
                "score": round(score, 1),
                "reps": int(r.get("total_reps", 0) or 0),
                "date": r.get("created_at"),
            }

    trend = [
        {
            "date": r.get("created_at"),
            "exercise_type": r.get("exercise_type"),
            "score": float(r.get("avg_correctness_score", 0) or 0),
            "reps": int(r.get("total_reps", 0) or 0),
            "fitness_level": r.get("fitness_level", "Unknown"),
        }
        for r in ordered[:10]
    ]

    recent_window = ordered[:10]
    previous_window = ordered[10:20]
    recent_avg = round(sum(float(r.get("avg_correctness_score", 0) or 0) for r in recent_window) / len(recent_window), 1) if recent_window else 0.0
    previous_avg = round(sum(float(r.get("avg_correctness_score", 0) or 0) for r in previous_window) / len(previous_window), 1) if previous_window else recent_avg
    delta = round(recent_avg - previous_avg, 1) if previous_window else 0.0

    goals = {
        "goal_avg_score": user_doc.get("goal_avg_score") if user_doc else None,
        "goal_tests_per_week": user_doc.get("goal_tests_per_week") if user_doc else None,
        "goal_primary_exercise": user_doc.get("goal_primary_exercise") if user_doc else None,
    }
    goal_progress = None
    if goals.get("goal_avg_score"):
        goal_progress = round(min(100.0, (avg_score / float(goals["goal_avg_score"])) * 100), 1) if goals["goal_avg_score"] else None

    badges = _compute_badges(ordered)

    return {
        "summary": {
            "total_tests": total_tests,
            "average_score": avg_score,
            "best_score": best_score,
            "best_reps": best_reps,
            "score_delta": delta,
            "recent_average": recent_avg,
        },
        "goals": goals,
        "goal_progress": goal_progress,
        "bests": list(exercise_best.values()),
        "trend": trend,
        "badges": badges,
    }


def _build_notifications(results: List[dict], registrations: List[dict], tests: List[dict]) -> List[dict]:
    """Build notification messages for athlete."""
    notices = []
    now = datetime.now()

    if results:
        latest = sorted(results, key=lambda d: d.get("created_at") or "", reverse=True)[0]
        created_at = latest.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at)
            except ValueError:
                created_at = None
        if isinstance(created_at, datetime) and (now - created_at) <= timedelta(hours=24):
            notices.append({
                "type": "result_ready",
                "title": "Latest result ready",
                "message": f"Your {latest.get('exercise_type', 'recent')} result is available.",
                "timestamp": created_at.isoformat() if created_at else None
            })

    test_map = {str(t.get("_id")): t for t in tests}
    for reg in registrations:
        test = test_map.get(reg.get("test_id"))
        if not test:
            continue
        scheduled_date = test.get("scheduled_date")
        start_time = test.get("start_time", "00:00")
        try:
            scheduled_dt = datetime.strptime(f"{scheduled_date} {start_time}", "%Y-%m-%d %H:%M")
        except Exception:
            continue
        hours_until = (scheduled_dt - now).total_seconds() / 3600
        if 0 <= hours_until <= 24:
            notices.append({
                "type": "reminder",
                "title": f"Assessment soon: {test.get('name', 'Scheduled test')}",
                "message": f"Starts in {max(1, int(hours_until))} hour{'s' if hours_until >= 2 else ''}.",
                "timestamp": scheduled_dt.isoformat(),
            })

    for badge in _compute_badges(results):
        notices.append({
            "type": "milestone",
            "title": f"Badge unlocked: {badge['title']}",
            "message": badge["description"],
            "timestamp": now.isoformat(),
        })

    return notices[:10]
