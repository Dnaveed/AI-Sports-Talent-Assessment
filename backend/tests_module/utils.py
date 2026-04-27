"""Test utilities and status computation."""
from datetime import datetime, timedelta


def compute_test_status(test: dict) -> str:
    """Compute actual test status based on scheduled date/time."""
    # Archived/deleted tests are always completed
    if test.get("is_archived"):
        return "completed"

    try:
        scheduled_date = test.get("scheduled_date")
        start_time = test.get("start_time", "00:00")
        duration_minutes = test.get("duration_minutes", 60)

        if not scheduled_date:
            return test.get("status", "upcoming")

        dt_str = f"{scheduled_date} {start_time}"
        scheduled_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        end_dt = scheduled_dt + timedelta(minutes=int(duration_minutes or 60))
        now = datetime.now()  # local time — matches what the user entered in the form

        if now < scheduled_dt:
            return "upcoming"
        elif scheduled_dt <= now <= end_dt:
            return "active"
        else:
            return "completed"
    except Exception:
        return test.get("status", "upcoming")


def compute_test_score(result: dict, exercises: list) -> float:
    """Compute the leaderboard score on the same 0-100 form scale shown in results."""
    del exercises  # Kept for the existing call signature used by routes.

    score = result.get("hybrid_form_score", result.get("avg_correctness_score", 0))
    try:
        score = float(score or 0)
    except (TypeError, ValueError):
        score = 0.0

    return round(max(0.0, min(100.0, score)), 1)
