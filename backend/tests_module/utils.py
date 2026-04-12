"""Test utilities and status computation."""
from datetime import datetime, timedelta
from config import BENCHMARKS


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
    """Compute test score from result."""
    ex_type = result.get("exercise_type", "")
    bm = BENCHMARKS.get(ex_type, {})
    adv = bm.get("advanced", 1) or 1
    raw_val = result.get("jump_height_cm") if ex_type == "vertical_jump" else result.get("total_reps", 0)
    raw_val = raw_val or 0
    rep_score = min(100.0, (raw_val / adv) * 100)
    form = result.get("avg_correctness_score") or 70
    final = rep_score * (form / 100)
    if result.get("cheat_detected"):
        final *= 0.5
    return round(final, 1)
