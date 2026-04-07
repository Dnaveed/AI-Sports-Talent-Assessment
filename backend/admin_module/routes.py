"""Admin routes - system dashboard and management."""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
from typing import Optional

from database import get_db
from dependencies import require_admin
from results.utils import serialize, sanitize_analysis_doc

router = APIRouter(prefix="/api/admin", tags=["admin"])


@router.get("/stats")
async def admin_stats(admin=Depends(require_admin)):
    """Get admin dashboard stats."""
    db = get_db()
    
    total_athletes = await db.users.count_documents({"role": "athlete"})
    total_authorities = await db.users.count_documents({"role": "authority"})
    total_sessions = await db.test_sessions.count_documents({})
    completed = await db.test_sessions.count_documents({"status": "completed"})
    cheat_flags = await db.analysis_results.count_documents({"cheat_detected": True})

    avg_r = await db.analysis_results.aggregate(
        [{"$group": {"_id": None, "avg": {"$avg": "$avg_correctness_score"}}}]
    ).to_list(1)
    avg_score = round(avg_r[0]["avg"], 1) if avg_r else 0.0

    ex_breakdown = await db.analysis_results.aggregate([
        {"$group": {"_id": "$exercise_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]).to_list(20)

    recent_docs = await db.analysis_results.find(
        {}, {"user_id": 1, "exercise_type": 1, "avg_correctness_score": 1, "fitness_level": 1, "created_at": 1}
    ).sort("created_at", -1).limit(10).to_list(10)

    uid_list = list({d["user_id"] for d in recent_docs})
    umap = {str(u["_id"]): u["name"] for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1}).to_list(50)}

    recent = [
        {
            "name": umap.get(d["user_id"], "Unknown"),
            "exercise_type": d["exercise_type"],
            "avg_correctness_score": d.get("avg_correctness_score", 0),
            "fitness_level": d.get("fitness_level", "—"),
            "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else ""
        }
        for d in recent_docs
    ]

    return {
        "total_athletes": total_athletes,
        "total_authorities": total_authorities,
        "total_sessions": total_sessions,
        "completed_sessions": completed,
        "avg_correctness_score": avg_score,
        "cheat_flags": cheat_flags,
        "exercise_breakdown": [{"exercise_type": d["_id"], "count": d["count"]} for d in ex_breakdown],
        "recent_activity": recent
    }


@router.get("/athletes")
async def admin_athletes(
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    fitness_level: Optional[str] = None,
    exercise_type: Optional[str] = None,
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    admin=Depends(require_admin),
):
    """Get athlete list with stats."""
    db = get_db()
    
    uf = {"role": "athlete"}
    if age_min:
        uf.setdefault("age", {})["$gte"] = age_min
    if age_max:
        uf.setdefault("age", {})["$lte"] = age_max

    users = await db.users.find(uf, {"password_hash": 0}).to_list(500)

    rf = {}
    if fitness_level:
        rf["fitness_level"] = fitness_level
    if exercise_type:
        rf["exercise_type"] = exercise_type

    stats = await db.analysis_results.aggregate([
        {"$match": rf},
        {"$group": {
            "_id": "$user_id",
            "total_tests": {"$sum": 1},
            "avg_score": {"$avg": "$avg_correctness_score"},
            "last_test": {"$max": "$created_at"}
        }}
    ]).to_list(500)

    smap = {d["_id"]: d for d in stats}
    rows = []
    search = q.strip().lower() if q else ""

    for u in users:
        stats_doc = smap.get(str(u["_id"]), {})
        row = {
            "id": str(u["_id"]),
            "name": u["name"],
            "email": u["email"],
            "age": u.get("age"),
            "height_cm": u.get("height_cm"),
            "weight_kg": u.get("weight_kg"),
            "total_tests": stats_doc.get("total_tests", 0),
            "avg_score": round(stats_doc.get("avg_score", 0), 1) if stats_doc.get("avg_score") is not None else None,
            "last_test": stats_doc.get("last_test").isoformat() if isinstance(stats_doc.get("last_test"), datetime) else None
        }
        if search and search not in f"{row['name']} {row['email']}".lower():
            continue
        rows.append(row)

    reverse = sort_dir.lower() != "asc"
    sort_key = sort_by or "name"

    def athlete_sort_key(item):
        if sort_key in {"age", "height_cm", "weight_kg", "total_tests"}:
            return int(item.get(sort_key) or 0)
        if sort_key == "avg_score":
            return float(item.get(sort_key) or 0)
        if sort_key == "last_test":
            return item.get(sort_key) or ""
        return str(item.get(sort_key) or "").lower()

    rows.sort(key=athlete_sort_key, reverse=reverse)
    return rows


@router.get("/authorities")
async def admin_authorities(
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    admin=Depends(require_admin),
):
    """Get authority list with stats."""
    db = get_db()
    
    authorities = await db.users.find({"role": "authority"}, {"password_hash": 0}).to_list(500)

    test_stats = await db.tests.aggregate([
        {"$group": {"_id": "$created_by", "tests_created": {"$sum": 1}}}
    ]).to_list(500)
    test_map = {d["_id"]: d["tests_created"] for d in test_stats}

    participant_stats = await db.test_registrations.aggregate([
        {"$lookup": {"from": "tests", "localField": "test_id", "foreignField": "_id", "as": "test_info"}},
        {"$unwind": "$test_info"},
        {"$group": {"_id": "$test_info.created_by", "total_participants": {"$sum": 1}}}
    ]).to_list(500)
    participant_map = {d["_id"]: d["total_participants"] for d in participant_stats}

    rows = []
    search = q.strip().lower() if q else ""

    for auth in authorities:
        row = {
            "id": str(auth["_id"]),
            "name": auth["name"],
            "email": auth["email"],
            "created_at": auth["created_at"].isoformat() if isinstance(auth.get("created_at"), datetime) else None,
            "tests_created": test_map.get(str(auth["_id"]), 0),
            "total_participants": participant_map.get(str(auth["_id"]), 0),
            "status": "Active",
        }
        if search and search not in f"{row['name']} {row['email']}".lower():
            continue
        rows.append(row)

    reverse = sort_dir.lower() != "asc"
    sort_key = sort_by or "name"

    def authority_sort_key(item):
        if sort_key in {"tests_created", "total_participants"}:
            return int(item.get(sort_key) or 0)
        if sort_key == "created_at":
            return item.get(sort_key) or ""
        return str(item.get(sort_key) or "").lower()

    rows.sort(key=authority_sort_key, reverse=reverse)
    return rows


@router.get("/results")
async def admin_all_results(
    q: Optional[str] = None,
    exercise_type: Optional[str] = None,
    fitness_level: Optional[str] = None,
    cheat: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "desc",
    admin=Depends(require_admin),
):
    """Get all results."""
    db = get_db()
    
    docs = await db.analysis_results.find({}, {"frame_analyses": 0}).limit(200).to_list(200)
    uid_list = list({d["user_id"] for d in docs})
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1, "email": 1}).to_list(200)}

    results = []
    search = q.strip().lower() if q else ""

    for d in docs:
        s = serialize(sanitize_analysis_doc(d))
        u = umap.get(d["user_id"], {})
        s["athlete_name"] = u.get("name", "Unknown")
        s["athlete_email"] = u.get("email", "")

        if exercise_type and s.get("exercise_type") != exercise_type:
            continue
        if fitness_level and s.get("fitness_level") != fitness_level:
            continue
        if cheat == "flagged" and not s.get("cheat_detected"):
            continue
        if cheat == "clean" and s.get("cheat_detected"):
            continue
        if search and search not in f"{s.get('athlete_name', '')} {s.get('athlete_email', '')} {s.get('exercise_type', '')}".lower():
            continue

        results.append(s)

    reverse = sort_dir.lower() != "asc"
    sort_key = sort_by or "created_at"

    def result_sort_key(item):
        if sort_key in {"avg_correctness_score", "total_reps", "jump_height_cm", "estimated_percentile", "confidence_score"}:
            return float(item.get(sort_key) or 0)
        if sort_key == "athlete_name":
            return str(item.get(sort_key) or "").lower()
        if sort_key == "exercise_type":
            return str(item.get(sort_key) or "").lower()
        if sort_key == "created_at":
            return item.get(sort_key) or ""
        return str(item.get(sort_key) or "").lower()

    results.sort(key=result_sort_key, reverse=reverse)
    return results


@router.get("/ai-metrics")
async def admin_ai_metrics(days: int = 14, admin=Depends(require_admin)):
    """Get AI scoring metrics."""
    db = get_db()
    
    clamped_days = max(1, min(days, 90))
    now_utc = datetime.utcnow()
    window_start = now_utc - timedelta(days=clamped_days)

    docs = await db.analysis_results.find(
        {"created_at": {"$gte": window_start}},
        {
            "exercise_type": 1,
            "rule_score": 1,
            "ml_score": 1,
            "hybrid_form_score": 1,
            "confidence_score": 1,
            "detailed_feedback": 1,
            "created_at": 1,
        },
    ).to_list(5000)

    if not docs:
        return {
            "window_days": clamped_days,
            "sample_size": 0,
            "status": "no_data",
            "scoring_mode": "N/A",
            "averages": {},
            "drift_alerts": [],
            "by_exercise": {},
            "daily_trend": []
        }

    def avg(vals):
        return round(sum(vals) / len(vals), 2) if vals else 0.0

    rule_vals = [float(d.get("rule_score", 0) or 0) for d in docs]
    ml_vals = [float(d.get("ml_score", 0) or 0) for d in docs]
    hybrid_vals = [float(d.get("hybrid_form_score", d.get("avg_correctness_score", 0)) or 0) for d in docs]
    conf_vals = [float(d.get("confidence_score", 0) or 0) for d in docs]
    disagreement_vals = [abs(r - m) for r, m in zip(rule_vals, ml_vals)]

    quality_flag_count = 0
    invalid_capture_count = 0
    valid_conf_vals = []
    for d in docs:
        fb = d.get("detailed_feedback") or {}
        qflags = fb.get("quality_flags") or []
        invalid_attempt = bool(fb.get("invalid_attempt")) or ("invalid_assessment_capture" in qflags)
        if qflags:
            quality_flag_count += 1
        if invalid_attempt:
            invalid_capture_count += 1
        else:
            valid_conf_vals.append(float(d.get("confidence_score", 0) or 0))

    drift_alerts = []
    if avg(conf_vals) < 62:
        drift_alerts.append("low_confidence_trend")
    if valid_conf_vals and avg(valid_conf_vals) < 70:
        drift_alerts.append("low_valid_capture_confidence")
    if avg(disagreement_vals) > 18:
        drift_alerts.append("high_rule_ml_disagreement")
    if quality_flag_count / max(1, len(docs)) > 0.35:
        drift_alerts.append("high_capture_quality_issues")

    by_exercise = {}
    daily_trend_buckets = {}
    for d in docs:
        fb = d.get("detailed_feedback") or {}
        qflags = fb.get("quality_flags") or []
        invalid_attempt = bool(fb.get("invalid_attempt")) or ("invalid_assessment_capture" in qflags)

        ex = d.get("exercise_type", "unknown")
        by_exercise.setdefault(ex, {"n": 0, "hybrid": [], "confidence": [], "valid_confidence": [], "invalid_n": 0})
        by_exercise[ex]["n"] += 1
        by_exercise[ex]["hybrid"].append(float(d.get("hybrid_form_score", 0) or 0))
        by_exercise[ex]["confidence"].append(float(d.get("confidence_score", 0) or 0))
        if invalid_attempt:
            by_exercise[ex]["invalid_n"] += 1
        else:
            by_exercise[ex]["valid_confidence"].append(float(d.get("confidence_score", 0) or 0))

        created_at = d.get("created_at")
        if isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None
        if isinstance(created_at, datetime):
            day_key = created_at.date().isoformat()
            daily_trend_buckets.setdefault(
                day_key,
                {"n": 0, "confidence": [], "valid_confidence": [], "disagreement": [], "qflags": 0, "invalid_n": 0},
            )
            bucket = daily_trend_buckets[day_key]
            rule_v = float(d.get("rule_score", 0) or 0)
            ml_v = float(d.get("ml_score", 0) or 0)
            bucket["n"] += 1
            bucket["confidence"].append(float(d.get("confidence_score", 0) or 0))
            bucket["disagreement"].append(abs(rule_v - ml_v))
            if qflags:
                bucket["qflags"] += 1
            if invalid_attempt:
                bucket["invalid_n"] += 1
            else:
                bucket["valid_confidence"].append(float(d.get("confidence_score", 0) or 0))

    daily_trend = []
    for idx in range(clamped_days):
        day_iso = (now_utc.date() - timedelta(days=(clamped_days - 1 - idx))).isoformat()
        bucket = daily_trend_buckets.get(day_iso)
        if not bucket:
            daily_trend.append({
                "date": day_iso,
                "sample_size": 0,
                "confidence_score": None,
                "valid_confidence_score": None,
                "valid_sample_size": 0,
                "rule_ml_disagreement": None,
                "quality_flag_rate": None,
                "invalid_capture_rate": None,
            })
            continue

        daily_trend.append({
            "date": day_iso,
            "sample_size": bucket["n"],
            "confidence_score": avg(bucket["confidence"]),
            "valid_confidence_score": avg(bucket["valid_confidence"]),
            "valid_sample_size": max(0, bucket["n"] - bucket["invalid_n"]),
            "rule_ml_disagreement": avg(bucket["disagreement"]),
            "quality_flag_rate": round(bucket["qflags"] / max(1, bucket["n"]), 3),
            "invalid_capture_rate": round(bucket["invalid_n"] / max(1, bucket["n"]), 3),
        })

    return {
        "window_days": clamped_days,
        "sample_size": len(docs),
        "scoring_mode": "hybrid",
        "averages": {
            "rule_score": avg(rule_vals),
            "ml_score": avg(ml_vals),
            "hybrid_score": avg(hybrid_vals),
            "confidence_score": avg(conf_vals),
            "valid_capture_confidence_score": avg(valid_conf_vals),
            "rule_ml_disagreement": avg(disagreement_vals),
            "quality_flag_rate": round(quality_flag_count / max(1, len(docs)), 3),
            "invalid_capture_rate": round(invalid_capture_count / max(1, len(docs)), 3),
        },
        "drift_alerts": drift_alerts,
        "by_exercise": {
            ex: {
                "sample_size": vals["n"],
                "hybrid_score": avg(vals["hybrid"]),
                "confidence_score": avg(vals["confidence"]),
                "valid_capture_confidence_score": avg(vals["valid_confidence"]),
                "invalid_capture_rate": round(vals["invalid_n"] / max(1, vals["n"]), 3),
            }
            for ex, vals in by_exercise.items()
        },
        "daily_trend": daily_trend,
    }
