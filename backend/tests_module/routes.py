"""Test management routes."""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from uuid import uuid4
from typing import Optional

from database import get_db
from dependencies import get_current_user, require_authority
from models import CreateTestRequest, UpdateTestRequest, TestTemplateRequest
from tests_module.utils import compute_test_status, compute_test_score

router = APIRouter(prefix="/api", tags=["tests"])


async def resolve_target_users(target_emails: Optional[list[str]]) -> list[str]:
    """Resolve email addresses to user IDs."""
    if not target_emails:
        return []
    cleaned = [email.strip().lower() for email in target_emails if email and email.strip()]
    if not cleaned:
        return []
    db = get_db()
    users = await db.users.find({"email": {"$in": cleaned}}, {"_id": 1}).to_list(500)
    return [str(u["_id"]) for u in users]


@router.post("/tests")
async def create_test(req: CreateTestRequest, user=Depends(require_authority)):
    """Create a new test/assessment."""
    db = get_db()
    creator = await db.users.find_one({"_id": user["user_id"]}, {"name": 1})
    target_user_ids = await resolve_target_users(req.target_emails)
    tid = str(uuid4())

    await db.tests.insert_one({
        "_id": tid,
        "name": req.name,
        "sport": req.sport,
        "exercises": req.exercises,
        "scheduled_date": req.scheduled_date,
        "start_time": req.start_time,
        "duration_minutes": req.duration_minutes,
        "description": req.description,
        "max_participants": req.max_participants,
        "assigned_roster": req.assigned_roster,
        "target_emails": req.target_emails or [],
        "target_user_ids": target_user_ids,
        "created_by": user["user_id"],
        "created_by_name": creator.get("name", "") if creator else "",
        "status": "upcoming",
        "is_archived": False,
        "created_at": datetime.utcnow(),
    })

    return {"test_id": tid, "status": "upcoming"}


@router.get("/tests")
async def list_tests(
    status: Optional[str] = None,
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    include_archived: bool = False,
    user=Depends(get_current_user),
):
    """List all tests."""
    db = get_db()
    docs = await db.tests.find({}).to_list(200)
    search = q.strip().lower() if q else ""
    result = []

    for doc in docs:
        if doc.get("is_archived") and not include_archived:
            continue

        computed_status = compute_test_status(doc)
        if status and computed_status != status:
            continue

        if search and search not in f"{doc.get('name', '')} {doc.get('sport', '')}".lower():
            continue

        if user.get("role") == "athlete":
            target_users = doc.get("target_user_ids") or []
            if target_users and user["user_id"] not in target_users:
                continue

        s = {
            "id": str(doc["_id"]),
            "name": doc.get("name"),
            "sport": doc.get("sport"),
            "status": computed_status,
            "scheduled_date": doc.get("scheduled_date"),
            "start_time": doc.get("start_time"),
        }
        result.append(s)

    return result


@router.get("/tests/{test_id}")
async def get_test(test_id: str, user=Depends(get_current_user)):
    """Get test details."""
    db = get_db()
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")

    if doc.get("is_archived") and user.get("role") == "athlete":
        raise HTTPException(403, "Test archived")

    target_users = doc.get("target_user_ids") or []
    if user.get("role") == "athlete" and target_users and user["user_id"] not in target_users:
        raise HTTPException(403, "Not registered for this test")

    from results.utils import serialize
    s = serialize(doc)
    s["status"] = compute_test_status(doc)
    s["is_archived"] = bool(doc.get("is_archived", False))
    
    reg = await db.test_registrations.find_one({"test_id": test_id, "user_id": user["user_id"]})
    s["is_registered"] = reg is not None
    s["participant_count"] = await db.test_registrations.count_documents({"test_id": test_id})
    s["target_count"] = len(doc.get("target_user_ids", [])) if isinstance(doc.get("target_user_ids"), list) else 0

    return s


@router.patch("/tests/{test_id}")
async def update_test(test_id: str, req: UpdateTestRequest, user=Depends(require_authority)):
    """Update test."""
    db = get_db()
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")

    updates = {k: v for k, v in req.dict().items() if v is not None}
    if not updates:
        return {"success": True}

    if "target_emails" in updates:
        updates["target_user_ids"] = await resolve_target_users(updates["target_emails"])

    await db.tests.update_one({"_id": test_id}, {"$set": updates})
    return {"success": True}


@router.patch("/tests/{test_id}/status")
async def update_test_status(test_id: str, status: str, user=Depends(require_authority)):
    """Update test status."""
    if status not in ("upcoming", "active", "completed"):
        raise HTTPException(400, "Invalid status")
    
    db = get_db()
    res = await db.tests.update_one({"_id": test_id}, {"$set": {"status": status}})
    if res.matched_count == 0:
        raise HTTPException(404, "Test not found")
    return {"success": True}


@router.delete("/tests/{test_id}")
async def soft_delete_test(test_id: str, user=Depends(require_authority)):
    """Soft delete test."""
    db = get_db()
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")

    await db.tests.update_one(
        {"_id": test_id},
        {"$set": {
            "is_archived": True,
            "status": "completed",
            "deleted_at": datetime.utcnow(),
            "deleted_by": user["user_id"]
        }}
    )
    return {"success": True, "is_archived": True}


@router.post("/tests/{test_id}/register")
async def register_for_test(test_id: str, user=Depends(get_current_user)):
    """Register for test."""
    if user.get("role") not in ("athlete",):
        raise HTTPException(403, "Only athletes can register")

    db = get_db()
    test = await db.tests.find_one({"_id": test_id})
    if not test:
        raise HTTPException(404, "Test not found")

    computed_status = compute_test_status(test)
    if computed_status == "completed":
        raise HTTPException(400, "Test already completed")

    try:
        await db.test_registrations.insert_one({
            "_id": f"{test_id}#{user['user_id']}",
            "test_id": test_id,
            "user_id": user["user_id"],
            "registered_at": datetime.utcnow(),
        })
    except Exception:
        raise HTTPException(409, "Already registered")

    return {"success": True}


@router.delete("/tests/{test_id}/register")
async def unregister_from_test(test_id: str, user=Depends(get_current_user)):
    """Unregister from test."""
    db = get_db()
    res = await db.test_registrations.delete_one({"test_id": test_id, "user_id": user["user_id"]})
    if res.deleted_count == 0:
        raise HTTPException(404, "Not registered")
    return {"success": True}


@router.get("/tests/{test_id}/participants")
async def test_participants(test_id: str, user=Depends(require_authority)):
    """Get test participants."""
    db = get_db()
    regs = await db.test_registrations.find({"test_id": test_id}).to_list(500)
    uid_list = [r["user_id"] for r in regs]
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"password_hash": 0}).to_list(500)}

    return [
        {
            "user_id": r["user_id"],
            "registered_at": r["registered_at"].isoformat() if isinstance(r.get("registered_at"), datetime) else "",
            "name": umap.get(r["user_id"], {}).get("name", "Unknown"),
            "email": umap.get(r["user_id"], {}).get("email", ""),
        }
        for r in regs
    ]


@router.get("/tests/{test_id}/leaderboard")
async def test_leaderboard(test_id: str, user=Depends(get_current_user)):
    """Get test leaderboard."""
    db = get_db()
    test = await db.tests.find_one({"_id": test_id})
    if not test:
        raise HTTPException(404, "Test not found")

    sessions = await db.test_sessions.find({"test_id": test_id, "status": "completed"}).to_list(500)
    sid_list = [str(s["_id"]) for s in sessions]
    uid_list = list({s["user_id"] for s in sessions})
    results = await db.analysis_results.find({"session_id": {"$in": sid_list}}).to_list(500)
    
    rmap = {r["session_id"]: r for r in results}
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1, "email": 1}).to_list(500)}

    exercises = test.get("exercises", [])
    best = {}

    for s in sessions:
        uid = s["user_id"]
        result = rmap.get(str(s["_id"]))
        if not result:
            continue

        score = compute_test_score(result, exercises)
        if uid not in best or score > best[uid]["score"]:
            user_info = umap.get(uid, {})
            best[uid] = {
                "user_id": uid,
                "name": user_info.get("name", "Unknown"),
                "email": user_info.get("email", ""),
                "score": score,
                "total_reps": result.get("total_reps", 0),
                "avg_correctness_score": result.get("avg_correctness_score", 0),
                "fitness_level": result.get("fitness_level", ""),
                "form_grade": result.get("form_grade", ""),
                "cheat_detected": result.get("cheat_detected", False),
            }

    entries = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    for i, e in enumerate(entries):
        e["rank"] = i + 1

    return entries


@router.get("/tests/{test_id}/analytics")
async def test_analytics(test_id: str, user=Depends(require_authority)):
    """Get test analytics."""
    db = get_db()
    test = await db.tests.find_one({"_id": test_id})
    if not test:
        raise HTTPException(404, "Test not found")

    registrations = await db.test_registrations.find({"test_id": test_id}).to_list(500)
    total_registered = len(registrations)

    sessions = await db.test_sessions.find({"test_id": test_id, "status": "completed"}).to_list(500)
    total_completed = len(sessions)
    completion_rate = (total_completed / total_registered * 100) if total_registered > 0 else 0

    sid_list = [str(s["_id"]) for s in sessions]
    results = await db.analysis_results.find({"session_id": {"$in": sid_list}}).to_list(500)

    scores = [r.get("avg_correctness_score", 0) for r in results]
    avg_score = sum(scores) / len(scores) if scores else 0

    cheat_count = sum(1 for r in results if r.get("cheat_detected", False))

    exercises = test.get("exercises", [])
    ex_performance = {}
    for ex in exercises:
        ex_results = [r for r in results if r.get("exercise_type") == ex]
        if ex_results:
            ex_scores = [r.get("avg_correctness_score", 0) for r in ex_results]
            ex_performance[ex] = round(sum(ex_scores) / len(ex_scores), 1)

    top_performers = sorted(
        [
            {
                "user_id": r.get("user_id"),
                "score": r.get("avg_correctness_score", 0),
                "exercise": r.get("exercise_type"),
            }
            for r in results
        ],
        key=lambda x: x["score"],
        reverse=True,
    )[:3]

    return {
        "test_id": test_id,
        "test_name": test.get("name"),
        "total_registered": total_registered,
        "total_completed": total_completed,
        "completion_rate": round(completion_rate, 1),
        "avg_score": round(avg_score, 1),
        "cheat_flags": cheat_count,
        "exercise_performance": ex_performance,
        "top_performers": top_performers,
    }


@router.get("/test-templates")
async def list_test_templates(user=Depends(require_authority)):
    """List test templates."""
    db = get_db()
    filt = {} if user.get("role") == "admin" else {"created_by": user["user_id"]}
    docs = await db.test_templates.find(filt).sort("created_at", -1).to_list(100)
    from results.utils import serialize
    return [serialize(d) for d in docs]


@router.post("/test-templates")
async def create_test_template(req: TestTemplateRequest, user=Depends(require_authority)):
    """Create test template."""
    db = get_db()
    tid = str(uuid4())
    target_user_ids = await resolve_target_users(req.target_emails)

    await db.test_templates.insert_one({
        "_id": tid,
        "name": req.name,
        "sport": req.sport,
        "exercises": req.exercises,
        "duration_minutes": req.duration_minutes or 60,
        "description": req.description,
        "max_participants": req.max_participants,
        "assigned_roster": req.assigned_roster,
        "target_emails": req.target_emails or [],
        "target_user_ids": target_user_ids,
        "created_by": user["user_id"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    })

    return {"template_id": tid, "success": True}
