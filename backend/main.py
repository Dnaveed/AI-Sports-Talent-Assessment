"""
AthleteAI Backend - FastAPI + MongoDB (Motor async driver)
Replaces SQLite with MongoDB for all data storage.
Connect MongoDB Compass to: mongodb://localhost:27017
Database: athleteai
Collections: users, test_sessions,processing_jobs, analysis_results
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn, os, uuid, json, hashlib, hmac, base64, asyncio
from datetime import datetime, timedelta
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SECRET_KEY = os.environ.get("SECRET_KEY", "athleteai-secret-key-change-in-production")
MONGO_URI  = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "athleteai")

client = db = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]
    await db.users.create_index("email", unique=True)
    await db.test_sessions.create_index("user_id")
    await db.processing_jobs.create_index("session_id")
    await db.analysis_results.create_index("user_id")
    await db.analysis_results.create_index("session_id")
    await db.analysis_results.create_index([("exercise_type", 1), ("fitness_level", 1)])
    await db.analysis_results.create_index("created_at")
    await db.tests.create_index("status")
    await db.tests.create_index("created_by")
    await db.test_registrations.create_index([("test_id", 1), ("user_id", 1)], unique=True)
    await db.test_registrations.create_index("user_id")
    if not await db.users.find_one({"email": "admin@athleteai.com"}):
        await db.users.insert_one({
            "_id": "admin-001", "email": "admin@athleteai.com",
            "name": "System Admin", "password_hash": hash_password("admin123"),
            "role": "admin", "age": None, "weight_kg": None, "height_cm": None,
            "created_at": datetime.utcnow(), "last_login": None,
        })
    print(f"✅  MongoDB connected → {MONGO_URI}/{MONGO_DB}")
    print(f"📡  Open Compass and connect to: {MONGO_URI}")
    yield
    client.close()


app = FastAPI(title="AthleteAI API", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer()


# ── Serialization ──────────────────────────────────────────────────────────────

def serialize(doc: dict) -> dict:
    if doc is None: return None
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

def sl(docs) -> list: return [serialize(d) for d in docs]


# ── Auth helpers ───────────────────────────────────────────────────────────────

def hash_password(p: str) -> str:
    return hashlib.sha256(f"{p}{SECRET_KEY}".encode()).hexdigest()

def create_token(user_id: str, role: str) -> str:
    payload = json.dumps({"user_id": user_id, "role": role,
                          "exp": (datetime.utcnow() + timedelta(hours=24)).isoformat()})
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(payload.encode()).decode() + "." + sig

def verify_token(token: str) -> Optional[dict]:
    try:
        b64, sig = token.rsplit(".", 1)
        payload = base64.b64decode(b64.encode()).decode()
        if not hmac.compare_digest(sig, hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()):
            return None
        data = json.loads(payload)
        return data if datetime.fromisoformat(data["exp"]) > datetime.utcnow() else None
    except Exception: return None

async def get_current_user(creds: HTTPAuthorizationCredentials = Depends(security)):
    data = verify_token(creds.credentials)
    if not data: raise HTTPException(401, "Invalid or expired token")
    return data

async def require_admin(user=Depends(get_current_user)):
    if user.get("role") != "admin": raise HTTPException(403, "Admin access required")
    return user


# ── Pydantic models ────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str; name: str; password: str
    role: Optional[str] = "athlete"
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None

class LoginRequest(BaseModel):
    email: str; password: str

class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


# ── Auth endpoints ─────────────────────────────────────────────────────────────

@app.post("/api/auth/register")
async def register(req: RegisterRequest):
    uid = str(uuid.uuid4())
    role = req.role if req.role in ("athlete", "authority") else "athlete"
    try:
        await db.users.insert_one({
            "_id": uid, "email": req.email, "name": req.name,
            "password_hash": hash_password(req.password), "role": role,
            "age": req.age, "weight_kg": req.weight_kg, "height_cm": req.height_cm,
            "created_at": datetime.utcnow(), "last_login": None,
        })
    except Exception:
        raise HTTPException(409, "Email already registered")
    return {"token": create_token(uid, role),
            "user": {"id": uid, "email": req.email, "name": req.name, "role": role}}

@app.post("/api/auth/login")
async def login(req: LoginRequest):
    user = await db.users.find_one({"email": req.email, "password_hash": hash_password(req.password)})
    if not user: raise HTTPException(401, "Invalid credentials")
    await db.users.update_one({"_id": user["_id"]}, {"$set": {"last_login": datetime.utcnow()}})
    return {"token": create_token(str(user["_id"]), user["role"]),
            "user": {"id": str(user["_id"]), "email": user["email"],
                     "name": user["name"], "role": user["role"]}}

@app.get("/api/auth/me")
async def get_me(user=Depends(get_current_user)):
    doc = await db.users.find_one({"_id": user["user_id"]}, {"password_hash": 0})
    if not doc: raise HTTPException(404, "User not found")
    return serialize(doc)


# ── Upload ─────────────────────────────────────────────────────────────────────

ALLOWED = {"video/mp4","video/quicktime","video/webm","video/x-msvideo"}

@app.post("/api/sessions/upload")
async def upload_video(background_tasks: BackgroundTasks,
                       exercise_type: str = "pushup",
                       test_id: Optional[str] = None,
                       file: UploadFile = File(...),
                       user=Depends(get_current_user)):
    if file.content_type not in ALLOWED:
        raise HTTPException(400, f"Invalid type: {file.content_type}")
    content = await file.read()
    if len(content) > 200 * 1024 * 1024: raise HTTPException(413, "File too large")

    sid = str(uuid.uuid4()); jid = str(uuid.uuid4())
    ext = Path(file.filename or "v.mp4").suffix or ".mp4"
    vpath = UPLOAD_DIR / f"{sid}{ext}"
    vpath.write_bytes(content)
    now = datetime.utcnow()

    await db.test_sessions.insert_one({
        "_id": sid, "user_id": user["user_id"], "exercise_type": exercise_type,
        "video_path": str(vpath), "status": "processing",
        "test_id": test_id,
        "created_at": now, "completed_at": None,
    })
    await db.processing_jobs.insert_one({
        "_id": jid, "session_id": sid, "status": "queued",
        "progress": 0.0, "error_message": None, "created_at": now, "updated_at": now,
    })
    background_tasks.add_task(process_video_job, jid, sid, str(vpath), exercise_type, user["user_id"])
    return {"session_id": sid, "job_id": jid, "status": "processing"}


async def process_video_job(jid, sid, vpath, exercise_type, user_id):
    async def prog(p):
        await db.processing_jobs.update_one({"_id": jid},
            {"$set": {"progress": round(p, 1), "updated_at": datetime.utcnow()}})

    await db.processing_jobs.update_one({"_id": jid}, {"$set": {"status": "processing"}})
    try:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "pose_module"))
            from pose_analyzer import VideoProcessor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: VideoProcessor().process_video(vpath, exercise_type))
            rd = result.__dict__
        except ImportError:
            await asyncio.sleep(3); rd = _sim(exercise_type)

        await prog(90)
        rid = str(uuid.uuid4()); pm = rd.get("performance_metrics", {})
        await db.analysis_results.insert_one({
            "_id": rid, "session_id": sid, "user_id": user_id,
            "exercise_type": exercise_type,
            "total_reps": rd.get("total_reps", 0),
            "avg_correctness_score": rd.get("avg_correctness_score", 0),
            "jump_height_cm": rd.get("jump_height_cm"),
            "duration_seconds": rd.get("duration", 0),
            "reps_per_minute": rd.get("summary", {}).get("reps_per_minute", 0),
            "fitness_level": pm.get("fitness_level", "Unknown"),
            "form_grade": pm.get("form_grade", "N/A"),
            "estimated_percentile": pm.get("estimated_percentile", 0),
            "cheat_detected": bool(rd.get("cheat_detected", False)),
            "cheat_reasons": rd.get("cheat_reasons", []),
            "frame_analyses": rd.get("frame_analyses", [])[:100],
            "performance_metrics": pm,
            "created_at": datetime.utcnow(),
        })
        await db.test_sessions.update_one({"_id": sid},
            {"$set": {"status": "completed", "completed_at": datetime.utcnow()}})
        await db.processing_jobs.update_one({"_id": jid},
            {"$set": {"status": "completed", "progress": 100.0, "updated_at": datetime.utcnow()}})
    except Exception as e:
        await db.processing_jobs.update_one({"_id": jid},
            {"$set": {"status": "failed", "error_message": str(e), "updated_at": datetime.utcnow()}})
        await db.test_sessions.update_one({"_id": sid}, {"$set": {"status": "failed"}})


def _sim(ex):
    import random; r = random.randint(10, 40); c = random.uniform(65, 95)
    return {"exercise_type": ex, "total_frames": 900, "fps": 30, "duration": 30.0,
            "total_reps": r, "avg_correctness_score": round(c, 1),
            "jump_height_cm": round(random.uniform(30, 60), 1) if ex == "vertical_jump" else None,
            "cheat_detected": False, "cheat_reasons": [], "frame_analyses": [],
            "summary": {"reps_per_minute": round(r * 2, 1)},
            "performance_metrics": {"fitness_level": random.choice(["Beginner","Intermediate","Advanced"]),
                                    "form_grade": random.choice(["A","B","C"]),
                                    "estimated_percentile": random.randint(30, 90),
                                    "benchmarks": {"beginner": 10, "intermediate": 25, "advanced": 40},
                                    "metric_value": r, "metric_unit": "reps"}}


@app.get("/api/jobs/{job_id}")
async def job_status(job_id: str, user=Depends(get_current_user)):
    doc = await db.processing_jobs.find_one({"_id": job_id})
    if not doc: raise HTTPException(404, "Job not found")
    return serialize(doc)


# ── Results ────────────────────────────────────────────────────────────────────

@app.get("/api/results")
async def my_results(user=Depends(get_current_user)):
    docs = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).limit(50).to_list(50)
    return sl(docs)

@app.get("/api/results/{result_id}")
async def get_result(result_id: str, user=Depends(get_current_user)):
    doc = await db.analysis_results.find_one({"_id": result_id})
    if not doc: raise HTTPException(404, "Not found")
    if doc["user_id"] != user["user_id"] and user.get("role") != "admin":
        raise HTTPException(403, "Access denied")
    return serialize(doc)

@app.get("/api/sessions/{session_id}/result")
async def session_result(session_id: str, user=Depends(get_current_user)):
    doc = await db.analysis_results.find_one({"session_id": session_id})
    if not doc: raise HTTPException(404, "No result yet")
    return serialize(doc)


# ── Admin ──────────────────────────────────────────────────────────────────────

@app.get("/api/admin/stats")
async def admin_stats(admin=Depends(require_admin)):
    total_athletes = await db.users.count_documents({"role": "athlete"})
    total_authorities = await db.users.count_documents({"role": "authority"})
    total_sessions = await db.test_sessions.count_documents({})
    completed      = await db.test_sessions.count_documents({"status": "completed"})
    cheat_flags    = await db.analysis_results.count_documents({"cheat_detected": True})
    avg_r = await db.analysis_results.aggregate(
        [{"$group": {"_id": None, "avg": {"$avg": "$avg_correctness_score"}}}]
    ).to_list(1)
    avg_score = round(avg_r[0]["avg"], 1) if avg_r else 0.0
    ex_breakdown = await db.analysis_results.aggregate([
        {"$group": {"_id": "$exercise_type", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}}
    ]).to_list(20)
    recent_docs = await db.analysis_results.find(
        {}, {"user_id":1,"exercise_type":1,"avg_correctness_score":1,"fitness_level":1,"created_at":1}
    ).sort("created_at", -1).limit(10).to_list(10)
    uid_list = list({d["user_id"] for d in recent_docs})
    umap = {str(u["_id"]): u["name"] for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1}).to_list(50)}
    recent = [{"name": umap.get(d["user_id"], "Unknown"), "exercise_type": d["exercise_type"],
               "avg_correctness_score": d.get("avg_correctness_score", 0),
               "fitness_level": d.get("fitness_level", "—"),
               "created_at": d["created_at"].isoformat() if isinstance(d.get("created_at"), datetime) else ""}
              for d in recent_docs]
    return {"total_athletes": total_athletes, "total_authorities": total_authorities,
            "total_sessions": total_sessions,
            "completed_sessions": completed, "avg_correctness_score": avg_score,
            "cheat_flags": cheat_flags,
            "exercise_breakdown": [{"exercise_type": d["_id"], "count": d["count"]} for d in ex_breakdown],
            "recent_activity": recent}

@app.get("/api/admin/athletes")
async def admin_athletes(age_min: Optional[int]=None, age_max: Optional[int]=None,
                         fitness_level: Optional[str]=None, exercise_type: Optional[str]=None,
                         admin=Depends(require_admin)):
    uf: dict = {"role": "athlete"}
    if age_min: uf.setdefault("age", {})["$gte"] = age_min
    if age_max: uf.setdefault("age", {})["$lte"] = age_max
    users = await db.users.find(uf, {"password_hash": 0}).to_list(500)
    rf: dict = {}
    if fitness_level: rf["fitness_level"] = fitness_level
    if exercise_type: rf["exercise_type"] = exercise_type
    stats = await db.analysis_results.aggregate([
        {"$match": rf},
        {"$group": {"_id": "$user_id", "total_tests": {"$sum": 1},
                    "avg_score": {"$avg": "$avg_correctness_score"},
                    "last_test": {"$max": "$created_at"}}}
    ]).to_list(500)
    smap = {d["_id"]: d for d in stats}
    return [{"id": str(u["_id"]), "name": u["name"], "email": u["email"],
             "age": u.get("age"), "height_cm": u.get("height_cm"), "weight_kg": u.get("weight_kg"),
             "total_tests": smap.get(str(u["_id"]), {}).get("total_tests", 0),
             "avg_score": round(smap[str(u["_id"])]["avg_score"], 1)
                          if smap.get(str(u["_id"]), {}).get("avg_score") else None,
             "last_test": smap[str(u["_id"])]["last_test"].isoformat()
                          if isinstance(smap.get(str(u["_id"]), {}).get("last_test"), datetime) else None}
            for u in users]

@app.get("/api/admin/authorities")
async def admin_authorities(admin=Depends(require_admin)):
    """Get all sports authority accounts with their test creation stats"""
    # Fetch all authority users
    authorities = await db.users.find({"role": "authority"}, {"password_hash": 0}).to_list(500)

    # Get test creation stats for each authority
    test_stats = await db.tests.aggregate([
        {"$group": {
            "_id": "$created_by",
            "tests_created": {"$sum": 1}
        }}
    ]).to_list(500)
    test_map = {d["_id"]: d["tests_created"] for d in test_stats}

    # Get participant stats for each authority's tests
    participant_stats = await db.test_registrations.aggregate([
        {"$lookup": {
            "from": "tests",
            "localField": "test_id",
            "foreignField": "_id",
            "as": "test_info"
        }},
        {"$unwind": "$test_info"},
        {"$group": {
            "_id": "$test_info.created_by",
            "total_participants": {"$sum": 1}
        }}
    ]).to_list(500)
    participant_map = {d["_id"]: d["total_participants"] for d in participant_stats}

    return [{
        "id": str(auth["_id"]),
        "name": auth["name"],
        "email": auth["email"],
        "created_at": auth["created_at"].isoformat() if isinstance(auth.get("created_at"), datetime) else None,
        "tests_created": test_map.get(str(auth["_id"]), 0),
        "total_participants": participant_map.get(str(auth["_id"]), 0),
        "status": "Active"
    } for auth in authorities]

@app.get("/api/admin/results")
async def admin_all_results(admin=Depends(require_admin)):
    docs = await db.analysis_results.find({}, {"frame_analyses": 0}).sort("created_at", -1).limit(100).to_list(100)
    uid_list = list({d["user_id"] for d in docs})
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1, "email": 1}).to_list(200)}
    results = []
    for d in docs:
        s = serialize(d)
        u = umap.get(d["user_id"], {})
        s["athlete_name"] = u.get("name", "Unknown"); s["athlete_email"] = u.get("email", "")
        results.append(s)
    return results


# ── Profile ────────────────────────────────────────────────────────────────────

@app.put("/api/profile")
async def update_profile(req: UpdateProfileRequest, user=Depends(get_current_user)):
    updates = {k: v for k, v in req.dict().items() if v is not None}
    if not updates: raise HTTPException(400, "No fields to update")
    await db.users.update_one({"_id": user["user_id"]}, {"$set": updates})
    return {"success": True}



# ── Tests / Assessments ────────────────────────────────────────────────────

BENCHMARKS = {
    "pushup":        {"beginner": 10, "intermediate": 25, "advanced": 40},
    "squat":         {"beginner": 8,  "intermediate": 20, "advanced": 35},
    "situp":         {"beginner": 12, "intermediate": 28, "advanced": 45},
    "vertical_jump": {"beginner": 20, "intermediate": 40, "advanced": 60},
    "jumping_jack":  {"beginner": 15, "intermediate": 30, "advanced": 50},
    "lunge":         {"beginner": 8,  "intermediate": 18, "advanced": 30},
}

def compute_test_score(result: dict, exercises: list) -> float:
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


class CreateTestRequest(BaseModel):
    name: str
    sport: str
    exercises: list
    scheduled_date: str
    start_time: str
    duration_minutes: int
    description: Optional[str] = None
    max_participants: Optional[int] = None


def require_authority(user=Depends(get_current_user)):
    if user.get("role") not in ("admin", "authority"):
        raise HTTPException(403, "Authority access required")
    return user


@app.post("/api/tests")
async def create_test(req: CreateTestRequest, user=Depends(require_authority)):
    creator = await db.users.find_one({"_id": user["user_id"]}, {"name": 1})
    tid = str(uuid.uuid4())
    await db.tests.insert_one({
        "_id": tid, "name": req.name, "sport": req.sport,
        "exercises": req.exercises,
        "scheduled_date": req.scheduled_date, "start_time": req.start_time,
        "duration_minutes": req.duration_minutes,
        "description": req.description, "max_participants": req.max_participants,
        "created_by": user["user_id"],
        "created_by_name": creator.get("name", "") if creator else "",
        "status": "upcoming",
        "created_at": datetime.utcnow(),
    })
    return {"test_id": tid, "status": "upcoming"}


def compute_test_status(test: dict) -> str:
    """
    Compute the actual test status based on scheduled date/time and duration.
    - upcoming: before scheduled_date + start_time
    - active: within the test window (scheduled_date + start_time to scheduled_date + start_time + duration)
    - completed: after the test window or manually marked as completed
    """
    stored_status = test.get("status", "upcoming")

    # If manually marked as completed, respect that
    if stored_status == "completed":
        return "completed"

    try:
        scheduled_date = test.get("scheduled_date")  # "YYYY-MM-DD"
        start_time = test.get("start_time", "00:00")  # "HH:MM"
        duration_minutes = test.get("duration_minutes", 60)

        if not scheduled_date:
            return stored_status

        # Parse scheduled datetime
        dt_str = f"{scheduled_date} {start_time}"
        scheduled_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
        end_dt = scheduled_dt + timedelta(minutes=duration_minutes)
        now = datetime.utcnow()

        # Determine status based on current time
        if now < scheduled_dt:
            return "upcoming"
        elif now >= scheduled_dt and now <= end_dt:
            return "active"
        else:
            return "completed"
    except Exception:
        return stored_status


@app.get("/api/tests")
async def list_tests(status: Optional[str] = None, user=Depends(get_current_user)):
    filt = {}
    # Note: we don't filter by status in the query anymore since we compute it dynamically
    docs = await db.tests.find(filt).sort("scheduled_date", 1).to_list(200)
    result = []
    for doc in docs:
        s = serialize(doc)
        # Compute real-time status
        computed_status = compute_test_status(doc)
        s["status"] = computed_status

        # Filter by status if requested (after computing)
        if status and computed_status != status:
            continue

        reg = await db.test_registrations.find_one({"test_id": str(doc["_id"]), "user_id": user["user_id"]})
        s["is_registered"] = reg is not None
        s["participant_count"] = await db.test_registrations.count_documents({"test_id": str(doc["_id"])})
        result.append(s)
    return result


@app.get("/api/tests/{test_id}")
async def get_test(test_id: str, user=Depends(get_current_user)):
    doc = await db.tests.find_one({"_id": test_id})
    if not doc: raise HTTPException(404, "Test not found")
    s = serialize(doc)
    # Compute real-time status
    s["status"] = compute_test_status(doc)
    reg = await db.test_registrations.find_one({"test_id": test_id, "user_id": user["user_id"]})
    s["is_registered"] = reg is not None
    s["participant_count"] = await db.test_registrations.count_documents({"test_id": test_id})
    return s


@app.patch("/api/tests/{test_id}/status")
async def update_test_status(test_id: str, status: str, user=Depends(require_authority)):
    if status not in ("upcoming", "active", "completed"):
        raise HTTPException(400, "Invalid status")
    res = await db.tests.update_one({"_id": test_id}, {"$set": {"status": status}})
    if res.matched_count == 0: raise HTTPException(404, "Test not found")
    return {"success": True}


@app.post("/api/tests/{test_id}/register")
async def register_for_test(test_id: str, user=Depends(get_current_user)):
    if user.get("role") not in ("athlete",):
        raise HTTPException(403, "Athletes only")
    test = await db.tests.find_one({"_id": test_id})
    if not test: raise HTTPException(404, "Test not found")

    # Check computed status
    computed_status = compute_test_status(test)
    if computed_status == "completed":
        raise HTTPException(400, "Test already completed")

    try:
        await db.test_registrations.insert_one({
            "_id": str(uuid.uuid4()), "test_id": test_id,
            "user_id": user["user_id"], "registered_at": datetime.utcnow(), "status": "registered",
        })
    except Exception:
        raise HTTPException(409, "Already registered")
    return {"success": True}


@app.delete("/api/tests/{test_id}/register")
async def unregister_from_test(test_id: str, user=Depends(get_current_user)):
    res = await db.test_registrations.delete_one({"test_id": test_id, "user_id": user["user_id"]})
    if res.deleted_count == 0: raise HTTPException(404, "Registration not found")
    return {"success": True}


@app.get("/api/tests/{test_id}/participants")
async def test_participants(test_id: str, user=Depends(require_authority)):
    regs = await db.test_registrations.find({"test_id": test_id}).to_list(500)
    uid_list = [r["user_id"] for r in regs]
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"password_hash": 0}).to_list(500)}
    return [{"user_id": r["user_id"],
             "registered_at": r["registered_at"].isoformat() if isinstance(r.get("registered_at"), datetime) else "",
             "name": umap.get(r["user_id"], {}).get("name", "Unknown"),
             "email": umap.get(r["user_id"], {}).get("email", "")} for r in regs]


@app.get("/api/tests/{test_id}/leaderboard")
async def test_leaderboard(test_id: str, user=Depends(get_current_user)):
    test = await db.tests.find_one({"_id": test_id})
    if not test: raise HTTPException(404, "Test not found")
    sessions = await db.test_sessions.find({"test_id": test_id, "status": "completed"}).to_list(500)
    sid_list = [str(s["_id"]) for s in sessions]
    uid_list  = list({s["user_id"] for s in sessions})
    results  = await db.analysis_results.find({"session_id": {"$in": sid_list}}).to_list(500)
    rmap = {r["session_id"]: r for r in results}
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1, "email": 1}).to_list(500)}
    exercises = test.get("exercises", [])
    best: dict = {}
    for s in sessions:
        uid  = s["user_id"]
        sid  = str(s["_id"])
        result = rmap.get(sid)
        if not result: continue
        score = compute_test_score(result, exercises)
        if uid not in best or score > best[uid]["score"]:
            best[uid] = {
                "user_id": uid,
                "name": umap.get(uid, {}).get("name", "Unknown"),
                "email": umap.get(uid, {}).get("email", ""),
                "total_reps": result.get("total_reps", 0),
                "jump_height_cm": result.get("jump_height_cm"),
                "avg_correctness_score": result.get("avg_correctness_score", 0),
                "fitness_level": result.get("fitness_level", "Unknown"),
                "form_grade": result.get("form_grade", "N/A"),
                "cheat_detected": bool(result.get("cheat_detected", False)),
                "score": score,
            }
    entries = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    for i, e in enumerate(entries): e["rank"] = i + 1
    return entries


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    try:
        await client.admin.command("ping"); mongo_ok = "connected"
    except Exception as e:
        mongo_ok = f"error: {e}"
    return {"status": "ok", "mongo": mongo_ok, "database": MONGO_DB,
            "timestamp": datetime.utcnow().isoformat()}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)