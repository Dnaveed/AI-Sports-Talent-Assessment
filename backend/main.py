"""
AthleteAI Backend - FastAPI + MongoDB (Motor async driver)
Replaces SQLite with MongoDB for all data storage.
Connect MongoDB Compass to: mongodb://localhost:27017
Database: athleteai
Collections: users, test_sessions, processing_jobs, analysis_results
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn, os, uuid, json, hashlib, hmac, base64, asyncio
from datetime import datetime, timedelta
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient


app = FastAPI(title="AthleteAI API", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
security = HTTPBearer()

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SECRET_KEY = os.environ.get("SECRET_KEY", "athleteai-secret-key-change-in-production")
MONGO_URI  = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "athleteai")

client = db = None


@app.on_event("startup")
async def startup_db():
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
    if not await db.users.find_one({"email": "admin@athleteai.com"}):
        await db.users.insert_one({
            "_id": "admin-001", "email": "admin@athleteai.com",
            "name": "System Admin", "password_hash": hash_password("admin123"),
            "role": "admin", "age": None, "weight_kg": None, "height_cm": None,
            "created_at": datetime.utcnow(), "last_login": None,
        })
    print(f"✅  MongoDB connected → {MONGO_URI}/{MONGO_DB}")
    print(f"📡  Open Compass and connect to: {MONGO_URI}")


@app.on_event("shutdown")
async def shutdown_db():
    client.close()


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
    try:
        await db.users.insert_one({
            "_id": uid, "email": req.email, "name": req.name,
            "password_hash": hash_password(req.password), "role": "athlete",
            "age": req.age, "weight_kg": req.weight_kg, "height_cm": req.height_cm,
            "created_at": datetime.utcnow(), "last_login": None,
        })
    except Exception:
        raise HTTPException(409, "Email already registered")
    return {"token": create_token(uid, "athlete"),
            "user": {"id": uid, "email": req.email, "name": req.name, "role": "athlete"}}

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
    return {"total_athletes": total_athletes, "total_sessions": total_sessions,
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