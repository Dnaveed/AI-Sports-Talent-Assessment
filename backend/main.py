"""
AthleteAI Backend - FastAPI + MongoDB (Motor async driver)
Replaces SQLite with MongoDB for all data storage.
Connect MongoDB Compass to: mongodb://localhost:27017
Database: athleteai
Collections: users, test_sessions,processing_jobs, analysis_results
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Form
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn, os, uuid, json, hashlib, hmac, base64, asyncio, csv, io
from datetime import datetime, timedelta
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
try:
    import certifi
except ImportError:
    certifi = None
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
SECRET_KEY = os.environ.get("SECRET_KEY", "athleteai-secret-key-change-in-production")
MONGO_URI  = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB   = os.environ.get("MONGO_DB", "athleteai")
SCORING_MODE = os.environ.get("SCORING_MODE", "hybrid")  # hybrid | shadow | rule_only
MONGO_TLS_ALLOW_INVALID_CERTS = os.environ.get("MONGO_TLS_ALLOW_INVALID_CERTS", "false").lower() == "true"
MONGO_TLS_ALLOW_INVALID_HOSTNAMES = os.environ.get("MONGO_TLS_ALLOW_INVALID_HOSTNAMES", "false").lower() == "true"
MONGO_TLS_INSECURE = os.environ.get("MONGO_TLS_INSECURE", "false").lower() == "true"

client = db = None
status_update_task = None  # Background task for updating test statuses


async def update_all_test_statuses():
    """
    Background task that runs periodically to update test statuses based on current time.
    Updates test statuses from upcoming→active→completed based on scheduled date/time.
    This ensures both athlete and authority modules see consistent, auto-updated statuses.
    """
    global db
    if db is None:
        return
    
    try:
        now = datetime.now()  # Use local time, not UTC
        
        # Get all tests that are not manually marked as completed
        tests = await db.tests.find({"status": {"$nin": ["completed", "archived"]}}).to_list(None)
        
        for test in tests:
            try:
                scheduled_date = test.get("scheduled_date")
                start_time = test.get("start_time", "00:00")
                duration_minutes = test.get("duration_minutes", 60)
                
                if not scheduled_date:
                    continue
                
                # Parse scheduled datetime
                dt_str = f"{scheduled_date} {start_time}"
                scheduled_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
                end_dt = scheduled_dt + timedelta(minutes=duration_minutes)
                
                # Determine new status based on current time
                new_status = None
                if now < scheduled_dt:
                    new_status = "upcoming"
                elif now >= scheduled_dt and now <= end_dt:
                    new_status = "active"
                else:
                    new_status = "completed"
                
                # Update status in database if it changed
                if new_status and new_status != test.get("status"):
                    await db.tests.update_one(
                        {"_id": test["_id"]},
                        {"$set": {"status": new_status, "last_status_update": now}}
                    )
                    print(f"✓ Test '{test.get('name')}' status updated: {test.get('status')} → {new_status}")
            except Exception as e:
                print(f"⚠ Error updating test {test.get('_id')}: {e}")
    except Exception as e:
        print(f"⚠ Error in status update task: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    global client, db, status_update_task

    mongo_client_kwargs = {
        "serverSelectionTimeoutMS": 30000,
        "connectTimeoutMS": 30000,
        "socketTimeoutMS": 30000,
    }
    if MONGO_URI.startswith("mongodb+srv://"):
        mongo_client_kwargs.update({"tls": True, "server_api": ServerApi("1")})
        if certifi is not None:
            mongo_client_kwargs["tlsCAFile"] = certifi.where()
        # Emergency switches for restrictive networks; keep disabled in normal use.
        if MONGO_TLS_INSECURE or MONGO_TLS_ALLOW_INVALID_CERTS:
            mongo_client_kwargs["tlsAllowInvalidCertificates"] = True
        if MONGO_TLS_INSECURE or MONGO_TLS_ALLOW_INVALID_HOSTNAMES:
            mongo_client_kwargs["tlsAllowInvalidHostnames"] = True

    client = AsyncIOMotorClient(MONGO_URI, **mongo_client_kwargs)
    db = client[MONGO_DB]

    try:
        await client.admin.command("ping")
    except Exception as e:
        print("❌ MongoDB connection failed during startup")
        print(f"   URI host: {MONGO_URI.split('@')[-1] if '@' in MONGO_URI else MONGO_URI}")
        print(f"   Hint: verify Atlas Network Access/IP allowlist and TLS settings. ({e})")
        raise

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
    
    # Start background task for automatic test status updates
    status_update_task = asyncio.create_task(periodic_status_update())
    print(f"✅  MongoDB connected → {MONGO_URI}/{MONGO_DB}")
    print(f"📡  Open Compass and connect to: {MONGO_URI}")
    print(f"🔄  Background test status updater started (updates every 5 minutes)")
    
    yield
    
    # Cancel background task on shutdown
    if status_update_task:
        status_update_task.cancel()
        try:
            await status_update_task
        except asyncio.CancelledError:
            pass
    client.close()


async def periodic_status_update():
    """
    Continuously runs test status updates every 5 minutes.
    This ensures tests automatically transition between statuses (upcoming→active→completed).
    """
    while True:
        try:
            await asyncio.sleep(300)  # Wait 5 minutes between updates
            await update_all_test_statuses()
        except asyncio.CancelledError:
            print("🛑  Test status updater task cancelled")
            break
        except Exception as e:
            print(f"⚠ Error in periodic update: {e}")
            await asyncio.sleep(5)  # Short delay on error


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


def sanitize_analysis_doc(doc: dict) -> dict:
    """Clamp/normalize analysis numerics so UI never gets sentinel/corrupt values."""
    if not doc:
        return doc

    out = dict(doc)

    def safe_float(value, default=0.0):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    # Duration should be a practical non-negative value in seconds.
    duration = safe_float(out.get("duration_seconds", 0.0), 0.0)
    if duration < 0 or duration > 10 * 60 * 60:  # 10 hours upper bound for safety
        duration = 0.0
    out["duration_seconds"] = round(duration, 1)

    total_reps = safe_float(out.get("total_reps", 0.0), 0.0)
    out["total_reps"] = int(max(0, min(total_reps, 10000)))

    for key in ("avg_correctness_score", "rule_score", "ml_score", "hybrid_form_score", "confidence_score"):
        if key in out:
            out[key] = round(max(0.0, min(100.0, safe_float(out.get(key, 0.0), 0.0))), 1)

    return out


def _safe_percent(value, default=0.0):
    try:
        return max(0.0, min(100.0, float(value)))
    except (TypeError, ValueError):
        return float(default)


def _compute_badges(results: list[dict]) -> list[dict]:
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
            badges.append({"id": f"{ex}-focus", "title": f"{ex.replace('_', ' ').title()} Specialist", "description": f"Logged 5 or more {ex.replace('_', ' ')} sessions."})
    return badges


def _compute_progress_summary(results: list[dict], user_doc: dict | None = None) -> dict:
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

    trend = []
    for r in ordered[:10]:
        trend.append({
            "date": r.get("created_at"),
            "exercise_type": r.get("exercise_type"),
            "score": float(r.get("avg_correctness_score", 0) or 0),
            "reps": int(r.get("total_reps", 0) or 0),
            "fitness_level": r.get("fitness_level", "Unknown"),
        })

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


def _build_notifications(results: list[dict], registrations: list[dict], tests: list[dict]) -> list[dict]:
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
            notices.append({"type": "result_ready", "title": "Latest result ready", "message": f"Your {latest.get('exercise_type', 'recent')} result is available.", "timestamp": created_at.isoformat() if created_at else None})

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


def _csv_download(filename: str, headers: list[str], rows: list[list[object]]) -> Response:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return Response(
        content=buffer.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _pdf_escape(text: object) -> str:
    safe = str(text).encode("latin-1", "replace").decode("latin-1")
    return safe.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _wrap_pdf_line(text: str, limit: int = 92) -> list[str]:
    raw = str(text)
    if len(raw) <= limit:
        return [raw]
    chunks = []
    remaining = raw
    while len(remaining) > limit:
        split_at = remaining.rfind(" ", 0, limit)
        if split_at <= 18:
            split_at = limit
        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks


def _build_pdf(title: str, sections: list[tuple[str, list[str]]]) -> bytes:
    page_width, page_height = 612, 792
    margin_left = 40
    margin_top = 52
    line_height = 12
    max_lines_per_page = 48

    raw_lines: list[str] = []
    for heading, lines in sections:
        raw_lines.append(heading)
        raw_lines.extend(lines)
        raw_lines.append("")

    wrapped_lines: list[str] = []
    for line in raw_lines:
        if line == "":
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_pdf_line(line))

    pages = [wrapped_lines[i:i + max_lines_per_page] for i in range(0, len(wrapped_lines), max_lines_per_page)]
    if not pages:
        pages = [[title]]

    objects: list[str] = []
    page_count = len(pages)
    page_ids = [4 + i * 2 for i in range(page_count)]
    content_ids = [5 + i * 2 for i in range(page_count)]

    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(f"<< /Type /Pages /Kids [{' '.join(f'{pid} 0 R' for pid in page_ids)}] /Count {page_count} >>")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for idx, page_lines in enumerate(pages):
        stream_lines = [
            "BT",
            "/F1 12 Tf",
            f"{margin_left} {page_height - margin_top} Td",
            f"({_pdf_escape(title)}) Tj",
            "/F1 9 Tf",
        ]
        for line in page_lines:
            if line == "":
                stream_lines.append("0 -10 Td")
                continue
            stream_lines.append(f"0 -{line_height} Td")
            stream_lines.append(f"({_pdf_escape(line)}) Tj")
        stream_lines.append("ET")
        stream = "\n".join(stream_lines)
        content = f"<< /Length {len(stream.encode('utf-8'))} >>\nstream\n{stream}\nendstream"
        page_obj = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_ids[idx]} 0 R >>"
        )
        objects.append(page_obj)
        objects.append(content)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj_id, body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{obj_id} 0 obj\n".encode("utf-8"))
        pdf.extend(body.encode("utf-8"))
        pdf.extend(b"\nendobj\n")

    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("utf-8"))
    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF".encode("utf-8")
    )
    return bytes(pdf)


def _pdf_download(filename: str, title: str, sections: list[tuple[str, list[str]]]) -> Response:
    return Response(
        content=_build_pdf(title, sections),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


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
    goal_avg_score: Optional[float] = None
    goal_tests_per_week: Optional[int] = None
    goal_primary_exercise: Optional[str] = None


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
                       live_total_reps: Optional[int] = Form(None),
                       live_valid_reps: Optional[int] = Form(None),
                       live_form_accuracy: Optional[float] = Form(None),
                       live_feedback: Optional[str] = Form(None),
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

    # Optional live AI stats from browser-side pose tracking.
    live_total = int(live_total_reps or 0)
    live_valid = int(live_valid_reps or 0)
    live_total = max(0, live_total)
    live_valid = max(0, min(live_valid, live_total))
    try:
        live_form = float(live_form_accuracy) if live_form_accuracy is not None else None
    except (TypeError, ValueError):
        live_form = None
    if live_form is not None:
        # Supports values in either [0,1] or [0,100].
        if live_form <= 1.0:
            live_form = live_form * 100.0
        live_form = max(0.0, min(100.0, live_form))
    live_meta = {
        "total_reps": live_total,
        "valid_reps": live_valid,
        "form_accuracy": round(live_form, 1) if live_form is not None else None,
        "feedback": (live_feedback or "")[:280],
        "captured_at": now,
    }

    await db.test_sessions.insert_one({
        "_id": sid, "user_id": user["user_id"], "exercise_type": exercise_type,
        "video_path": str(vpath), "status": "processing",
        "test_id": test_id,
        "live_pose_input": live_meta,
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
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "pose_module"))
        from pose_analyzer import VideoProcessor
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: VideoProcessor().process_video(vpath, exercise_type))
        rd = result.__dict__
        session_doc = await db.test_sessions.find_one({"_id": sid}, {"live_pose_input": 1})
        live_pose_input = (session_doc or {}).get("live_pose_input") or {}

        await prog(90)
        rid = str(uuid.uuid4()); pm = rd.get("performance_metrics", {})

        # Build athlete-specific baseline for bounded personalization.
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

        # Rollout safety: in shadow mode we keep legacy/rule score user-facing,
        # while still storing hybrid score for offline comparison.
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

        await db.analysis_results.insert_one({
            "_id": rid, "session_id": sid, "user_id": user_id,
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
            "rule_score": round(c - random.uniform(1, 4), 1),
            "ml_score": round(c + random.uniform(1, 3), 1),
            "hybrid_form_score": round(c, 1),
            "confidence_score": round(random.uniform(70, 95), 1),
            "analysis_version": "v3.0-hybrid-rules",
            "rep_breakdown": [
                {"rep": i + 1, "quality_score": round(max(45, min(100, c + random.uniform(-8, 8))), 1), "faults": []}
                for i in range(min(10, r))
            ],
            "detailed_feedback": {
                "top_faults": [],
                "phase_scores": {"setup": c, "eccentric": c - 3, "bottom": c - 5, "concentric": c - 2, "finish": c - 1},
                "recommendations": ["Maintain consistent pace and full range on every rep."],
                "quality_flags": [],
                "usable_frame_rate": 0.9,
            },
            "summary": {"reps_per_minute": round(r * 2, 1), "analysis_version": "v3.0-hybrid-rules"},
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


@app.get("/api/sessions/{session_id}/video")
async def get_session_video(session_id: str, user=Depends(get_current_user)):
    """Stream a recorded test video for authorized users.

    Access:
    - athlete: own session only
    - admin: any session
    - authority: only sessions tied to tests they created
    """
    session = await db.test_sessions.find_one({"_id": session_id})
    if not session:
        raise HTTPException(404, "Session not found")

    role = user.get("role")
    uid = user.get("user_id")

    if role == "athlete":
        if session.get("user_id") != uid:
            raise HTTPException(403, "Access denied")
    elif role == "authority":
        test_id = session.get("test_id")
        if not test_id:
            raise HTTPException(403, "Access denied")
        test = await db.tests.find_one({"_id": test_id}, {"created_by": 1})
        if not test or test.get("created_by") != uid:
            raise HTTPException(403, "Access denied")
    elif role != "admin":
        raise HTTPException(403, "Access denied")

    video_path = session.get("video_path")
    if not video_path:
        raise HTTPException(404, "Video not available")
    path_obj = Path(video_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise HTTPException(404, "Video file not found")

    media_type = "video/mp4"
    suffix = path_obj.suffix.lower()
    if suffix == ".webm":
        media_type = "video/webm"
    elif suffix in (".mov", ".qt"):
        media_type = "video/quicktime"
    elif suffix == ".avi":
        media_type = "video/x-msvideo"

    return FileResponse(path_obj, media_type=media_type, filename=path_obj.name)


# ── Results ────────────────────────────────────────────────────────────────────

@app.get("/api/results")
async def my_results(user=Depends(get_current_user)):
    docs = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).limit(50).to_list(50)
    return [serialize(sanitize_analysis_doc(d)) for d in docs]

@app.get("/api/results/{result_id}")
async def get_result(result_id: str, user=Depends(get_current_user)):
    doc = await db.analysis_results.find_one({"_id": result_id})
    if not doc: raise HTTPException(404, "Not found")
    if doc["user_id"] != user["user_id"] and user.get("role") != "admin":
        raise HTTPException(403, "Access denied")
    return serialize(sanitize_analysis_doc(doc))

@app.get("/api/sessions/{session_id}/result")
async def session_result(session_id: str, user=Depends(get_current_user)):
    doc = await db.analysis_results.find_one({"session_id": session_id})
    if not doc: raise HTTPException(404, "No result yet")
    return serialize(sanitize_analysis_doc(doc))


@app.get("/api/progress")
async def athlete_progress(user=Depends(get_current_user)):
    results = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).to_list(200)
    user_doc = await db.users.find_one({"_id": user["user_id"]}, {"password_hash": 0})
    summary = _compute_progress_summary([sanitize_analysis_doc(d) for d in results], user_doc)
    return summary


@app.get("/api/notifications")
async def athlete_notifications(user=Depends(get_current_user)):
    results = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).to_list(200)
    registrations = await db.test_registrations.find({"user_id": user["user_id"]}).to_list(200)
    test_ids = [r.get("test_id") for r in registrations if r.get("test_id")]
    tests = await db.tests.find({"_id": {"$in": test_ids}}).to_list(200) if test_ids else []
    notifications = _build_notifications([sanitize_analysis_doc(d) for d in results], registrations, tests)
    return {"items": notifications, "count": len(notifications)}


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
    rows = []
    search = q.strip().lower() if q else ""
    for u in users:
        stats_doc = smap.get(str(u["_id"]), {})
        row = {"id": str(u["_id"]), "name": u["name"], "email": u["email"],
               "age": u.get("age"), "height_cm": u.get("height_cm"), "weight_kg": u.get("weight_kg"),
               "total_tests": stats_doc.get("total_tests", 0),
               "avg_score": round(stats_doc.get("avg_score", 0), 1) if stats_doc.get("avg_score") is not None else None,
               "last_test": stats_doc.get("last_test").isoformat() if isinstance(stats_doc.get("last_test"), datetime) else None}
        if search:
            haystack = f"{row['name']} {row['email']}".lower()
            if search not in haystack:
                continue
        rows.append(row)

    reverse = sort_dir.lower() != "asc"

    def athlete_sort_key(item: dict):
        key = sort_by or "name"
        if key in {"age", "height_cm", "weight_kg", "total_tests"}:
            return item.get(key) or 0
        if key == "avg_score":
            return item.get(key) or 0
        if key == "last_test":
            return item.get(key) or ""
        return str(item.get(key) or "").lower()

    rows.sort(key=athlete_sort_key, reverse=reverse)
    return rows

@app.get("/api/admin/authorities")
async def admin_authorities(
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    admin=Depends(require_admin),
):
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

    def authority_sort_key(item: dict):
        key = sort_by or "name"
        if key in {"tests_created", "total_participants"}:
            return item.get(key) or 0
        if key == "created_at":
            return item.get(key) or ""
        return str(item.get(key) or "").lower()

    rows.sort(key=authority_sort_key, reverse=reverse)
    return rows

@app.get("/api/admin/results")
async def admin_all_results(
    q: Optional[str] = None,
    exercise_type: Optional[str] = None,
    fitness_level: Optional[str] = None,
    cheat: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "desc",
    admin=Depends(require_admin),
):
    docs = await db.analysis_results.find({}, {"frame_analyses": 0}).limit(200).to_list(200)
    uid_list = list({d["user_id"] for d in docs})
    umap = {str(u["_id"]): u for u in
            await db.users.find({"_id": {"$in": uid_list}}, {"name": 1, "email": 1}).to_list(200)}
    results = []
    search = q.strip().lower() if q else ""
    for d in docs:
        s = serialize(sanitize_analysis_doc(d))
        u = umap.get(d["user_id"], {})
        s["athlete_name"] = u.get("name", "Unknown"); s["athlete_email"] = u.get("email", "")
        if exercise_type and s.get("exercise_type") != exercise_type:
            continue
        if fitness_level and s.get("fitness_level") != fitness_level:
            continue
        if cheat == "flagged" and not s.get("cheat_detected"):
            continue
        if cheat == "clean" and s.get("cheat_detected"):
            continue
        if search and search not in f"{s.get('athlete_name','')} {s.get('athlete_email','')} {s.get('exercise_type','')}".lower():
            continue
        results.append(s)

    reverse = sort_dir.lower() != "asc"

    def result_sort_key(item: dict):
        key = sort_by or "created_at"
        if key in {"avg_correctness_score", "total_reps", "jump_height_cm", "estimated_percentile", "confidence_score"}:
            return item.get(key) or 0
        if key == "athlete_name":
            return str(item.get("athlete_name") or "").lower()
        if key == "exercise_type":
            return str(item.get("exercise_type") or "").lower()
        if key == "created_at":
            return item.get(key) or ""
        return str(item.get(key) or "").lower()

    results.sort(key=result_sort_key, reverse=reverse)
    return results


@app.get("/api/results/export")
async def export_my_results(
    format: str = "csv",
    exercise_type: Optional[str] = None,
    fitness_level: Optional[str] = None,
    user=Depends(get_current_user),
):
    docs = await db.analysis_results.find(
        {"user_id": user["user_id"]}, {"frame_analyses": 0}
    ).sort("created_at", -1).to_list(200)
    rows = [serialize(sanitize_analysis_doc(d)) for d in docs]
    if exercise_type:
        rows = [r for r in rows if r.get("exercise_type") == exercise_type]
    if fitness_level:
        rows = [r for r in rows if r.get("fitness_level") == fitness_level]
    headers = ["Date", "Exercise", "Reps", "Form Score", "Level", "Grade", "Cheat", "Confidence", "Score"]
    csv_rows = [
        [
            r.get("created_at", ""),
            r.get("exercise_type", ""),
            r.get("total_reps", 0),
            r.get("avg_correctness_score", 0),
            r.get("fitness_level", ""),
            r.get("form_grade", ""),
            "Yes" if r.get("cheat_detected") else "No",
            r.get("confidence_score", 0),
            r.get("hybrid_form_score", r.get("avg_correctness_score", 0)),
        ]
        for r in rows
    ]
    if format.lower() == "pdf":
        sections = [
            ("Results Summary", [f"Total sessions: {len(rows)}"]),
            ("Records", [
                f"{r.get('created_at','')} | {r.get('exercise_type','')} | reps {r.get('total_reps',0)} | score {r.get('avg_correctness_score',0)} | grade {r.get('form_grade','')} | cheat {'yes' if r.get('cheat_detected') else 'no'}"
                for r in rows
            ]),
        ]
        return _pdf_download("my_results.pdf", "AthleteAI Results Export", sections)
    return _csv_download("my_results.csv", headers, csv_rows)


@app.get("/api/admin/results/export")
async def export_admin_results(
    q: Optional[str] = None,
    exercise_type: Optional[str] = None,
    fitness_level: Optional[str] = None,
    cheat: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "desc",
    format: str = "csv",
    admin=Depends(require_admin),
):
    docs = await admin_all_results(q=q, exercise_type=exercise_type, fitness_level=fitness_level, cheat=cheat, sort_by=sort_by, sort_dir=sort_dir, admin=admin)
    headers = ["Date", "Athlete", "Email", "Exercise", "Reps", "Form Score", "Level", "Grade", "Cheat", "Confidence", "Score"]
    csv_rows = [
        [
            r.get("created_at", ""),
            r.get("athlete_name", ""),
            r.get("athlete_email", ""),
            r.get("exercise_type", ""),
            r.get("total_reps", 0),
            r.get("avg_correctness_score", 0),
            r.get("fitness_level", ""),
            r.get("form_grade", ""),
            "Yes" if r.get("cheat_detected") else "No",
            r.get("confidence_score", 0),
            r.get("hybrid_form_score", r.get("avg_correctness_score", 0)),
        ]
        for r in docs
    ]
    if format.lower() == "pdf":
        sections = [
            ("Results Summary", [f"Total records: {len(docs)}"]),
            ("Records", [
                f"{r.get('created_at','')} | {r.get('athlete_name','')} | {r.get('exercise_type','')} | reps {r.get('total_reps',0)} | score {r.get('avg_correctness_score',0)} | cheat {'yes' if r.get('cheat_detected') else 'no'}"
                for r in docs
            ]),
        ]
        return _pdf_download("admin_results.pdf", "AthleteAI Admin Results Export", sections)
    return _csv_download("admin_results.csv", headers, csv_rows)


@app.get("/api/admin/ai-metrics")
async def admin_ai_metrics(days: int = 14, admin=Depends(require_admin)):
    """Operational monitoring for scorer quality, disagreement, and confidence drift."""
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
            "scoring_mode": SCORING_MODE,
            "daily_trend": [],
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
        "scoring_mode": SCORING_MODE,
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
    assigned_roster: Optional[str] = None
    target_emails: Optional[list[str]] = None
    template_id: Optional[str] = None


class UpdateTestRequest(BaseModel):
    name: Optional[str] = None
    sport: Optional[str] = None
    exercises: Optional[list] = None
    scheduled_date: Optional[str] = None
    start_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    description: Optional[str] = None
    max_participants: Optional[int] = None
    assigned_roster: Optional[str] = None
    target_emails: Optional[list[str]] = None


class TestTemplateRequest(BaseModel):
    name: str
    sport: str
    exercises: list
    duration_minutes: Optional[int] = 60
    description: Optional[str] = None
    max_participants: Optional[int] = None
    assigned_roster: Optional[str] = None
    target_emails: Optional[list[str]] = None


async def resolve_target_users(target_emails: Optional[list[str]]) -> list[str]:
    if not target_emails:
        return []
    cleaned = [email.strip().lower() for email in target_emails if email and email.strip()]
    if not cleaned:
        return []
    users = await db.users.find({"email": {"$in": cleaned}}, {"_id": 1}).to_list(500)
    return [str(u["_id"]) for u in users]


def require_authority(user=Depends(get_current_user)):
    if user.get("role") not in ("admin", "authority"):
        raise HTTPException(403, "Authority access required")
    return user


@app.post("/api/tests")
async def create_test(req: CreateTestRequest, user=Depends(require_authority)):
    creator = await db.users.find_one({"_id": user["user_id"]}, {"name": 1})
    template_doc = None
    if req.template_id:
        template_doc = await db.test_templates.find_one({"_id": req.template_id})
        if not template_doc:
            raise HTTPException(404, "Template not found")
    target_user_ids = await resolve_target_users(req.target_emails)
    template_payload = template_doc or {}
    tid = str(uuid.uuid4())
    await db.tests.insert_one({
        "_id": tid,
        "name": req.name or template_payload.get("name", "Untitled Test"),
        "sport": req.sport or template_payload.get("sport", "General"),
        "exercises": req.exercises or template_payload.get("exercises", []),
        "scheduled_date": req.scheduled_date,
        "start_time": req.start_time,
        "duration_minutes": req.duration_minutes or template_payload.get("duration_minutes", 60),
        "description": req.description if req.description is not None else template_payload.get("description"),
        "max_participants": req.max_participants if req.max_participants is not None else template_payload.get("max_participants"),
        "assigned_roster": req.assigned_roster if req.assigned_roster is not None else template_payload.get("assigned_roster"),
        "target_emails": req.target_emails if req.target_emails is not None else template_payload.get("target_emails", []),
        "target_user_ids": target_user_ids if target_user_ids else template_payload.get("target_user_ids", []),
        "created_by": user["user_id"],
        "created_by_name": creator.get("name", "") if creator else "",
        "status": "upcoming",
        "is_archived": False,
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
        now = datetime.now()  # Use local time, not UTC

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
async def list_tests(
    status: Optional[str] = None,
    q: Optional[str] = None,
    sort_by: Optional[str] = None,
    sort_dir: str = "asc",
    include_archived: bool = False,
    user=Depends(get_current_user),
):
    docs = await db.tests.find({}).to_list(200)
    search = q.strip().lower() if q else ""
    result = []
    for doc in docs:
        s = serialize(doc)
        computed_status = compute_test_status(doc)
        s["status"] = computed_status
        s["is_archived"] = bool(doc.get("is_archived", False))
        reg = await db.test_registrations.find_one({"test_id": str(doc["_id"]), "user_id": user["user_id"]})
        s["is_registered"] = reg is not None
        s["participant_count"] = await db.test_registrations.count_documents({"test_id": str(doc["_id"])})
        s["target_count"] = len(doc.get("target_user_ids", [])) if isinstance(doc.get("target_user_ids"), list) else 0

        if s["is_archived"] and not include_archived:
            continue
        if user.get("role") == "athlete":
            target_users = doc.get("target_user_ids") or []
            if target_users and user.get("user_id") not in target_users:
                continue

        if status and computed_status != status:
            continue
        if search:
            haystack = " ".join([
                str(s.get("name", "")),
                str(s.get("sport", "")),
                str(s.get("description", "")),
                str(s.get("created_by_name", "")),
                " ".join(str(e.get("type", e)).lower() if isinstance(e, dict) else str(e).lower() for e in (s.get("exercises") or [])),
            ]).lower()
            if search not in haystack:
                continue
        result.append(s)

    reverse = sort_dir.lower() != "asc"

    def test_sort_key(item: dict):
        key = sort_by or "scheduled_date"
        value = item.get(key)
        if key in {"participant_count", "duration_minutes"}:
            return int(value or 0)
        if key == "name":
            return str(value or "").lower()
        if key == "status":
            order = {"upcoming": 0, "active": 1, "completed": 2}
            return order.get(str(value or ""), 9)
        return str(value or "")

    result.sort(key=test_sort_key, reverse=reverse)
    return result


@app.get("/api/tests/{test_id}")
async def get_test(test_id: str, user=Depends(get_current_user)):
    doc = await db.tests.find_one({"_id": test_id})
    if not doc: raise HTTPException(404, "Test not found")
    if doc.get("is_archived") and user.get("role") == "athlete":
        raise HTTPException(404, "Test not found")
    target_users = doc.get("target_user_ids") or []
    if user.get("role") == "athlete" and target_users and user["user_id"] not in target_users:
        raise HTTPException(404, "Test not found")
    s = serialize(doc)
    # Compute real-time status
    s["status"] = compute_test_status(doc)
    s["is_archived"] = bool(doc.get("is_archived", False))
    reg = await db.test_registrations.find_one({"test_id": test_id, "user_id": user["user_id"]})
    s["is_registered"] = reg is not None
    s["participant_count"] = await db.test_registrations.count_documents({"test_id": test_id})
    s["target_count"] = len(doc.get("target_user_ids", [])) if isinstance(doc.get("target_user_ids"), list) else 0
    return s


@app.patch("/api/tests/{test_id}/status")
async def update_test_status(test_id: str, status: str, user=Depends(require_authority)):
    if status not in ("upcoming", "active", "completed"):
        raise HTTPException(400, "Invalid status")
    res = await db.tests.update_one({"_id": test_id}, {"$set": {"status": status}})
    if res.matched_count == 0: raise HTTPException(404, "Test not found")
    return {"success": True}


@app.patch("/api/tests/{test_id}")
async def update_test(test_id: str, req: UpdateTestRequest, user=Depends(require_authority)):
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")
    updates = {k: v for k, v in req.dict().items() if v is not None}
    if not updates:
        raise HTTPException(400, "No fields to update")
    if "target_emails" in updates:
        updates["target_user_ids"] = await resolve_target_users(updates.pop("target_emails"))
    await db.tests.update_one({"_id": test_id}, {"$set": updates})
    return {"success": True}


@app.patch("/api/tests/{test_id}/archive")
async def archive_test(test_id: str, archived: bool = True, user=Depends(require_authority)):
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")
    updates = {"is_archived": archived}
    if archived:
        updates.update({"status": "completed", "archived_at": datetime.utcnow(), "archived_by": user["user_id"]})
    else:
        updates.update({"archived_at": None, "archived_by": None})
    await db.tests.update_one({"_id": test_id}, {"$set": updates})
    return {"success": True, "is_archived": archived}


@app.delete("/api/tests/{test_id}")
async def soft_delete_test(test_id: str, user=Depends(require_authority)):
    doc = await db.tests.find_one({"_id": test_id})
    if not doc:
        raise HTTPException(404, "Test not found")
    await db.tests.update_one({"_id": test_id}, {"$set": {"is_archived": True, "status": "completed", "deleted_at": datetime.utcnow(), "deleted_by": user["user_id"]}})
    return {"success": True, "is_archived": True}


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


@app.get("/api/test-templates")
async def list_test_templates(user=Depends(require_authority)):
    filt = {} if user.get("role") == "admin" else {"created_by": user["user_id"]}
    docs = await db.test_templates.find(filt).sort("created_at", -1).to_list(100)
    return [serialize(d) for d in docs]


@app.post("/api/test-templates")
async def create_test_template(req: TestTemplateRequest, user=Depends(require_authority)):
    tid = str(uuid.uuid4())
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


@app.post("/api/test-templates/{template_id}/clone")
async def clone_test_template(template_id: str, overrides: Optional[CreateTestRequest] = None, user=Depends(require_authority)):
    template = await db.test_templates.find_one({"_id": template_id})
    if not template:
        raise HTTPException(404, "Template not found")
    req = overrides or CreateTestRequest(
        name=template.get("name", "Untitled Test"),
        sport=template.get("sport", "General"),
        exercises=template.get("exercises", []),
        scheduled_date=datetime.utcnow().date().isoformat(),
        start_time="09:00",
        duration_minutes=int(template.get("duration_minutes", 60) or 60),
        description=template.get("description"),
        max_participants=template.get("max_participants"),
        assigned_roster=template.get("assigned_roster"),
        target_emails=template.get("target_emails", []),
        template_id=template_id,
    )
    created = await create_test(req, user=user)
    return created


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
        has_video = bool(s.get("video_path")) and Path(s.get("video_path", "")).exists()
        if uid not in best or score > best[uid]["score"]:
            best[uid] = {
                "user_id": uid,
                "session_id": sid,
                "name": umap.get(uid, {}).get("name", "Unknown"),
                "email": umap.get(uid, {}).get("email", ""),
                "total_reps": result.get("total_reps", 0),
                "jump_height_cm": result.get("jump_height_cm"),
                "avg_correctness_score": result.get("avg_correctness_score", 0),
                "fitness_level": result.get("fitness_level", "Unknown"),
                "form_grade": result.get("form_grade", "N/A"),
                "cheat_detected": bool(result.get("cheat_detected", False)),
                "video_available": has_video,
                "score": score,
            }
    entries = sorted(best.values(), key=lambda x: x["score"], reverse=True)
    for i, e in enumerate(entries): e["rank"] = i + 1
    return entries


@app.get("/api/tests/{test_id}/leaderboard/export")
async def export_test_leaderboard(test_id: str, format: str = "csv", user=Depends(get_current_user)):
    entries = await test_leaderboard(test_id, user=user)
    headers = ["Rank", "Athlete", "Email", "Score", "Grade", "Reps", "Form Score", "Level", "Cheat"]
    csv_rows = [
        [
            e.get("rank", 0),
            e.get("name", ""),
            e.get("email", ""),
            e.get("score", 0),
            e.get("form_grade", ""),
            e.get("total_reps", 0),
            e.get("avg_correctness_score", 0),
            e.get("fitness_level", ""),
            "Yes" if e.get("cheat_detected") else "No",
        ]
        for e in entries
    ]
    if format.lower() == "pdf":
        sections = [
            ("Leaderboard Summary", [f"Entries: {len(entries)}"]),
            ("Rankings", [
                f"#{e.get('rank', 0)} | {e.get('name', '')} | score {e.get('score', 0)} | grade {e.get('form_grade', '')} | reps {e.get('total_reps', 0)} | cheat {'yes' if e.get('cheat_detected') else 'no'}"
                for e in entries
            ]),
        ]
        return _pdf_download(f"leaderboard_{test_id}.pdf", "AthleteAI Leaderboard Export", sections)
    return _csv_download(f"leaderboard_{test_id}.csv", headers, csv_rows)


@app.get("/api/tests/{test_id}/analytics")
async def test_analytics(test_id: str, user=Depends(require_authority)):
    """Analytics for a specific test: participation, completion, performance."""
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
        ex_type = ex.get("type") if isinstance(ex, dict) else ex
        ex_results = [r for r in results if r.get("exercise_type") == ex_type]
        ex_performance[ex_type] = {
            "count": len(ex_results),
            "avg_score": sum(r.get("avg_correctness_score", 0) for r in ex_results) / len(ex_results) if ex_results else 0,
            "avg_reps": sum(r.get("total_reps", 0) for r in ex_results) / len(ex_results) if ex_results else 0,
            "cheat_detected_count": sum(1 for r in ex_results if r.get("cheat_detected", False)),
        }
    
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


@app.get("/api/cohort-analytics")
async def cohort_analytics(user=Depends(require_authority)):
    """Aggregate performance by roster/cohort."""
    if user.get("role") != "admin":
        raise HTTPException(403, "Admin only")
    
    tests = await db.tests.find({"assigned_roster": {"$exists": True, "$ne": None}}).to_list(500)
    cohorts = {}
    
    for test in tests:
        roster = test.get("assigned_roster")
        if roster not in cohorts:
            cohorts[roster] = {
                "roster": roster,
                "tests": 0,
                "total_participants": 0,
                "avg_completion_rate": 0,
                "avg_score": 0,
            }
        
        registrations = await db.test_registrations.find({"test_id": str(test["_id"])}).to_list(500)
        sessions = await db.test_sessions.find({"test_id": str(test["_id"]), "status": "completed"}).to_list(500)
        
        cohorts[roster]["tests"] += 1
        cohorts[roster]["total_participants"] += len(registrations)
        
        if registrations:
            completion_rate = (len(sessions) / len(registrations)) * 100
        else:
            completion_rate = 0
        
        sid_list = [str(s["_id"]) for s in sessions]
        results = await db.analysis_results.find({"session_id": {"$in": sid_list}}).to_list(500)
        
        scores = [r.get("avg_correctness_score", 0) for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        cohorts[roster]["avg_completion_rate"] = (cohorts[roster]["avg_completion_rate"] + completion_rate) / 2
        cohorts[roster]["avg_score"] = (cohorts[roster]["avg_score"] + avg_score) / 2
    
    return {"cohorts": list(cohorts.values())}


@app.get("/api/results/{result_id}/exercise-metrics")
async def result_exercise_metrics(result_id: str, user=Depends(get_current_user)):
    """Detailed metrics for a single result."""
    result = await db.analysis_results.find_one({"_id": result_id})
    if not result:
        raise HTTPException(404, "Result not found")
    
    if user.get("role") == "athlete" and result.get("user_id") != user["user_id"]:
        raise HTTPException(403, "Cannot view other athlete's results")
    
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