"""Video upload routes."""
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends, BackgroundTasks
from pathlib import Path
from datetime import datetime
from uuid import uuid4

from database import get_db
from config import UPLOAD_DIR, ALLOWED_VIDEO_TYPES, MAX_VIDEO_SIZE
from dependencies import get_current_user
from uploads_module.processor import process_video_job

router = APIRouter(prefix="/api", tags=["uploads"])


@router.post("/sessions/upload")
async def upload_video(
    background_tasks: BackgroundTasks,
    exercise_type: str = "pushup",
    test_id: str | None = None,
    live_total_reps: int | None = Form(None),
    live_valid_reps: int | None = Form(None),
    live_form_accuracy: float | None = Form(None),
    live_feedback: str | None = Form(None),
    file: UploadFile = File(...),
    user=Depends(get_current_user),
):
    """Upload and process video."""
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Invalid type: {file.content_type}")

    content = await file.read()
    if len(content) > MAX_VIDEO_SIZE:
        raise HTTPException(413, "File too large")

    sid = str(uuid4())
    jid = str(uuid4())
    ext = Path(file.filename or "v.mp4").suffix or ".mp4"
    vpath = UPLOAD_DIR / f"{sid}{ext}"
    vpath.write_bytes(content)
    now = datetime.utcnow()

    # Process live pose data
    live_total = int(live_total_reps or 0)
    live_valid = int(live_valid_reps or 0)
    live_total = max(0, live_total)
    live_valid = max(0, min(live_valid, live_total))

    try:
        live_form = float(live_form_accuracy) if live_form_accuracy is not None else None
    except (TypeError, ValueError):
        live_form = None

    if live_form is not None:
        if live_form <= 1.0:
            live_form *= 100.0
        live_form = max(0.0, min(100.0, live_form))

    live_meta = {
        "total_reps": live_total,
        "valid_reps": live_valid,
        "form_accuracy": round(live_form, 1) if live_form is not None else None,
        "feedback": (live_feedback or "")[:280],
        "captured_at": now,
    }

    db = get_db()

    await db.test_sessions.insert_one({
        "_id": sid,
        "user_id": user["user_id"],
        "exercise_type": exercise_type,
        "video_path": str(vpath),
        "status": "processing",
        "test_id": test_id,
        "live_pose_input": live_meta,
        "created_at": now,
        "completed_at": None,
    })

    await db.processing_jobs.insert_one({
        "_id": jid,
        "session_id": sid,
        "status": "queued",
        "progress": 0.0,
        "error_message": None,
        "created_at": now,
        "updated_at": now,
    })

    background_tasks.add_task(process_video_job, jid, sid, str(vpath), exercise_type, user["user_id"])

    return {"session_id": sid, "job_id": jid, "status": "processing"}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str, user=Depends(get_current_user)):
    """Get job status."""
    db = get_db()
    doc = await db.processing_jobs.find_one({"_id": job_id})
    if not doc:
        raise HTTPException(404, "Job not found")

    from results.utils import serialize
    return serialize(doc)


@router.get("/sessions/{session_id}/video")
async def get_session_video(session_id: str, user=Depends(get_current_user)):
    """Stream recorded video."""
    from fastapi.responses import FileResponse

    db = get_db()
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
        if test_id:
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
