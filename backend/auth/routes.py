"""Authentication routes - register, login, profile."""
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
import uuid

from database import get_db
from models import RegisterRequest, LoginRequest, UpdateProfileRequest
from auth.utils import hash_password, create_token
from dependencies import get_current_user

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/register")
async def register(req: RegisterRequest):
    """Register a new user (athlete or authority)."""
    db = get_db()
    uid = str(uuid.uuid4())
    role = req.role if req.role in ("athlete", "authority") else "athlete"

    try:
        await db.users.insert_one({
            "_id": uid,
            "email": req.email,
            "name": req.name,
            "password_hash": hash_password(req.password),
            "role": role,
            "age": req.age,
            "weight_kg": req.weight_kg,
            "height_cm": req.height_cm,
            "created_at": datetime.utcnow(),
            "last_login": None,
        })
    except Exception:
        raise HTTPException(409, "Email already registered")

    return {
        "token": create_token(uid, role),
        "user": {"id": uid, "email": req.email, "name": req.name, "role": role}
    }


@router.post("/login")
async def login(req: LoginRequest):
    """Login user with email and password."""
    db = get_db()
    user = await db.users.find_one({
        "email": req.email,
        "password_hash": hash_password(req.password)
    })

    if not user:
        raise HTTPException(401, "Invalid credentials")

    await db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"last_login": datetime.utcnow()}}
    )

    return {
        "token": create_token(str(user["_id"]), user["role"]),
        "user": {
            "id": str(user["_id"]),
            "email": user["email"],
            "name": user["name"],
            "role": user["role"]
        }
    }


@router.get("/me")
async def get_me(user=Depends(get_current_user)):
    """Get authenticated user profile."""
    db = get_db()
    doc = await db.users.find_one({"_id": user["user_id"]}, {"password_hash": 0})

    if not doc:
        raise HTTPException(404, "User not found")

    from results.utils import serialize
    return serialize(doc)


@router.put("/profile")
async def update_profile(req: UpdateProfileRequest, user=Depends(get_current_user)):
    """Update user profile and goals."""
    db = get_db()
    
    update_fields = {}
    if req.name is not None:
        update_fields["name"] = req.name
    if req.age is not None:
        update_fields["age"] = req.age
    if req.height_cm is not None:
        update_fields["height_cm"] = req.height_cm
    if req.weight_kg is not None:
        update_fields["weight_kg"] = req.weight_kg
    if req.goal_avg_score is not None:
        update_fields["goal_avg_score"] = req.goal_avg_score
    if req.goal_tests_per_week is not None:
        update_fields["goal_tests_per_week"] = req.goal_tests_per_week
    if req.goal_primary_exercise is not None:
        update_fields["goal_primary_exercise"] = req.goal_primary_exercise
    
    if update_fields:
        update_fields["updated_at"] = datetime.utcnow()
        await db.users.update_one(
            {"_id": user["user_id"]},
            {"$set": update_fields}
        )
    
    return {"success": True}
