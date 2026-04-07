"""Pydantic models for request/response validation."""
from pydantic import BaseModel
from typing import Optional, List


# ── Auth Models ────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    name: str
    password: str
    role: Optional[str] = "athlete"
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None


class LoginRequest(BaseModel):
    email: str
    password: str


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    weight_kg: Optional[float] = None
    height_cm: Optional[float] = None
    goal_avg_score: Optional[float] = None
    goal_tests_per_week: Optional[int] = None
    goal_primary_exercise: Optional[str] = None


# ── Test Models ────────────────────────────────────────────────────────────────

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
    target_emails: Optional[List[str]] = None
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
    target_emails: Optional[List[str]] = None


class TestTemplateRequest(BaseModel):
    name: str
    sport: str
    exercises: list
    duration_minutes: Optional[int] = 60
    description: Optional[str] = None
    max_participants: Optional[int] = None
    assigned_roster: Optional[str] = None
    target_emails: Optional[List[str]] = None
