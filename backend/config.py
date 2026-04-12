"""Configuration and environment variables."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Ensure .env values override any existing OS env vars (e.g., old MONGO_URI)
load_dotenv(override=True)

# ── File & Database ────────────────────────────────────────────────────────────
UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

SECRET_KEY = os.environ.get("SECRET_KEY", "athleteai-secret-key-change-in-production")
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.environ.get("MONGO_DB", "athleteai")

# ── Scoring & Feature Flags ────────────────────────────────────────────────────
SCORING_MODE = os.environ.get("SCORING_MODE", "hybrid")  # hybrid | shadow | rule_only

# ── MongoDB TLS Settings ──────────────────────────────────────────────────────
MONGO_TLS_ALLOW_INVALID_CERTS = os.environ.get("MONGO_TLS_ALLOW_INVALID_CERTS", "false").lower() == "true"
MONGO_TLS_ALLOW_INVALID_HOSTNAMES = os.environ.get("MONGO_TLS_ALLOW_INVALID_HOSTNAMES", "false").lower() == "true"
MONGO_TLS_INSECURE = os.environ.get("MONGO_TLS_INSECURE", "false").lower() == "true"

# ── Video Upload ──────────────────────────────────────────────────────────────
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/quicktime", "video/webm", "video/x-msvideo"}
MAX_VIDEO_SIZE = 200 * 1024 * 1024  # 200 MB

# ── Benchmarks for Test Scoring ─────────────────────────────────────────────
BENCHMARKS = {
    "pushup":        {"beginner": 10, "intermediate": 25, "advanced": 40},
    "squat":         {"beginner": 8,  "intermediate": 20, "advanced": 35},
    "situp":         {"beginner": 12, "intermediate": 28, "advanced": 45},
    "vertical_jump": {"beginner": 20, "intermediate": 40, "advanced": 60},
    "jumping_jack":  {"beginner": 15, "intermediate": 30, "advanced": 50},
    "lunge":         {"beginner": 8,  "intermediate": 18, "advanced": 30},
}
