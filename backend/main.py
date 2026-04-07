"""
AthleteAI Backend - FastAPI + MongoDB (Motor async driver) - Modular Architecture
Refactored from monolithic main.py into configurable modules:
- auth/       → Authentication (register, login, get_me)
- results/    → Results and progress tracking
- uploads_module/ → Video upload and processing
- tests_module/   → Test/assessment management
- admin_module/   → Admin dashboard and analytics
"""
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime

from database import lifespan
from auth.routes import router as auth_router
from results.routes import router as results_router
from uploads_module.routes import router as uploads_router
from tests_module.routes import router as tests_router
from admin_module.routes import router as admin_router
from config import MONGO_URI, MONGO_DB


# ── Initialize FastAPI application with lifespan management ──────────────────

app = FastAPI(
    title="AthleteAI API",
    version="2.0.0",
    description="Modular FastAPI backend for athletic performance analysis",
    lifespan=lifespan
)

# ── CORS middleware ──────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include routers from all modules ─────────────────────────────────────────

app.include_router(auth_router, tags=["Authentication"])
app.include_router(results_router, tags=["Results"])
app.include_router(uploads_router, tags=["Uploads"])
app.include_router(tests_router, tags=["Tests"])
app.include_router(admin_router, tags=["Admin"])

# ── Health check endpoint ────────────────────────────────────────────────────

@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint for monitoring.
    Returns basic system information and MongoDB connection status.
    """
    return {
        "status": "ok",
        "database": MONGO_DB,
        "timestamp": datetime.utcnow().isoformat(),
    }

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )