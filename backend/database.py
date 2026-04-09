"""MongoDB connection and initialization."""
import asyncio
from contextlib import asynccontextmanager
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.server_api import ServerApi
from datetime import datetime
import os

try:
    import certifi
except ImportError:
    certifi = None

from config import MONGO_URI, MONGO_DB, MONGO_TLS_INSECURE, MONGO_TLS_ALLOW_INVALID_CERTS, MONGO_TLS_ALLOW_INVALID_HOSTNAMES

# Global database and client
client = None
db = None
status_update_task = None


async def update_all_test_statuses():
    """Background task that updates test statuses based on current time."""
    global db
    if db is None:
        return
    try:
        now = datetime.now()
        tests = await db.tests.find({"status": {"$nin": ["completed", "archived"]}}).to_list(None)
        for test in tests:
            try:
                from tests_module.utils import compute_test_status
                real_status = compute_test_status(test)
                if real_status != test.get("status"):
                    await db.tests.update_one({"_id": test["_id"]}, {"$set": {"status": real_status}})
            except Exception as e:
                print(f"⚠ Error updating test {test.get('_id')}: {e}")
    except Exception as e:
        print(f"⚠ Error in status update task: {e}")


async def periodic_status_update():
    """Continuously runs test status updates every 5 minutes."""
    while True:
        try:
            await asyncio.sleep(300)
            await update_all_test_statuses()
        except asyncio.CancelledError:
            print("🛑  Test status updater task cancelled")
            break
        except Exception as e:
            print(f"⚠ Error in periodic update: {e}")
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan context for startup and shutdown."""
    global client, db, status_update_task

    mongo_client_kwargs = {
        "serverSelectionTimeoutMS": 30000,
        "connectTimeoutMS": 30000,
        "socketTimeoutMS": 30000,
    }

    if MONGO_URI.startswith("mongodb+srv://"):
        mongo_client_kwargs.update({
            "tls": True, 
            "server_api": ServerApi("1"),
            "tlsAllowInvalidCertificates": True,
        })
        if certifi is not None:
            try:
                mongo_client_kwargs["tlsCAFile"] = certifi.where()
            except Exception:
                pass
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

    # Create indexes
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

    # Seed admin user
    from auth.utils import hash_password
    if not await db.users.find_one({"email": "admin@athleteai.com"}):
        await db.users.insert_one({
            "_id": "admin-001",
            "email": "admin@athleteai.com",
            "name": "System Admin",
            "password_hash": hash_password("admin123"),
            "role": "admin",
            "age": None,
            "weight_kg": None,
            "height_cm": None,
            "created_at": datetime.utcnow(),
            "last_login": None,
        })

    status_update_task = asyncio.create_task(periodic_status_update())
    print(f"✅  MongoDB connected → {MONGO_URI}/{MONGO_DB}")
    print(f"📡  Open Compass and connect to: {MONGO_URI}")
    print(f"🔄  Background test status updater started (updates every 5 minutes)")

    yield

    # Cleanup
    if status_update_task:
        status_update_task.cancel()
        try:
            await status_update_task
        except asyncio.CancelledError:
            pass
    client.close()


def get_db():
    """Get the database instance."""
    global db
    return db
