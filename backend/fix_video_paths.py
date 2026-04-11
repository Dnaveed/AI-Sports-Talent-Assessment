"""Run this from the backend directory to diagnose orphaned video files."""
import asyncio
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.environ["MONGO_URI"]
MONGO_DB = os.environ["MONGO_DB"]
UPLOAD_DIR = Path(__file__).parent / "uploads"

async def main():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[MONGO_DB]

    # Find what sessions the orphaned files belong to
    orphaned_files = list(UPLOAD_DIR.iterdir())
    print(f"\nFiles in uploads/ ({len(orphaned_files)}):")
    for f in orphaned_files:
        stem = f.stem  # UUID without extension
        session = await db.test_sessions.find_one({"_id": stem})
        if session:
            print(f"  {f.name} → session found: user={session.get('user_id')}, exercise={session.get('exercise_type')}, path_in_db={session.get('video_path')}")
            # Fix the path
            await db.test_sessions.update_one(
                {"_id": stem},
                {"$set": {"video_path": str(f)}}
            )
            print(f"    ✓ Fixed path to: {f}")
        else:
            print(f"  {f.name} → NO matching session in DB")

    client.close()

asyncio.run(main())
