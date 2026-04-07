"""Authentication utilities - password hashing and JWT token management."""
import hashlib
import hmac
import json
import base64
from typing import Optional
from datetime import datetime, timedelta
from config import SECRET_KEY


def hash_password(password: str) -> str:
    """Hash password using SHA256 with secret key."""
    return hashlib.sha256(f"{password}{SECRET_KEY}".encode()).hexdigest()


def create_token(user_id: str, role: str) -> str:
    """Create JWT token with expiration."""
    payload = json.dumps({
        "user_id": user_id,
        "role": role,
        "exp": (datetime.utcnow() + timedelta(hours=24)).isoformat()
    })
    sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return base64.b64encode(payload.encode()).decode() + "." + sig


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and extract payload."""
    try:
        b64, sig = token.rsplit(".", 1)
        payload = base64.b64decode(b64.encode()).decode()
        expected_sig = hmac.new(SECRET_KEY.encode(), payload.encode(), hashlib.sha256).hexdigest()
        
        if not hmac.compare_digest(sig, expected_sig):
            return None
        
        data = json.loads(payload)
        if datetime.fromisoformat(data["exp"]) > datetime.utcnow():
            return data
    except Exception:
        pass
    return None
