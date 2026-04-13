"""Authentication utilities - password hashing and JWT token management."""
import hashlib
import hmac
import json
import base64
import smtplib
import secrets
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Optional
from datetime import datetime, timedelta
from config import SECRET_KEY, SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, FRONTEND_URL


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


def generate_reset_token() -> str:
    """Generate a secure random password reset token."""
    return secrets.token_urlsafe(32)


def send_reset_email(to_email: str, reset_token: str) -> bool:
    """Send password reset email. Returns True on success, False on failure."""
    if not SMTP_USER or not SMTP_PASS:
        # No email configured — log token for dev use
        print(f"[DEV] Password reset token for {to_email}: {reset_token}")
        print(f"[DEV] Reset URL: {FRONTEND_URL}/reset-password.html?token={reset_token}")
        return True

    reset_url = f"{FRONTEND_URL}/reset-password.html?token={reset_token}"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "AthleteAI — Reset Your Password"
    msg["From"] = f"AthleteAI <{SMTP_USER}>"
    msg["To"] = to_email

    html = f"""
    <div style="font-family:sans-serif;max-width:480px;margin:auto;background:#04080f;color:#e8f0fe;padding:2rem;border-radius:12px;border:1px solid #162034">
      <h2 style="color:#00d4ff;font-size:1.5rem;margin-bottom:0.5rem">Reset Your Password</h2>
      <p style="color:#5a7090;margin-bottom:1.5rem">Click the button below to reset your AthleteAI password. This link expires in 1 hour.</p>
      <a href="{reset_url}" style="display:inline-block;background:linear-gradient(135deg,#00d4ff,#0096b3);color:#000;font-weight:600;padding:0.75rem 1.5rem;border-radius:8px;text-decoration:none">Reset Password</a>
      <p style="color:#5a7090;font-size:0.8rem;margin-top:1.5rem">If you didn't request this, you can safely ignore this email.</p>
    </div>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        print(f"[ERROR] Failed to send reset email: {e}")
        return False
