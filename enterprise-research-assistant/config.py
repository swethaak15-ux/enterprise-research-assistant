"""
config.py

Centralised configuration and environment-variable management for the
Enterprise Research & Report Generation AI Assistant.

Every external integration is optional except the LLM provider (Groq).
Optional integrations (Gmail) are detected at runtime so the
application can start and run correctly even when they are not configured.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# LLM (Groq - free tier)
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

# ---------------------------------------------------------------------------
# Embeddings (local / free - no API key required)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "chroma_db")
AGENT_MEMORY_DB = os.getenv("AGENT_MEMORY_DB", "agent_memory.db")
PROFILE_DB = os.getenv("PROFILE_DB", "client_profiles.db")
REPORTS_FOLDER = os.getenv("REPORTS_FOLDER", "generated_reports")

# ---------------------------------------------------------------------------
# Gmail integration (optional)
# ---------------------------------------------------------------------------
GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token_gmail.json")
DEFAULT_REPORT_RECIPIENT = os.getenv("DEFAULT_REPORT_RECIPIENT", "manager@company.com")


def is_gmail_configured() -> bool:
    """Gmail is considered configured when an OAuth client-secret file is present."""
    return os.path.exists(GMAIL_CREDENTIALS_FILE)


def ensure_directories() -> None:
    """Create runtime directories that are not shipped with the project."""
    for path in (UPLOAD_FOLDER, REPORTS_FOLDER):
        os.makedirs(path, exist_ok=True)


def require_groq_key() -> None:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file. "
            "Free API keys are available at https://console.groq.com/keys"
        )
