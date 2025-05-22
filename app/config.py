import os
from dotenv import load_dotenv

load_dotenv()

# 🔐 Gemini API Key from .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 💬 Available Gemini models (LLM)
AVAILABLE_MODELS = [
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5",
    "gemini-pro-v2",
    "gemini-pro-v1",
    "gemini-pro-v0",
    "gemini-pro",
    "gemini-2",
    "gemini-1.5",
    "gemini-1",
    "models/embedding-004",
    "gemini-embedding-001",
    "models/embedding-001",
]

# 🌟 Default model (used for dropdown preselect)
DEFAULT_MODEL = "gemini-2.5-flash-preview-05-20"

# 📎 Embedding model
GEMINI_EMBED_MODEL = "text-embedding-004"
