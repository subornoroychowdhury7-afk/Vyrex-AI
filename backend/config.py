"""
Configuration management for VYREX AI Render Backend.
Loads environment variables with sensible production defaults.
"""

import os
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).resolve().parent

# Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")

# CORS Origins (Allow Vercel frontend deployments)
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:8000,https://*.vercel.app"
).split(",")

# Remote GPU Worker (Optional: if user connects Render CPU to Modal/RunPod or local tunnel)
REMOTE_GPU_URL = os.getenv("REMOTE_GPU_URL", "")

# Upstream Fallback API Key (Optional: Groq/OpenRouter/HuggingFace API key if running in pure serverless mode)
FALLBACK_API_KEY = os.getenv("FALLBACK_API_KEY", "")
FALLBACK_API_BASE = os.getenv("FALLBACK_API_BASE", "https://api.groq.com/openai/v1")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "llama-3.1-8b-instant")
