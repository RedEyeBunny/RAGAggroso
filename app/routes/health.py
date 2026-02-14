from fastapi import APIRouter
from sqlalchemy import text
from app.database import SessionLocal
from app.services.embeddings import get_embedding
from openai import OpenAI
import os

router = APIRouter()

client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")

@router.get("/health")
def health_check():
    status = {
        "backend": "ok",
        "database": "unknown",
        "llm_embedding": "unknown",
        "llm_chat": "unknown"
    }

    # Database check
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        status["database"] = "connected"
    except Exception as e:
        status["database"] = f"error: {str(e)}"
    finally:
        db.close()

    # Embedding check
    try:
        get_embedding("health check")
        status["llm_embedding"] = "working"
    except Exception as e:
        status["llm_embedding"] = f"error: {str(e)}"

    # Chat completion check
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5
        )
        status["llm_chat"] = "working"
    except Exception as e:
        status["llm_chat"] = f"error: {str(e)}"

    return status
