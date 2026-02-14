import os
from fastapi import APIRouter, Depends
from openai import OpenAI
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.schemas import AnswerResponse, QuestionRequest
from app.services.embeddings import get_embedding

router = APIRouter()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@router.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest, db: Session = Depends(get_db)):
    question = payload.question

    if not question.strip():
        return {
            "answer": "Question cannot be empty.",
            "sources": []
        }

    query_embedding = get_embedding(question)

    results = db.execute(text("""
        SELECT id, content, document_id
        FROM document_chunks
        ORDER BY embedding <-> CAST(:embedding AS vector)
        LIMIT 3
    """), {"embedding": query_embedding}).fetchall()

    context = "\n\n".join([r[1] for r in results])

    prompt = f"""
Answer using only the context below.
If not found, say "Not found in documents."

Context:
{context}

Question:
{question}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": [
            {"chunk": r[1], "document_id": r[2]}
            for r in results
        ]
    }
