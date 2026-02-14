from pydantic import BaseModel
from datetime import datetime
from typing import List

class DocumentResponse(BaseModel):
    id: int
    filename: str
    uploaded_at: datetime

    class Config:
        from_attributes = True


class QuestionRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    document_id: int
    chunk: str


class AnswerResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]

class HealthResponse(BaseModel):
    backend: str
    database: str
    llm: str
