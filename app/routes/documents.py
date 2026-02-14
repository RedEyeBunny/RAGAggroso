from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import SessionLocal
from app.models import Document, DocumentChunk
from app.services.embeddings import get_embedding
from app.schemas import DocumentResponse
from typing import List

from app.schemas import DocumentResponse

router = APIRouter()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def chunk_text(text, size=800):
    return [text[i:i + size] for i in range(0, len(text), size)]

@router.get("/", response_model=List[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files allowed")

    content = (await file.read()).decode("utf-8")

    document = Document(filename=file.filename)
    db.add(document)
    db.commit()
    db.refresh(document)

    chunks = chunk_text(content)

    for idx, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        db_chunk = DocumentChunk(
            document_id=document.id,
            content=chunk,
            embedding=embedding,
            chunk_index=idx
        )
        db.add(db_chunk)

    db.commit()

    return {"message": "Uploaded successfully"}


@router.get("/documents", response_model=list[DocumentResponse])
def list_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()

from fastapi import Request
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="app/templates")

@router.get("/view")
def view_documents(request: Request, db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.uploaded_at.desc()).all()
    return templates.TemplateResponse("documents.html", {
        "request": request,
        "documents": docs
    })
