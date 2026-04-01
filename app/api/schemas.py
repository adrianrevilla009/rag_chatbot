"""
SCHEMAS PYDANTIC
=================
¿Por qué schemas separados de los modelos ORM?

Los modelos ORM (models.py) representan la estructura de la base de datos.
Los schemas Pydantic representan los contratos de la API (qué entra y qué sale).

Son distintos por diseño:
- El modelo ORM tiene `embedding` (vector de 384 floats) — nunca lo expones en la API.
- El schema de respuesta tiene `chunk_count` calculado — no existe como columna en la DB.
- Puedes cambiar la DB sin romper la API, y viceversa.

FastAPI usa estos schemas para:
1. Validar automáticamente los datos de entrada (400 Bad Request si algo falla)
2. Serializar la respuesta a JSON
3. Generar la documentación OpenAPI en /docs
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field, field_validator


# ─── Documents ────────────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    file_size: int
    status: str
    chunk_count: int
    created_at: datetime

    model_config = {"from_attributes": True}
    # from_attributes=True: permite crear el schema desde un objeto ORM
    # (antes se llamaba orm_mode=True en Pydantic v1)


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ─── Jobs ─────────────────────────────────────────────────────────────────────

class JobResponse(BaseModel):
    id: uuid.UUID
    document_id: uuid.UUID | None
    status: str   # queued | running | done | failed
    progress: int  # 0-100
    message: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pregunta para el chatbot"
    )
    conversation_id: uuid.UUID | None = Field(
        default=None,
        description="ID de conversación para mantener contexto. Si es None, se crea una nueva."
    )
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Limitar la búsqueda a estos documentos. Si es None, busca en todos."
    )

    @field_validator("question")
    @classmethod
    def question_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("La pregunta no puede estar vacía")
        return v.strip()


class SourceResponse(BaseModel):
    chunk_id: str
    filename: str
    page_number: int | None
    score: float
    excerpt: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    conversation_id: uuid.UUID
    cached: bool = False


# ─── Conversations ────────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    id: uuid.UUID
    messages: list[MessageResponse]
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── Ingest ───────────────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    document_id: uuid.UUID
    job_id: uuid.UUID
    filename: str
    message: str = "Documento recibido. Procesando en background."


# ─── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    services: dict[str, str]
