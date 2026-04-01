"""
ENDPOINTS DE DOCUMENTOS
========================
Aquí gestionamos el ciclo de vida de los documentos:
  POST /documents/ingest  → Subir un documento (async)
  GET  /documents         → Listar todos los documentos
  GET  /documents/{id}    → Ver detalles de un documento
  DELETE /documents/{id}  → Eliminar documento y sus chunks
  GET  /jobs/{id}         → Estado de un job de procesamiento
"""

import os
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    IngestResponse,
    JobResponse,
)
from app.core.config import settings
from app.core.database import get_db
from app.core.logging import get_logger
from app.models.models import Document, Job
from app.services.cache_service import get_cache_service
from app.workers.tasks import process_document

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_202_ACCEPTED)
async def ingest_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Sube un documento para indexarlo en el vector store.

    Retorna 202 Accepted (no 200 OK) porque el procesamiento es async.
    202 significa: "He recibido tu petición y la estoy procesando".
    El cliente debe consultar GET /jobs/{job_id} para saber cuándo termina.
    """
    # Validar extensión
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Tipo de archivo no soportado. Permitidos: {settings.allowed_extensions}"
        )

    # Validar tamaño
    content = await file.read()
    file_size = len(content)
    if file_size > settings.max_file_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"Archivo demasiado grande. Máximo: {settings.max_file_size_mb}MB"
        )

    # Crear registro en la DB
    document = Document(
        filename=file.filename,
        file_type=file_ext.lstrip("."),
        file_size=file_size,
        status="pending",
    )
    db.add(document)
    await db.flush()  # flush sin commit para obtener el ID generado

    # Crear el job asociado
    job = Job(document_id=document.id, status="queued", progress=0)
    db.add(job)
    await db.commit()
    await db.refresh(document)
    await db.refresh(job)

    # Guardar el fichero en disco temporalmente
    # En producción: subirías a S3/MinIO y pasarías la URL al worker
    upload_path = Path(settings.upload_dir) / f"{document.id}{file_ext}"
    upload_path.parent.mkdir(parents=True, exist_ok=True)

    with open(upload_path, "wb") as f:
        f.write(content)

    # Encolar el job en Celery
    # .delay() es el método de Celery para encolar una tarea async
    process_document.delay(
        document_id=str(document.id),
        file_path=str(upload_path),
        file_type=file_ext.lstrip("."),
    )

    logger.info("document_ingested", document_id=str(document.id), filename=file.filename)

    return IngestResponse(
        document_id=document.id,
        job_id=job.id,
        filename=file.filename,
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    db: AsyncSession = Depends(get_db),
    status_filter: str | None = None,
):
    """Lista todos los documentos indexados."""
    query = select(Document).order_by(Document.created_at.desc())

    if status_filter:
        query = query.where(Document.status == status_filter)

    result = await db.execute(query)
    documents = result.scalars().all()

    count_result = await db.execute(select(func.count(Document.id)))
    total = count_result.scalar()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Obtiene detalles de un documento específico."""
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Elimina un documento y todos sus chunks del vector store.

    ON DELETE CASCADE en la DB se encarga de borrar los chunks automáticamente.
    También invalidamos la caché para no servir respuestas obsoletas.
    """
    result = await db.execute(
        select(Document).where(Document.id == document_id)
    )
    document = result.scalar_one_or_none()

    if not document:
        raise HTTPException(status_code=404, detail="Documento no encontrado")

    await db.delete(document)
    await db.commit()

    # Invalidar caché
    cache = get_cache_service()
    await cache.invalidate_document(str(document_id))

    logger.info("document_deleted", document_id=str(document_id))


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """
    Consulta el estado de un job de procesamiento.

    El cliente hace polling a este endpoint cada 2-3 segundos mientras
    progress < 100. En una app más avanzada usarías WebSockets o SSE
    para que el servidor notifique al cliente (push vs pull).
    """
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job no encontrado")

    return JobResponse.model_validate(job)
