"""
ENDPOINTS DE EVALUACIÓN RAGAS
================================
POST /eval/datasets                     → Crear dataset
GET  /eval/datasets                     → Listar datasets
POST /eval/datasets/{id}/samples        → Añadir muestras (pregunta + ground truth)
GET  /eval/datasets/{id}/samples        → Ver muestras de un dataset
DELETE /eval/datasets/{id}              → Eliminar dataset

POST /eval/datasets/{id}/run            → Lanzar evaluación en background
GET  /eval/runs                         → Listar runs
GET  /eval/runs/{run_id}                → Detalle de un run (métricas globales)
GET  /eval/runs/{run_id}/results        → Métricas por muestra
"""

import asyncio
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.logging import get_logger
from app.services.evaluation_service import get_evaluation_service

logger = get_logger(__name__)
router = APIRouter(prefix="/eval", tags=["evaluation"])


# ─── Schemas ──────────────────────────────────────────────────────────────────

class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None


class SampleCreate(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: str = Field(..., min_length=1)
    document_ids: list[uuid.UUID] | None = None


class SamplesBulkCreate(BaseModel):
    samples: list[SampleCreate] = Field(..., min_length=1)


# ─── Datasets ─────────────────────────────────────────────────────────────────

@router.post("/datasets", status_code=201)
async def create_dataset(body: DatasetCreate, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
            INSERT INTO eval_datasets (name, description)
            VALUES (:name, :description)
            RETURNING id, name, description, created_at
        """),
        {"name": body.name, "description": body.description},
    )
    await db.commit()
    row = result.mappings().one()
    return dict(row)


@router.get("/datasets")
async def list_datasets(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
            SELECT
                d.id, d.name, d.description, d.created_at,
                COUNT(s.id) AS sample_count
            FROM eval_datasets d
            LEFT JOIN eval_samples s ON s.dataset_id = d.id
            GROUP BY d.id
            ORDER BY d.created_at DESC
        """)
    )
    return [dict(r) for r in result.mappings()]


@router.delete("/datasets/{dataset_id}", status_code=204)
async def delete_dataset(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("DELETE FROM eval_datasets WHERE id = :id RETURNING id"),
        {"id": str(dataset_id)},
    )
    await db.commit()
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Dataset no encontrado")


# ─── Samples ──────────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/samples", status_code=201)
async def add_samples(
    dataset_id: uuid.UUID,
    body: SamplesBulkCreate,
    db: AsyncSession = Depends(get_db),
):
    # Verificar que el dataset existe
    ds = await db.execute(
        text("SELECT id FROM eval_datasets WHERE id = :id"),
        {"id": str(dataset_id)},
    )
    if not ds.fetchone():
        raise HTTPException(status_code=404, detail="Dataset no encontrado")

    inserted = []
    for s in body.samples:
        doc_ids = [str(d) for d in s.document_ids] if s.document_ids else None
        r = await db.execute(
            text("""
                INSERT INTO eval_samples (dataset_id, question, ground_truth, document_ids)
                VALUES (:dataset_id, :question, :ground_truth, :document_ids)
                RETURNING id, question, ground_truth, document_ids, created_at
            """),
            {
                "dataset_id": str(dataset_id),
                "question": s.question,
                "ground_truth": s.ground_truth,
                "document_ids": doc_ids,
            },
        )
        inserted.append(dict(r.mappings().one()))

    await db.commit()
    return {"inserted": len(inserted), "samples": inserted}


@router.get("/datasets/{dataset_id}/samples")
async def list_samples(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
            SELECT id, question, ground_truth, document_ids, created_at
            FROM eval_samples
            WHERE dataset_id = :dataset_id
            ORDER BY created_at ASC
        """),
        {"dataset_id": str(dataset_id)},
    )
    return [dict(r) for r in result.mappings()]


@router.delete("/datasets/{dataset_id}/samples/{sample_id}", status_code=204)
async def delete_sample(
    dataset_id: uuid.UUID,
    sample_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("DELETE FROM eval_samples WHERE id = :id AND dataset_id = :dataset_id RETURNING id"),
        {"id": str(sample_id), "dataset_id": str(dataset_id)},
    )
    await db.commit()
    if not result.fetchone():
        raise HTTPException(status_code=404, detail="Muestra no encontrada")


# ─── Runs ─────────────────────────────────────────────────────────────────────

@router.post("/datasets/{dataset_id}/run", status_code=202)
async def start_run(dataset_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """
    Lanza una evaluación en background.
    Devuelve el run_id inmediatamente. Consultar GET /eval/runs/{run_id}
    para ver el progreso.
    """
    # Cargar muestras
    samples_result = await db.execute(
        text("""
            SELECT id::text, question, ground_truth, document_ids
            FROM eval_samples WHERE dataset_id = :dataset_id
        """),
        {"dataset_id": str(dataset_id)},
    )
    samples = [dict(r) for r in samples_result.mappings()]

    if not samples:
        raise HTTPException(
            status_code=400,
            detail="El dataset no tiene muestras. Añade al menos una pregunta con su ground truth."
        )

    # Crear el run
    run_result = await db.execute(
        text("""
            INSERT INTO eval_runs (dataset_id, status, total_samples)
            VALUES (:dataset_id, 'running', :total)
            RETURNING id
        """),
        {"dataset_id": str(dataset_id), "total": len(samples)},
    )
    await db.commit()
    run_id = str(run_result.scalar())

    # Lanzar en background sin bloquear la respuesta HTTP
    eval_service = get_evaluation_service()

    async def _run_in_background():
        # Nueva sesión de DB para el background task
        from app.core.database import AsyncSessionLocal
        async with AsyncSessionLocal() as bg_db:
            try:
                await eval_service.run_evaluation(bg_db, run_id, samples)
            except Exception as e:
                logger.error("eval_background_error", run_id=run_id, error=str(e))
                await bg_db.execute(
                    text("UPDATE eval_runs SET status='failed', error_msg=:err WHERE id=:id"),
                    {"err": str(e), "id": run_id},
                )
                await bg_db.commit()

    asyncio.create_task(_run_in_background())

    return {
        "run_id": run_id,
        "status": "running",
        "total_samples": len(samples),
        "message": f"Evaluación iniciada con {len(samples)} muestras. Consulta GET /eval/runs/{run_id} para el progreso.",
    }


@router.get("/runs")
async def list_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
            SELECT
                r.id, r.status, r.total_samples,
                r.faithfulness, r.answer_relevancy,
                r.context_precision, r.context_recall,
                r.created_at, r.finished_at,
                d.name AS dataset_name
            FROM eval_runs r
            JOIN eval_datasets d ON d.id = r.dataset_id
            ORDER BY r.created_at DESC
            LIMIT 50
        """)
    )
    return [dict(r) for r in result.mappings()]


@router.get("/runs/{run_id}")
async def get_run(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Métricas globales del run + progreso."""
    run_result = await db.execute(
        text("""
            SELECT
                r.id, r.status, r.total_samples, r.error_msg,
                r.faithfulness, r.answer_relevancy,
                r.context_precision, r.context_recall,
                r.created_at, r.finished_at,
                d.name AS dataset_name,
                COUNT(res.id) AS completed_samples
            FROM eval_runs r
            JOIN eval_datasets d ON d.id = r.dataset_id
            LEFT JOIN eval_results res ON res.run_id = r.id
            WHERE r.id = :run_id
            GROUP BY r.id, d.name
        """),
        {"run_id": str(run_id)},
    )
    row = run_result.mappings().one_or_none()
    if not row:
        raise HTTPException(status_code=404, detail="Run no encontrado")
    return dict(row)


@router.get("/runs/{run_id}/results")
async def get_run_results(run_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Métricas por muestra individual."""
    result = await db.execute(
        text("""
            SELECT
                res.id, res.generated_answer, res.retrieved_contexts,
                res.faithfulness, res.answer_relevancy,
                res.context_precision, res.context_recall,
                res.error_msg, res.created_at,
                s.question, s.ground_truth
            FROM eval_results res
            JOIN eval_samples s ON s.id = res.sample_id
            WHERE res.run_id = :run_id
            ORDER BY res.created_at ASC
        """),
        {"run_id": str(run_id)},
    )
    return [dict(r) for r in result.mappings()]
