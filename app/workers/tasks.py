"""
TAREAS CELERY
==============
POR QUÉ USAMOS psycopg2 (SÍNCRONO) EN VEZ DE asyncpg AQUÍ:

Celery forkea procesos worker. asyncpg crea un event loop de asyncio al
conectarse por primera vez. Cuando Celery reutiliza ese proceso forkeado
para un segundo job, el event loop anterior ya está cerrado/inválido →
error "Future attached to a different loop".

La solución correcta: usar el driver SÍNCRONO (psycopg2) directamente
en los workers. El async (asyncpg) es para la API HTTP donde importa
manejar miles de conexiones concurrentes. En un worker que procesa
un job a la vez, el driver síncrono es más simple y más robusto.
"""

import os
import uuid

import psycopg2
import psycopg2.extras

from app.core.config import settings
from app.core.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)


def get_sync_conn():
    """
    Conexión síncrona a PostgreSQL con psycopg2.
    Creamos una conexión nueva por task para evitar problemas de estado
    entre jobs en el mismo proceso forkeado.
    """
    return psycopg2.connect(
        settings.database_url,
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


@celery_app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=30,
    name="process_document",
)
def process_document(self, document_id: str, file_path: str, file_type: str):
    """
    Task principal: procesa un documento completo de forma SINCRONA.

    Flujo:
    1. Extrae texto (pypdf, o Tesseract OCR si es PDF de imagen)
    2. Divide en chunks semanticos con overlap
    3. Genera embeddings en batch (MiniLM local, ~90MB RAM)
    4. Inserta chunks + embeddings en pgvector
    5. Marca el documento como "ready"
    """
    logger.info("task_started", document_id=document_id)
    conn = None

    try:
        conn = get_sync_conn()
        conn.autocommit = False
        cur = conn.cursor()

        # Paso 1: Marcar como "processing"
        cur.execute(
            "UPDATE documents SET status='processing', updated_at=NOW() WHERE id=%s",
            (document_id,)
        )
        cur.execute(
            "UPDATE jobs SET status='running', progress=10, message='Extrayendo texto...', updated_at=NOW() WHERE document_id=%s",
            (document_id,)
        )
        conn.commit()

        # Paso 2: Extraer texto del fichero
        logger.info("extracting_text", document_id=document_id, file_type=file_type)
        from app.services.document_processor import DocumentProcessor
        processor = DocumentProcessor()
        chunks_data = processor.process_file(file_path, file_type)

        if not chunks_data:
            raise ValueError("El documento no contiene texto extraible")

        cur.execute(
            "UPDATE jobs SET progress=40, message=%s, updated_at=NOW() WHERE document_id=%s",
            (f"Texto extraido. {len(chunks_data)} fragmentos. Generando embeddings...", document_id)
        )
        conn.commit()

        # Paso 3: Generar embeddings en batch
        logger.info("generating_embeddings", chunks=len(chunks_data))
        from app.services.embedding_service import get_embedding_service
        embedding_service = get_embedding_service()

        all_embeddings = []
        batch_size = 50
        total_batches = (len(chunks_data) + batch_size - 1) // batch_size

        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i + batch_size]
            texts = [c["content"] for c in batch]
            batch_embeddings = embedding_service.embed_batch(texts)
            all_embeddings.extend(batch_embeddings)

            batch_num = i // batch_size + 1
            progress = 40 + int((batch_num / total_batches) * 40)
            cur.execute(
                "UPDATE jobs SET progress=%s, updated_at=NOW() WHERE document_id=%s",
                (progress, document_id)
            )
            conn.commit()

        # Paso 4: Insertar chunks en pgvector
        logger.info("inserting_chunks", chunks=len(chunks_data))
        cur.execute(
            "UPDATE jobs SET progress=85, message='Guardando en base de datos...', updated_at=NOW() WHERE document_id=%s",
            (document_id,)
        )
        conn.commit()

        rows = []
        for chunk, embedding in zip(chunks_data, all_embeddings):
            rows.append((
                str(uuid.uuid4()),
                document_id,
                chunk["content"],
                str(embedding),
                chunk["chunk_index"],
                chunk.get("page_number"),
                chunk.get("token_count"),
            ))

        # execute_values: INSERT batch eficiente, una sola query para todos los chunks
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO chunks
                (id, document_id, content, embedding, chunk_index, page_number, token_count)
            VALUES %s
            """,
            rows,
            template="(%s, %s, %s, %s::vector, %s, %s, %s)",
        )

        inserted = len(rows)

        # Paso 5: Marcar como "ready"
        cur.execute(
            "UPDATE documents SET status='ready', chunk_count=%s, updated_at=NOW() WHERE id=%s",
            (inserted, document_id)
        )
        cur.execute(
            "UPDATE jobs SET status='done', progress=100, message=%s, updated_at=NOW() WHERE document_id=%s",
            (f"Listo. {inserted} fragmentos indexados correctamente.", document_id)
        )
        conn.commit()

        try:
            os.remove(file_path)
        except Exception:
            pass

        logger.info("task_completed", document_id=document_id, chunks=inserted)

    except Exception as exc:
        if conn:
            try:
                conn.rollback()
                cur = conn.cursor()
                cur.execute(
                    "UPDATE documents SET status='error', error_msg=%s, updated_at=NOW() WHERE id=%s",
                    (str(exc)[:500], document_id)
                )
                cur.execute(
                    "UPDATE jobs SET status='failed', message=%s, updated_at=NOW() WHERE document_id=%s",
                    (str(exc)[:500], document_id)
                )
                conn.commit()
            except Exception:
                pass

        logger.error("task_failed", document_id=document_id, error=str(exc))
        raise self.retry(exc=exc, countdown=2 ** self.request.retries * 30)

    finally:
        if conn:
            conn.close()
