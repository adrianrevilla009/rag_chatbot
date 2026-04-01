"""
VECTOR STORE
=============
Búsqueda semántica con pgvector usando asyncpg directamente.

POR QUÉ ASYNCPG DIRECTO Y NO SQLAlchemy text():
SQLAlchemy text() con asyncpg tiene un problema conocido: mezcla parámetros
nombrados (:param) con posicionales ($1) dependiendo del driver, y el cast
::vector en un parámetro interpolado junto a otros parámetros bindados
rompe el parser de asyncpg.

Solución definitiva: extraer la conexión asyncpg subyacente y ejecutar
la query directamente con parámetros posicionales $1, $2... que asyncpg
sí maneja correctamente. El embedding va como string y se castea en el SQL.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class VectorStore:

    async def similarity_search(
        self,
        db: AsyncSession,
        query_embedding: list[float],
        top_k: int = None,
        min_score: float = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[dict]:
        """
        Búsqueda vectorial usando asyncpg directamente.

        Estrategia:
        1. Intentar con min_score
        2. Si no hay resultados (cross-idioma), repetir sin min_score
        """
        top_k = top_k or settings.retrieval_top_k
        min_score = min_score if min_score is not None else settings.retrieval_min_score

        # Embedding como string literal — asyncpg no puede castear parámetros a ::vector
        emb = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Extraer la conexión asyncpg subyacente desde SQLAlchemy
        # Esto nos da acceso directo a asyncpg con parámetros posicionales $1, $2...
        raw_conn = await db.connection()
        asyncpg_conn = await raw_conn.get_raw_connection()
        inner = asyncpg_conn.driver_connection

        rows = await self._run_query(inner, emb, min_score, top_k, document_ids)

        # Fallback sin min_score si no hay resultados (búsqueda cross-idioma)
        if not rows:
            logger.info("similarity_search_fallback", reason="no_results_with_min_score")
            rows = await self._run_query(inner, emb, None, top_k, document_ids)

        results = [
            {
                "id": str(row["id"]),
                "content": row["content"],
                "chunk_index": row["chunk_index"],
                "page_number": row["page_number"],
                "document_id": str(row["document_id"]),
                "filename": row["filename"],
                "score": float(row["score"]),
            }
            for row in rows
        ]

        logger.info("similarity_search_done", results=len(results), top_k=top_k)
        return results

    async def _run_query(
        self,
        conn,
        emb: str,
        min_score: float | None,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list:
        """
        Ejecuta la query vectorial con parámetros posicionales puros ($1, $2...).
        El embedding va interpolado como literal string en el SQL (seguro — es
        generado internamente, no viene del usuario).
        """
        params = []
        conditions = ["d.status = 'ready'"]

        if min_score is not None:
            params.append(min_score)
            conditions.append(f"1 - (c.embedding <=> '{emb}'::vector) >= ${len(params)}")

        if document_ids:
            params.append([str(d) for d in document_ids])
            conditions.append(f"c.document_id = ANY(${len(params)}::uuid[])")

        where_clause = " AND ".join(conditions)

        params.append(top_k)
        limit_param = f"${len(params)}"

        sql = f"""
            SELECT
                c.id,
                c.content,
                c.chunk_index,
                c.page_number,
                c.document_id,
                d.filename,
                1 - (c.embedding <=> '{emb}'::vector) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
            ORDER BY c.embedding <=> '{emb}'::vector
            LIMIT {limit_param}
        """

        return await conn.fetch(sql, *params)

    async def delete_document_chunks(
        self, db: AsyncSession, document_id: uuid.UUID
    ) -> int:
        from sqlalchemy import text
        result = await db.execute(
            text("DELETE FROM chunks WHERE document_id = :doc_id RETURNING id"),
            {"doc_id": str(document_id)}
        )
        deleted = len(result.fetchall())
        logger.info("chunks_deleted", count=deleted, document_id=str(document_id))
        return deleted


def get_vector_store() -> VectorStore:
    return VectorStore()
