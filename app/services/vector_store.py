"""
VECTOR STORE — HYBRID SEARCH
==============================
Combina búsqueda vectorial (semántica) con BM25 (full-text) usando
Reciprocal Rank Fusion (RRF) para fusionar los rankings.

ARQUITECTURA:
    1. _vector_search()   → top-K por similitud coseno (pgvector)
    2. _bm25_search()     → top-K por BM25 (tsvector + GIN index en Postgres)
    3. _rrf_fusion()      → fusiona las dos listas en un único ranking RRF
    4. reranker.rerank()  → cross-encoder reordena el resultado final

POR QUÉ ASYNCPG DIRECTO Y NO SQLAlchemy text():
    SQLAlchemy text() con asyncpg tiene un problema conocido: mezcla parámetros
    nombrados (:param) con posicionales ($1) dependiendo del driver, y el cast
    ::vector en un parámetro interpolado junto a otros parámetros bindados
    rompe el parser de asyncpg.

    Solución definitiva: extraer la conexión asyncpg subyacente y ejecutar
    la query directamente con parámetros posicionales $1, $2... que asyncpg
    sí maneja correctamente. El embedding va como string y se castea en el SQL.

RRF (Reciprocal Rank Fusion):
    score_rrf(doc) = Σ 1/(k + rank_i)
    donde k=60 (constante estándar de la literatura, Cormack et al. 2009)
    y rank_i es la posición del documento en cada lista de resultados.

    Ventaja clave: no asume nada sobre la escala de los scores originales
    (coseno ≠ BM25). Solo usa la posición — robusto y sin hiperparámetros.
"""

import uuid

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.services.reranker_service import get_reranker_service

logger = get_logger(__name__)

# Parámetro k de RRF. 60 es el valor estándar de la literatura.
# Valores más bajos dan más peso a los top-1 de cada lista.
# Valores más altos suavizan las diferencias de ranking.
RRF_K = 60


class VectorStore:

    async def similarity_search(
        self,
        db: AsyncSession,
        query_embedding: list[float],
        query_text: str = "",
        top_k: int = None,
        min_score: float = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[dict]:
        """
        Hybrid search: vector + BM25 fusionados con RRF, luego reranking.

        Args:
            query_embedding: Vector del embedding de la query
            query_text:      Texto original de la query (para BM25 y reranker)
            top_k:           Chunks finales a devolver (tras reranking)
            min_score:       Score mínimo para búsqueda vectorial
            document_ids:    Filtrar por documentos específicos

        El método recupera top_k * 3 candidatos por cada método antes de
        fusionar, para dar más margen al RRF y al reranker.
        """
        top_k = top_k or settings.retrieval_top_k
        min_score = min_score if min_score is not None else settings.retrieval_min_score

        emb = "[" + ",".join(str(x) for x in query_embedding) + "]"

        # Extraer conexión asyncpg subyacente
        raw_conn = await db.connection()
        asyncpg_conn = await raw_conn.get_raw_connection()
        inner = asyncpg_conn.driver_connection

        # Candidatos ampliados: recuperamos más de los necesarios para
        # que RRF y reranker tengan con qué trabajar
        fetch_k = top_k * 3

        # ── 1. Búsqueda vectorial ──────────────────────────────────────
        vector_rows = await self._vector_search(inner, emb, min_score, fetch_k, document_ids)

        # Fallback sin min_score si no hay resultados
        if not vector_rows:
            logger.info("vector_search_fallback", reason="no_results_with_min_score")
            vector_rows = await self._vector_search(inner, emb, None, fetch_k, document_ids)

        # ── 2. Búsqueda BM25 (solo si hay texto) ─────────────────────
        bm25_rows = []
        if query_text.strip():
            bm25_rows = await self._bm25_search(inner, query_text, fetch_k, document_ids)

        # ── 3. RRF fusion ─────────────────────────────────────────────
        fused = self._rrf_fusion(vector_rows, bm25_rows, top_n=fetch_k)

        if not fused:
            return []

        # ── 4. Reranking con cross-encoder ───────────────────────────
        # Solo si hay texto de query (el reranker necesita la pregunta)
        if query_text.strip():
            reranker = get_reranker_service()
            fused = await reranker.rerank(
                query=query_text,
                chunks=fused,
                top_n=top_k,
            )
        else:
            fused = fused[:top_k]

        logger.info(
            "hybrid_search_done",
            vector_hits=len(vector_rows),
            bm25_hits=len(bm25_rows),
            final=len(fused),
        )

        return fused

    # ─── Búsqueda vectorial ───────────────────────────────────────────────────

    async def _vector_search(
        self,
        conn,
        emb: str,
        min_score: float | None,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[dict]:
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
            LIMIT ${len(params)}
        """

        rows = await conn.fetch(sql, *params)
        return [self._row_to_dict(row, score_field="score") for row in rows]

    # ─── Búsqueda BM25 ────────────────────────────────────────────────────────

    async def _bm25_search(
        self,
        conn,
        query_text: str,
        top_k: int,
        document_ids: list[uuid.UUID] | None,
    ) -> list[dict]:
        """
        Búsqueda full-text usando tsvector y ts_rank_cd.

        ts_rank_cd: variante de ts_rank que tiene en cuenta la densidad
        de los términos de búsqueda en el documento (cover density).
        Más representativa de BM25 que ts_rank a secas.

        plainto_tsquery: convierte texto libre en tsquery sin necesitar
        sintaxis especial. "machine learning" → 'machine' & 'learning'
        websearch_to_tsquery sería mejor pero requiere Postgres 11+.
        """
        params: list = []
        conditions = ["d.status = 'ready'", "c.content_tsv IS NOT NULL"]

        # Intentar con español primero, luego inglés como fallback
        # usando OR entre las dos variantes de la query
        params.append(query_text)
        tsquery_expr = f"""(
            plainto_tsquery('spanish', ${len(params)})
            || plainto_tsquery('english', ${len(params)})
        )"""
        conditions.append(f"c.content_tsv @@ {tsquery_expr}")

        if document_ids:
            params.append([str(d) for d in document_ids])
            conditions.append(f"c.document_id = ANY(${len(params)}::uuid[])")

        where_clause = " AND ".join(conditions)
        params.append(top_k)

        sql = f"""
            SELECT
                c.id,
                c.content,
                c.chunk_index,
                c.page_number,
                c.document_id,
                d.filename,
                ts_rank_cd(c.content_tsv, {tsquery_expr}) AS score
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE {where_clause}
            ORDER BY score DESC
            LIMIT ${len(params)}
        """

        try:
            rows = await conn.fetch(sql, *params)
            return [self._row_to_dict(row, score_field="score") for row in rows]
        except Exception as e:
            # Si la tsquery está vacía o mal formada, devolver vacío en lugar de error
            logger.warning("bm25_search_failed", error=str(e))
            return []

    # ─── RRF Fusion ───────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        vector_results: list[dict],
        bm25_results: list[dict],
        top_n: int,
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.

        Para cada chunk, suma 1/(k + rank) de cada lista donde aparece.
        Un chunk en posición 1 de ambas listas obtiene el score máximo.
        Un chunk que solo aparece en una lista obtiene la mitad.

        El resultado preserva todos los campos del chunk más el campo
        'hybrid_score' con el score RRF, y 'search_method' indicando
        si vino de ambas listas, solo vectorial, o solo BM25.
        """
        scores: dict[str, float] = {}
        chunks_by_id: dict[str, dict] = {}

        # Contribución de la búsqueda vectorial
        for rank, chunk in enumerate(vector_results, start=1):
            cid = chunk["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)
            if cid not in chunks_by_id:
                chunk["search_method"] = "vector"
                chunks_by_id[cid] = chunk

        # Contribución de BM25
        for rank, chunk in enumerate(bm25_results, start=1):
            cid = chunk["id"]
            if cid in scores:
                # Ya estaba en vectorial → aparece en ambas listas
                chunks_by_id[cid]["search_method"] = "hybrid"
            else:
                chunk["search_method"] = "bm25"
                chunks_by_id[cid] = chunk
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (RRF_K + rank)

        # Asignar hybrid_score y ordenar
        for cid, rrf_score in scores.items():
            chunks_by_id[cid]["hybrid_score"] = rrf_score

        sorted_chunks = sorted(
            chunks_by_id.values(),
            key=lambda c: c["hybrid_score"],
            reverse=True,
        )

        return sorted_chunks[:top_n]

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _row_to_dict(self, row, score_field: str = "score") -> dict:
        return {
            "id":          str(row["id"]),
            "content":     row["content"],
            "chunk_index": row["chunk_index"],
            "page_number": row["page_number"],
            "document_id": str(row["document_id"]),
            "filename":    row["filename"],
            "score":       float(row[score_field]),
        }

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
