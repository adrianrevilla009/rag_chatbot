"""
RERANKER SERVICE
=================
Reordena los chunks recuperados por hybrid search usando un cross-encoder.

POR QUÉ RERANKING:
    El retrieval (vectorial + BM25) opera con representaciones independientes
    de pregunta y chunk. El cross-encoder los evalúa juntos — es mucho más
    preciso pero también mucho más lento, así que se aplica solo sobre
    el top-k del retrieval (no sobre toda la base de datos).

    Bi-encoder (embedding):  encode(pregunta) · encode(chunk) → score
    Cross-encoder (reranker): encode(pregunta + chunk juntos)  → score  ← más preciso

MODELO:
    cross-encoder/ms-marco-MiniLM-L-6-v2
    - Entrenado en MS MARCO (dataset de búsqueda de Microsoft)
    - 22M parámetros, corre bien en CPU (~50ms por batch de 10 chunks)
    - Output: logit (sin normalizar). Más alto = más relevante.
    - Tamaño: ~85MB al descargar

ALTERNATIVAS CLOUD (sin coste de inferencia local):
    - Cohere Rerank API: mejor calidad, $0.001/1k docs, sin GPU necesaria
    - Jina Reranker API: similar precio
    Para activar Cohere, cambiar _rerank_with_crossencoder → _rerank_with_cohere
    y añadir COHERE_API_KEY al .env
"""

import asyncio
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Umbral mínimo de score del reranker para incluir un chunk.
# El cross-encoder devuelve logits sin normalizar:
#   > 1    → muy relevante
#   -1..1  → relevante o dudoso
#   < -3   → claramente irrelevante
# Subimos a 0.0 para ser más estrictos: solo incluir chunks que el
# reranker considera positivamente relevantes para la pregunta.
# Con -2.0 pasaban chunks con score -11 (completamente irrelevantes).
RERANKER_SCORE_THRESHOLD = 0.0


class RerankerService:

    def __init__(self):
        self._model = None
        self._model_loaded = False

    def _load_model(self):
        """Carga el cross-encoder en memoria (lazy — solo cuando se necesita)."""
        if self._model_loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            model_name = getattr(settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            logger.info("reranker_loading", model=model_name)
            self._model = CrossEncoder(model_name, max_length=512)
            self._model_loaded = True
            logger.info("reranker_loaded", model=model_name)
        except Exception as e:
            logger.error("reranker_load_failed", error=str(e))
            self._model = None
            self._model_loaded = True  # no reintentar

    async def rerank(
        self,
        query: str,
        chunks: list[dict],
        top_n: int | None = None,
    ) -> list[dict]:
        """
        Reordena los chunks por relevancia a la query.

        Args:
            query:   Pregunta del usuario
            chunks:  Lista de chunks del hybrid search (con campo 'content')
            top_n:   Cuántos chunks devolver tras el reranking. None = todos.

        Returns:
            Lista de chunks enriquecida con campo 'rerank_score', ordenada
            de mayor a menor relevancia.
        """
        if not chunks:
            return chunks

        self._load_model()

        if self._model is None:
            # Si el reranker no está disponible, devolver chunks sin reordenar
            logger.warning("reranker_unavailable_using_original_order")
            return chunks[:top_n] if top_n else chunks

        # El cross-encoder es síncrono — ejecutarlo en un thread pool
        # para no bloquear el event loop de FastAPI
        loop = asyncio.get_event_loop()
        reranked = await loop.run_in_executor(
            None,
            self._rerank_sync,
            query,
            chunks,
            top_n,
        )
        return reranked

    def _rerank_sync(
        self,
        query: str,
        chunks: list[dict],
        top_n: int | None,
    ) -> list[dict]:
        """
        Ejecución síncrona del cross-encoder.
        Corre en thread pool para no bloquear async.
        """
        # Crear pares (query, chunk_content) para el cross-encoder
        pairs = [(query, chunk["content"]) for chunk in chunks]

        # Obtener scores — devuelve array de floats (logits)
        scores = self._model.predict(pairs)

        # Enriquecer cada chunk con su rerank_score
        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)

        # Filtrar por umbral mínimo (eliminar chunks claramente irrelevantes)
        filtered = [c for c in chunks if c["rerank_score"] >= RERANKER_SCORE_THRESHOLD]

        # Si el filtro elimina todo, caer de nuevo a los originales (sin umbral)
        if not filtered:
            filtered = chunks

        # Ordenar de mayor a menor score del reranker
        filtered.sort(key=lambda x: x["rerank_score"], reverse=True)

        logger.info(
            "reranker_done",
            input_chunks=len(chunks),
            output_chunks=len(filtered[:top_n] if top_n else filtered),
        )

        return filtered[:top_n] if top_n else filtered


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    return RerankerService()