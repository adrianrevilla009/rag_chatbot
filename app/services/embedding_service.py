"""
SERVICIO DE EMBEDDINGS
=======================
¿Qué es un embedding?

Un embedding es una representación numérica del significado de un texto.
El modelo convierte "¿Cómo funciona el motor?" en un vector de 384 números.
Textos con significado similar tienen vectores similares (cercanos en el espacio).

¿Por qué all-MiniLM-L6-v2?
- Solo 90MB de RAM (cabe en cualquier máquina)
- Muy rápido en CPU (no necesita GPU)
- Buena calidad para recuperación de información
- El más usado en producción para RAG con recursos limitados

El modelo se carga UNA vez (singleton) y se reutiliza en todas las peticiones.
Cargar un modelo de 90MB en cada petición sería catastrófico para la latencia.
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Singleton que gestiona el modelo de embeddings.

    ¿Por qué singleton?
    Cargar sentence-transformers tarda ~2 segundos y consume ~90MB de RAM.
    Si lo instancias en cada request, tu app se destruye.
    El singleton garantiza carga única al arrancar el worker.
    """

    def __init__(self):
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Carga lazy: el modelo solo se carga cuando se necesita por primera vez."""
        if self._model is None:
            logger.info("loading_embedding_model", model=settings.embedding_model)
            self._model = SentenceTransformer(settings.embedding_model)
            logger.info("embedding_model_loaded")
        return self._model

    def embed_text(self, text: str) -> list[float]:
        """
        Genera el embedding de un texto.

        normalize_embeddings=True: normaliza el vector a longitud 1.
        Necesario para usar similitud coseno correctamente.
        Sin normalización, vectores de textos largos tendrían magnitudes
        mayores y dominarían incorrectamente los resultados.
        """
        model = self._load_model()
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_batch(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Genera embeddings para múltiples textos de forma eficiente.

        ¿Por qué batch y no embed_text() en un loop?
        Los modelos de ML son más eficientes procesando varios textos juntos
        (aprovechan paralelismo interno). batch_size=32 es óptimo para CPU.
        """
        model = self._load_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
        )
        return [emb.tolist() for emb in embeddings]

    def compute_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Similitud coseno entre dos vectores. Útil para debugging."""
        a = np.array(vec1)
        b = np.array(vec2)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


@lru_cache()
def get_embedding_service() -> EmbeddingService:
    """Factory con caché. Devuelve siempre la misma instancia."""
    return EmbeddingService()
