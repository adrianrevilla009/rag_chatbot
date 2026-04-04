"""
SERVICIO DE CACHÉ
==================
¿Por qué cachear respuestas del LLM?

1. LATENCIA: Groq tarda 1-3 segundos. Redis responde en <1ms.
2. RATE LIMITS: El tier gratuito de Groq tiene límites. Cachear reduce
   el número de llamadas reales a la API.
3. COSTE: En producción con OpenAI, cachear puede ahorrar un 30-50% en costes.

¿Cómo funciona la caché semántica?
Generamos una key basada en un hash de la pregunta + los IDs de documentos
disponibles. Preguntas idénticas reutilizan la respuesta cacheada.

Limitación conocida: "¿Qué es la fotosíntesis?" y "Explícame la fotosíntesis"
son semánticamente iguales pero tienen keys distintas. Para resolver esto
se usa "semantic caching" (comparar embeddings de la pregunta con preguntas
cacheadas). Lo dejamos como mejora futura — es overkill para aprender RAG.
"""

import hashlib
import json
from typing import Any

import redis.asyncio as aioredis

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class CacheService:

    def __init__(self):
        self._client: aioredis.Redis | None = None

    async def _get_client(self) -> aioredis.Redis:
        """Conexión lazy a Redis."""
        if self._client is None:
            self._client = aioredis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
        return self._client

    def _make_key(self, question: str, document_ids: list[str] | None = None) -> str:
        """
        Genera una cache key determinista.

        Incluimos los document_ids porque la misma pregunta sobre documentos
        distintos puede tener respuestas distintas.
        """
        content = question.lower().strip()
        if document_ids:
            content += "|" + ",".join(sorted(document_ids))

        hash_val = hashlib.sha256(content.encode()).hexdigest()[:16]
        return f"rag:response:{hash_val}"

    async def get(self, question: str, document_ids: list[str] | None = None) -> dict | None:
        """Recupera respuesta cacheada si existe."""
        try:
            client = await self._get_client()
            key = self._make_key(question, document_ids)
            data = await client.get(key)

            if data:
                logger.info("cache_hit", key=key)
                return json.loads(data)

            logger.info("cache_miss", key=key)
            return None
        except Exception as e:
            # Si Redis falla, no rompemos la app — simplemente no usamos caché
            logger.warning("cache_get_error", error=str(e))
            return None

    async def set(
        self,
        question: str,
        response: dict,
        document_ids: list[str] | None = None,
        ttl: int | None = None,
    ) -> None:
        """Guarda respuesta en caché."""
        try:
            client = await self._get_client()
            key = self._make_key(question, document_ids)
            ttl = ttl or settings.cache_ttl_seconds

            await client.setex(key, ttl, json.dumps(response))
            logger.info("cache_set", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("cache_set_error", error=str(e))

    async def invalidate_document(self, document_id: str) -> None:
        """
        Invalida caché relacionado con un documento.

        Cuando un documento se actualiza o elimina, las respuestas
        cacheadas basadas en él pueden estar desactualizadas.
        En producción harías esto con tags de caché más sofisticados.
        """
        try:
            client = await self._get_client()
            # Borramos todas las keys del namespace rag:response:*
            # (estrategia conservadora: borra toda la caché)
            # En producción usarías Redis tags para borrar solo las afectadas
            keys = await client.keys("rag:response:*")
            if keys:
                await client.delete(*keys)
                logger.info("cache_invalidated", keys_deleted=len(keys))
        except Exception as e:
            logger.warning("cache_invalidate_error", error=str(e))

    async def close(self) -> None:
        if self._client:
            await self._client.close()


_cache_service: CacheService | None = None


def get_cache_service() -> CacheService:
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
