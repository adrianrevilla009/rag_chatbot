"""
WEB SEARCH SERVICE — TAVILY PRIMERO, DDG COMO FALLBACK
========================================================
Tavily (https://tavily.com) es el estándar de facto para búsqueda web
en agentes LLM. Diferencias clave con DuckDuckGo:

  TAVILY                          DUCKDUCKGO (scraping)
  ─────────────────────────────   ──────────────────────────────
  API oficial con auth            Scraping sin auth → rate limit
  Contenido completo de páginas   Solo snippets de 200 chars
  Diseñado para LLMs              Diseñado para humanos
  Resultados rankeados por LLM    Resultados de búsqueda raw
  1.000 búsquedas/mes gratis      Gratis pero poco fiable
  Latencia ~1s                    Latencia variable + fallos

CONFIGURACIÓN:
  1. Regístrate en https://tavily.com (gratis, sin tarjeta)
  2. Copia tu API key del dashboard
  3. Añade al .env:  TAVILY_API_KEY=tvly-xxxxxxxxxx
  4. Reinicia: docker compose restart api worker

Si TAVILY_API_KEY no está configurada o falla, cae automáticamente
a DuckDuckGo como fallback para no romper nada.
"""

import asyncio
import random
import time
from functools import lru_cache

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Rate limiting para DDG fallback
_last_ddg_search_time: float = 0
_DDG_MIN_INTERVAL = 2.0


class WebSearchService:

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self._tavily_client = None
        self._tavily_available = None  # None = no chequeado aún

    def _get_tavily_client(self):
        """Inicializa el cliente Tavily (lazy) si la key está configurada."""
        if self._tavily_available is False:
            return None
        if self._tavily_client is not None:
            return self._tavily_client

        api_key = getattr(settings, "tavily_api_key", "") or ""
        if not api_key or api_key.startswith("tvly-xxx"):
            logger.info("tavily_not_configured", msg="Usando DuckDuckGo como fallback")
            self._tavily_available = False
            return None

        try:
            from tavily import TavilyClient
            self._tavily_client = TavilyClient(api_key=api_key)
            self._tavily_available = True
            logger.info("tavily_initialized")
            return self._tavily_client
        except ImportError:
            logger.warning("tavily_not_installed", msg="Instala: pip install tavily-python")
            self._tavily_available = False
            return None
        except Exception as e:
            logger.warning("tavily_init_failed", error=str(e))
            self._tavily_available = False
            return None

    async def search(self, query: str) -> list[dict]:
        """
        Búsqueda web: Tavily primero, DDG como fallback automático.
        """
        logger.info("web_search_started", query=query)

        # ── Intento 1: Tavily ──────────────────────────────────────────
        client = self._get_tavily_client()
        if client:
            try:
                results = await asyncio.to_thread(self._tavily_search, client, query)
                if results:
                    logger.info("web_search_done", provider="tavily", query=query, results=len(results))
                    return results
            except Exception as e:
                logger.warning("tavily_search_failed", query=query, error=str(e)[:150])

        # ── Fallback: DuckDuckGo con retry ────────────────────────────
        logger.info("web_search_fallback_ddg", query=query)
        return await self._ddg_search_with_retry(query)

    def _tavily_search(self, client, query: str) -> list[dict]:
        """
        Búsqueda síncrona con Tavily (ejecutada en thread pool).

        search_depth="advanced": Tavily crawlea las páginas y extrae
        contenido completo, no solo snippets. Más lento (~2s) pero
        el contexto que llega al LLM es mucho más rico.

        include_answer=True: Tavily también genera una respuesta corta
        usando sus propios LLMs — útil como contexto adicional.
        """
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=self.max_results,
            include_answer=True,         # respuesta sintetizada de Tavily
            include_raw_content=False,   # el contenido ya viene en content
        )

        results = []

        # Incluir la respuesta sintetizada de Tavily si existe
        if response.get("answer"):
            results.append({
                "title": "Síntesis de búsqueda",
                "url": "",
                "snippet": response["answer"],
                "source": "tavily_answer",
            })

        for r in response.get("results", []):
            # Tavily devuelve `content` con el texto completo de la página
            # Truncamos a 1500 chars para no saturar el contexto del LLM
            content = r.get("content", r.get("snippet", ""))[:1500]
            results.append({
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": content,
                "source":  "tavily",
            })

        return results

    async def _ddg_search_with_retry(self, query: str) -> list[dict]:
        """DDG con rate limiting y retry exponencial."""
        global _last_ddg_search_time

        now = time.monotonic()
        elapsed = now - _last_ddg_search_time
        if elapsed < _DDG_MIN_INTERVAL:
            await asyncio.sleep(_DDG_MIN_INTERVAL - elapsed)

        for attempt in range(3):
            try:
                results = await asyncio.to_thread(self._ddg_sync_search, query)
                _last_ddg_search_time = time.monotonic()
                logger.info("web_search_done", provider="ddg", query=query,
                            results=len(results), attempt=attempt + 1)
                return results
            except Exception as e:
                wait = (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning("ddg_search_retry", query=query, attempt=attempt + 1,
                               wait=round(wait, 1), error=str(e)[:100])
                if attempt < 2:
                    await asyncio.sleep(wait)

        logger.warning("web_search_all_failed", query=query)
        return []

    def _ddg_sync_search(self, query: str) -> list[dict]:
        from duckduckgo_search import DDGS
        results = []
        with DDGS(timeout=10) as ddgs:
            for r in ddgs.text(query, max_results=self.max_results, safesearch="moderate"):
                results.append({
                    "title":   r.get("title", ""),
                    "url":     r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source":  "ddg",
                })
        return results

    def format_for_llm(self, results: list[dict], query: str) -> str:
        if not results:
            return (
                f"La búsqueda web para '{query}' no devolvió resultados. "
                f"Responde con tu conocimiento propio si puedes, indicando claramente "
                f"que es conocimiento de entrenamiento y puede no estar actualizado."
            )

        parts = [f"=== Resultados web para: '{query}' ==="]
        for i, r in enumerate(results, 1):
            parts.append(f"\n[Fuente {i}] {r['title']}")
            if r["url"]:
                parts.append(f"URL: {r['url']}")
            parts.append(r["snippet"])

        return "\n".join(parts)


@lru_cache(maxsize=1)
def get_web_search_service() -> WebSearchService:
    return WebSearchService()