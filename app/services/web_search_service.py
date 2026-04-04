"""
WEB SEARCH SERVICE
===================
Búsqueda web con DuckDuckGo — sin API key, sin límites, sin registro.

¿Por qué DuckDuckGo y no Google/Bing?
- Google Custom Search: 100 queries/día gratis, luego de pago
- Bing Search API: requiere tarjeta
- DuckDuckGo: scraping libre, sin autenticación, resultados decentes

La librería `duckduckgo-search` hace scraping de DDG y devuelve
resultados estructurados (título, URL, snippet) listos para consumir.

LIMITACIÓN CONOCIDA:
DDG puede hacer rate limiting si se hacen muchas búsquedas seguidas.
En producción real usarías Tavily (1000/mes gratis) o SerpAPI.
Para aprender y desarrollo, DDG es más que suficiente.
"""

import asyncio
from functools import lru_cache

from duckduckgo_search import DDGS

from app.core.logging import get_logger

logger = get_logger(__name__)


class WebSearchService:

    def __init__(self, max_results: int = 4):
        self.max_results = max_results

    async def search(self, query: str) -> list[dict]:
        """
        Busca en DuckDuckGo y devuelve resultados estructurados.

        DDGS es síncrono, lo ejecutamos en un executor para no bloquear
        el event loop de FastAPI (asyncio.to_thread hace exactamente eso:
        corre una función síncrona en un thread pool sin bloquear async).

        Retorna lista de:
        {
            "title": str,
            "url": str,
            "snippet": str,   # fragmento de texto relevante
            "source": "web"
        }
        """
        logger.info("web_search_started", query=query)

        try:
            results = await asyncio.to_thread(self._sync_search, query)
            logger.info("web_search_done", query=query, results=len(results))
            return results
        except Exception as e:
            logger.warning("web_search_failed", query=query, error=str(e))
            return []

    def _sync_search(self, query: str) -> list[dict]:
        """Búsqueda síncrona — se ejecuta en thread pool."""
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(
                query,
                max_results=self.max_results,
                safesearch="moderate",
            ):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                    "source": "web",
                })
        return results

    def format_for_llm(self, results: list[dict], query: str) -> str:
        """
        Formatea los resultados web para incluirlos en el prompt del LLM.

        El LLM necesita el contexto bien estructurado para poder citarlo
        correctamente en su respuesta.
        """
        if not results:
            return f"No se encontraron resultados web para: {query}"

        parts = [f"=== Resultados web para: '{query}' ==="]
        for i, r in enumerate(results, 1):
            parts.append(f"\n[Fuente {i}] {r['title']}")
            parts.append(f"URL: {r['url']}")
            parts.append(r['snippet'])

        return "\n".join(parts)


@lru_cache()
def get_web_search_service() -> WebSearchService:
    return WebSearchService()
