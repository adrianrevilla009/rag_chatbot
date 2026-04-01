"""
HEALTH CHECK
=============
¿Por qué un endpoint de health check?

En producción, Kubernetes/Docker/el load balancer necesita saber si tu
app está viva y lista para recibir tráfico. Lo hace llamando a /health
cada 10-30 segundos. Si falla, saca el pod de la rotación.

Hay dos tipos de health checks:
- /health/live:  ¿El proceso está vivo? (liveness probe)
  Solo falla si el proceso está colgado o muerto.
  
- /health/ready: ¿Puede atender peticiones? (readiness probe)
  Falla si la DB o Redis no están disponibles.
  Kubernetes deja de enviarle tráfico hasta que se recupere.

Esta distinción evita que Kubernetes mate y reinicie un pod que solo
está esperando a que la DB esté disponible.
"""

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live")
async def liveness():
    """Liveness probe: el proceso está vivo."""
    return {"status": "alive"}


@router.get("/ready")
async def readiness():
    """
    Readiness probe: todos los servicios externos están disponibles.
    Retorna 200 si todo OK, 503 si algo falla.
    """
    import redis.asyncio as aioredis
    from fastapi import HTTPException

    services = {}

    # Verificar PostgreSQL
    try:
        async with AsyncSessionLocal() as db:
            await db.execute(text("SELECT 1"))
        services["postgres"] = "ok"
    except Exception as e:
        logger.error("postgres_health_check_failed", error=str(e))
        services["postgres"] = f"error: {str(e)[:50]}"

    # Verificar Redis
    try:
        client = aioredis.from_url(settings.redis_url)
        await client.ping()
        await client.close()
        services["redis"] = "ok"
    except Exception as e:
        logger.error("redis_health_check_failed", error=str(e))
        services["redis"] = f"error: {str(e)[:50]}"

    all_ok = all(v == "ok" for v in services.values())

    if not all_ok:
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "services": services}
        )

    return {
        "status": "ready",
        "version": "1.0.0",
        "model": settings.groq_model,
        "services": services,
    }
