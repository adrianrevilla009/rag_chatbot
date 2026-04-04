"""
MAIN.PY — PUNTO DE ENTRADA DE FASTAPI
=======================================
Aquí se monta la aplicación: middleware, routers, eventos de ciclo de vida.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

from app.api.routes import chat, documents, health
from app.core.config import settings
from app.core.logging import get_logger, setup_logging

setup_logging(settings.environment)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan: código que se ejecuta al arrancar y al parar la app.

    ¿Por qué lifespan y no @app.on_event("startup")?
    lifespan es el patrón moderno de FastAPI (on_event está deprecated).
    El código antes del `yield` es startup, después es shutdown.
    """
    # ── STARTUP ───────────────────────────────────────────────────────────
    logger.info("app_starting", environment=settings.environment)

    # Precargamos el modelo de embeddings al arrancar para que el primer
    # request no tenga cold start de 2 segundos.
    # NOTA: esto solo aplica a la API si necesitara hacer embeddings directamente.
    # El worker los precarga en su propio Dockerfile.

    logger.info("app_ready", model=settings.groq_model)

    yield

    # ── SHUTDOWN ──────────────────────────────────────────────────────────
    logger.info("app_shutting_down")
    from app.services.cache_service import get_cache_service
    cache = get_cache_service()
    await cache.close()


# Crear la app
app = FastAPI(
    title="RAG Chatbot API",
    description="""
API productiva para un chatbot con Retrieval-Augmented Generation (RAG).

## Flujo básico
1. **Subir documento**: `POST /documents/ingest`
2. **Esperar procesamiento**: `GET /documents/jobs/{job_id}` (polling)
3. **Preguntar**: `POST /chat`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",      # Swagger UI — accesible en localhost:8000/docs
    redoc_url="/redoc",    # ReDoc — alternativa más limpia a Swagger
)

# ─── Middleware ────────────────────────────────────────────────────────────────

# CORS: permite que el frontend (puerto 3000) llame a la API (puerto 8000)
# En producción restringirías los orígenes a tu dominio real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ──────────────────────────────────────────────────────────────────
# Cada router agrupa endpoints relacionados y les añade un prefijo.
# Esto mantiene el código organizado y la URL limpia.

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": settings.app_name,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health/ready",
    }
