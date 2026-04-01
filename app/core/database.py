"""
CAPA DE BASE DE DATOS
======================
¿Por qué SQLAlchemy async?

FastAPI es ASGI (async). Si usas un driver síncrono (psycopg2), cada query
BLOQUEA el event loop entero — es como si tu app solo pudiera atender una
petición a la vez. asyncpg + SQLAlchemy async permite que mientras esperas
la respuesta de Postgres, el event loop atiende otras peticiones.

Analogía: un camarero síncrono lleva el pedido a cocina y espera parado.
Un camarero async lleva el pedido, y mientras espera atiende otras mesas.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings


# Motor de base de datos
# pool_size: conexiones permanentes en el pool
# max_overflow: conexiones extra en picos de tráfico
# pool_pre_ping: verifica que la conexión sigue viva antes de usarla
#   (evita el error "connection closed" después de inactividad)
engine = create_async_engine(
    settings.async_database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.debug,  # Si debug=True, loguea todas las queries SQL
)

# Factory de sesiones
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    # expire_on_commit=False: los objetos no se "caducan" después del commit.
    # Por defecto SQLAlchemy invalida los objetos tras commit, forzando
    # una query extra para releerlos. Innecesario en APIs stateless.
)


class Base(DeclarativeBase):
    """Clase base para todos los modelos ORM."""
    pass


@asynccontextmanager
async def get_db_context() -> AsyncGenerator[AsyncSession, None]:
    """Context manager para usar en workers (Celery usa esto)."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency injection para FastAPI.

    Uso en los endpoints:
        async def mi_endpoint(db: AsyncSession = Depends(get_db)):
            ...

    FastAPI llama a get_db(), inyecta la sesión, y garantiza que
    se cierra al terminar la petición (incluso si hay un error).
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
