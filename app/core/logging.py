"""
LOGGING ESTRUCTURADO
=====================
¿Por qué structlog y no print() o logging estándar?

En producción, los logs los procesa una herramienta (Datadog, ELK, CloudWatch).
Estas herramientas esperan JSON: {"level": "info", "event": "chat_request",
"user_id": "123", "latency_ms": 245}.

Con print() tienes: "chat_request user 123 took 245ms" — imposible de filtrar
y analizar automáticamente.

structlog genera JSON estructurado y añade contexto automáticamente
(timestamp, nivel, nombre del módulo, etc.).
"""

import logging
import sys

import structlog


def setup_logging(environment: str = "development") -> None:
    """Configura el sistema de logging según el entorno."""

    # En desarrollo: logs legibles por humanos con colores
    # En producción: JSON puro para que lo procesen las herramientas
    if environment == "production":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)
