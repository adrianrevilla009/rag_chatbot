"""
CELERY - PROCESAMIENTO ASÍNCRONO
==================================
¿Por qué Celery?

Procesar un documento implica:
  1. Leer el fichero (I/O)
  2. Dividirlo en chunks (CPU)
  3. Generar embeddings para cada chunk (CPU intensivo — puede ser 10-30s)
  4. Insertar en PostgreSQL (I/O)

Si hicieras esto dentro del endpoint HTTP, el usuario tendría que esperar
con la conexión abierta. Si la conexión se corta (timeout, red, etc.),
pierdes el trabajo. Con Celery:

  - La API responde en <100ms con un job_id
  - El worker procesa en background, aunque el usuario cierre el navegador
  - Si el worker falla, Celery puede reintentar automáticamente
  - Puedes escalar workers independientemente de la API
"""

from celery import Celery
from app.core.config import settings

# Creamos la app de Celery
# broker: Redis actúa de cola de mensajes (la API publica jobs aquí)
# backend: Redis también guarda el resultado de los jobs (para consultar estado)
celery_app = Celery(
    "rag_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["app.workers.tasks"],  # módulo donde están las tareas
)

celery_app.conf.update(
    # Serialización JSON (más seguro que pickle por defecto)
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # Timezone
    timezone="Europe/Madrid",
    enable_utc=True,

    # Si un task falla, reintentar hasta 3 veces con backoff
    task_acks_late=True,
    # acks_late=True: el mensaje no se elimina de la cola hasta que el task
    # termina correctamente. Si el worker muere a mitad, el mensaje vuelve
    # a la cola y otro worker lo retoma. Fundamental para no perder jobs.

    worker_prefetch_multiplier=1,
    # Prefetch=1: el worker solo coge un job a la vez.
    # Por defecto Celery coge 4. Para tasks pesados (embeddings), mejor 1
    # para no saturar la RAM del worker.

    task_track_started=True,
    # Registra cuándo empieza a ejecutarse un task (estado "STARTED")

    result_expires=86400,  # Los resultados se guardan 24h en Redis
)
