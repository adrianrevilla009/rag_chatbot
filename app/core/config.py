"""
CONFIGURACIÓN CENTRALIZADA
===========================
¿Por qué pydantic-settings?

En producción necesitas que la app falle INMEDIATAMENTE al arrancar si falta
una variable crítica (como GROQ_API_KEY), no cuando llega la primera petición.
pydantic-settings hace exactamente eso: valida todas las env vars al importar
el módulo. Si falta algo, el proceso muere con un error claro.

Alternativa naive: os.getenv("GROQ_API_KEY") disperso por todo el código.
Problemas: typos silenciosos, sin validación de tipos, sin valores por defecto
documentados, impossible de testear.
"""

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # extra="ignore": Docker Compose inyecta POSTGRES_USER, POSTGRES_PASSWORD
        # y POSTGRES_DB como variables de entorno (las necesita el contenedor de
        # PostgreSQL). Pydantic las ve y, por defecto, las rechaza si no están
        # definidas como campos. Con "ignore" simplemente las descarta.
    )

    # ─── App ──────────────────────────────────────────────────────────────
    app_name: str = "RAG Chatbot"
    environment: str = Field(default="development")
    debug: bool = Field(default=False)

    # ─── Base de datos ────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql://raguser:ragpassword@localhost:5432/ragdb"
    )

    @property
    def async_database_url(self) -> str:
        """asyncpg necesita el prefijo postgresql+asyncpg://"""
        return self.database_url.replace(
            "postgresql://", "postgresql+asyncpg://"
        )

    # ─── Redis ────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")

    # ─── Groq ─────────────────────────────────────────────────────────────
    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="openai/gpt-oss-120b")
    # llama3-groq-70b-8192-tool-use-preview: modelo de Groq fine-tuneado
    # específicamente para tool calling. llama-3.3-70b-versatile genera el
    # tool call en formato <function=...> incorrecto en vez de JSON estándar.
    # llama-3.3-70b-versatile: el modelo más capaz de Groq en tier gratuito.
    # Para más velocidad puedes usar "llama-3.1-8b-instant" (peor calidad).

    # ─── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384)
    # 384 dimensiones = tamaño de all-MiniLM-L6-v2.
    # Debe coincidir con vector(384) en init.sql.

    # ─── RAG / Chunking ───────────────────────────────────────────────────
    chunk_size: int = Field(default=512)
    # ¿Por qué 512? Balance entre:
    # - Muy pequeño (128): pierde contexto, respuestas incompletas
    # - Muy grande (2048): ruido, el embedding "diluye" el significado
    # 512 tokens es el sweet spot para documentos técnicos/empresariales.

    chunk_overlap: int = Field(default=64)
    # Overlap de 64 tokens entre chunks consecutivos.
    # ¿Por qué? Para no cortar frases a la mitad. Si un concepto importante
    # está en la frontera de dos chunks, el overlap garantiza que al menos
    # uno de los dos lo tiene completo.

    retrieval_top_k: int = Field(default=5)
    # Cuántos chunks recuperar por pregunta.
    # Más chunks = más contexto pero más tokens enviados al LLM (más lento y caro).
    # 5 es el estándar de la industria para documentos empresariales.

    retrieval_min_score: float = Field(default=0.1)
    # Similitud mínima para incluir un chunk (0 = cualquiera, 1 = idéntico).
    # 0.1 para tolerar búsquedas cross-idioma (pregunta en ES, doc en EN).
    # Con 0.3 las búsquedas cross-idioma fallan porque los embeddings de textos
    # en idiomas distintos tienen similitud coseno más baja aunque sean equivalentes.

    # ─── Cache ────────────────────────────────────────────────────────────
    cache_ttl_seconds: int = Field(default=3600)
    # TTL de 1 hora para respuestas cacheadas.
    # En producción ajustarías esto según la frecuencia de actualización
    # de tus documentos.

    # ─── Upload ───────────────────────────────────────────────────────────
    max_file_size_mb: int = Field(default=50)
    upload_dir: str = Field(default="/app/uploads")
    allowed_extensions: list[str] = Field(default=[".pdf", ".txt", ".docx", ".md"])


@lru_cache()
def get_settings() -> Settings:
    """
    lru_cache garantiza que Settings() se instancia UNA sola vez.
    Así no re-lees el .env en cada petición HTTP.
    """
    return Settings()


settings = get_settings()