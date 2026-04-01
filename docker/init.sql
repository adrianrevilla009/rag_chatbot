-- ═══════════════════════════════════════════════════════════════════════════
-- INICIALIZACIÓN DE LA BASE DE DATOS
--
-- Este script corre SOLO la primera vez que se crea el contenedor de Postgres.
-- Aquí activamos la extensión pgvector y creamos el esquema completo.
-- ═══════════════════════════════════════════════════════════════════════════

-- Activar la extensión vectorial
-- pgvector añade un tipo de dato "vector(N)" a PostgreSQL, donde N es la
-- dimensión. all-MiniLM-L6-v2 genera vectores de 384 dimensiones.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─── Tabla de documentos ──────────────────────────────────────────────────
-- Un "documento" es la unidad que el usuario sube (un PDF, un TXT, etc.)
-- Guardamos metadatos aquí: nombre, tipo, cuándo se subió, estado de proceso.
CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename    TEXT NOT NULL,
    file_type   TEXT NOT NULL,                    -- 'pdf', 'txt', 'docx', etc.
    file_size   INTEGER NOT NULL,                 -- bytes
    status      TEXT NOT NULL DEFAULT 'pending',  -- pending | processing | ready | error
    error_msg   TEXT,
    chunk_count INTEGER DEFAULT 0,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Tabla de chunks (fragmentos) ─────────────────────────────────────────
-- Aquí está el corazón del RAG. Un documento se divide en trozos pequeños
-- ("chunks") porque los LLMs tienen límite de contexto y porque la búsqueda
-- semántica funciona mejor con fragmentos específicos que con documentos enteros.
--
-- Cada chunk tiene:
--   - Su texto original
--   - Su embedding (vector de 384 floats que representa el significado)
--   - Metadata: de qué doc viene, qué posición ocupa, etc.
CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    -- ON DELETE CASCADE: si borras el documento, se borran sus chunks automáticamente

    content     TEXT NOT NULL,
    embedding   vector(384),
    -- 384 dimensiones = tamaño del modelo all-MiniLM-L6-v2

    chunk_index INTEGER NOT NULL,   -- posición del chunk dentro del documento
    page_number INTEGER,            -- útil para PDFs: en qué página estaba
    token_count INTEGER,            -- cuántos tokens tiene (para respetar el límite del LLM)

    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Índice HNSW para búsqueda vectorial ──────────────────────────────────
-- Sin índice, buscar el vector más similar requiere comparar con TODOS los
-- chunks (O(n)). Con HNSW (Hierarchical Navigable Small World) la búsqueda
-- es aproximada pero O(log n) — miles de veces más rápida.
--
-- cosine: usamos similitud coseno porque nuestro modelo de embeddings
-- fue entrenado con esta métrica. Otros modelos usan distancia euclidiana (l2).
CREATE INDEX IF NOT EXISTS chunks_embedding_idx
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
-- m=16: conexiones por nodo (más = más preciso, más memoria)
-- ef_construction=64: calidad del índice al construirlo (más = mejor, más lento)

-- ─── Tabla de jobs (trabajos async) ──────────────────────────────────────
-- Cuando el usuario sube un documento, creamos un job y respondemos
-- inmediatamente. El worker actualiza este registro mientras procesa.
CREATE TABLE IF NOT EXISTS jobs (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued | running | done | failed
    progress    INTEGER DEFAULT 0,               -- 0-100
    message     TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    updated_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ─── Tabla de conversaciones ──────────────────────────────────────────────
-- Guardamos el historial de chat para poder mantener contexto entre mensajes.
-- En un RAG empresarial esto es fundamental: "¿qué dijiste antes sobre X?"
CREATE TABLE IF NOT EXISTS conversations (
    id         UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS messages (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT NOT NULL,
    -- sources: JSON con los chunks usados para generar la respuesta
    -- Esto es clave en RAG productivo: siempre citas tus fuentes
    sources         JSONB,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Función para actualizar updated_at automáticamente
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_jobs_updated_at
    BEFORE UPDATE ON jobs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();
