-- ═══════════════════════════════════════════════════════════════════════
-- MIGRACIÓN: Tablas para evaluación RAGAS
--
-- QUÉ GUARDA:
--   eval_datasets   → conjuntos de preguntas+respuestas de referencia
--   eval_samples    → cada par (pregunta, respuesta_esperada) del dataset
--   eval_runs       → cada vez que ejecutas una evaluación
--   eval_results    → métricas por muestra de cada run
-- ═══════════════════════════════════════════════════════════════════════

-- Conjunto de preguntas de evaluación
CREATE TABLE IF NOT EXISTS eval_datasets (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Cada muestra del dataset: pregunta + respuesta de referencia (ground truth)
CREATE TABLE IF NOT EXISTS eval_samples (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id      UUID NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
    question        TEXT NOT NULL,
    ground_truth    TEXT NOT NULL,   -- respuesta correcta escrita por un humano
    document_ids    UUID[],          -- documentos relevantes para esta pregunta
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Cada ejecución de evaluación
CREATE TABLE IF NOT EXISTS eval_runs (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_id      UUID NOT NULL REFERENCES eval_datasets(id) ON DELETE CASCADE,
    status          TEXT NOT NULL DEFAULT 'running',  -- running | done | failed
    -- Métricas agregadas (media de todas las muestras)
    faithfulness        FLOAT,   -- ¿La respuesta está soportada por los contextos?
    answer_relevancy    FLOAT,   -- ¿La respuesta es relevante a la pregunta?
    context_precision   FLOAT,   -- ¿Los chunks recuperados son los correctos?
    context_recall      FLOAT,   -- ¿Se recuperaron todos los chunks relevantes?
    -- Metadata
    total_samples   INTEGER DEFAULT 0,
    error_msg       TEXT,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ
);

-- Resultado por muestra dentro de un run
CREATE TABLE IF NOT EXISTS eval_results (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id              UUID NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    sample_id           UUID NOT NULL REFERENCES eval_samples(id) ON DELETE CASCADE,
    -- Respuesta generada por el RAG
    generated_answer    TEXT,
    -- Chunks recuperados (para calcular context precision/recall)
    retrieved_contexts  JSONB,
    -- Métricas individuales para esta muestra
    faithfulness        FLOAT,
    answer_relevancy    FLOAT,
    context_precision   FLOAT,
    context_recall      FLOAT,
    -- Para debugging: qué falló
    error_msg           TEXT,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS eval_results_run_idx ON eval_results(run_id);
CREATE INDEX IF NOT EXISTS eval_samples_dataset_idx ON eval_samples(dataset_id);
CREATE INDEX IF NOT EXISTS eval_runs_dataset_idx ON eval_runs(dataset_id);
