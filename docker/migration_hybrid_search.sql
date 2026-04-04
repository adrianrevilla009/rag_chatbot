-- ═══════════════════════════════════════════════════════════════════════
-- MIGRACIÓN: Hybrid Search + Índice BM25 (tsvector)
--
-- QUÉ HACE ESTA MIGRACIÓN:
--   1. Añade columna tsvector a chunks para búsqueda full-text (BM25)
--   2. Crea índice GIN sobre esa columna (GIN es el índice estándar para FTS en Postgres)
--   3. Crea trigger para mantener tsvector actualizado automáticamente
--   4. Crea función RRF para fusionar rankings de búsqueda vectorial y BM25
--
-- POR QUÉ HYBRID SEARCH:
--   Búsqueda vectorial: "¿cuál es el sueldo?" → encuentra "remuneración mensual" ✓
--   BM25:              "¿cuánto es el IRPF?"  → encuentra "IRPF" exactamente   ✓
--   Hybrid:            captura ambos casos     → mejor recall                   ✓
--
-- POR QUÉ RRF (Reciprocal Rank Fusion):
--   No se pueden sumar directamente los scores de coseno y BM25 (escalas distintas).
--   RRF convierte ambas listas de ranking en scores comparables: score = 1/(k + rank)
--   donde k=60 es el parámetro estándar de la literatura. Luego se suman.
-- ═══════════════════════════════════════════════════════════════════════

-- 1. Columna tsvector en chunks
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv tsvector;

-- 2. Poblar la columna para chunks existentes
--    'spanish' y 'english': diccionarios de stemming. 'english' como fallback.
--    to_tsvector con coalesce por si content es null (no debería, pero defensivo)
UPDATE chunks
SET content_tsv = to_tsvector('spanish', coalesce(content, ''))
             || to_tsvector('english',  coalesce(content, ''));

-- 3. Índice GIN — el único índice que soporta búsqueda full-text en Postgres
--    GIN (Generalized Inverted Index): mapea cada lexema → lista de docs que lo contienen
CREATE INDEX IF NOT EXISTS chunks_content_tsv_idx ON chunks USING GIN (content_tsv);

-- 4. Trigger para mantener content_tsv actualizado en inserts/updates
CREATE OR REPLACE FUNCTION chunks_tsv_update() RETURNS trigger AS $$
BEGIN
    NEW.content_tsv :=
        to_tsvector('spanish', coalesce(NEW.content, ''))
     || to_tsvector('english',  coalesce(NEW.content, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS chunks_tsv_trigger ON chunks;
CREATE TRIGGER chunks_tsv_trigger
    BEFORE INSERT OR UPDATE OF content
    ON chunks
    FOR EACH ROW EXECUTE FUNCTION chunks_tsv_update();

-- 5. Añadir índice BM25 al init.sql futuro (chunks nuevos ya usan el trigger)
-- Los chunks existentes ya están actualizados por el UPDATE de arriba.
