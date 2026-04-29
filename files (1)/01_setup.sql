-- ============================================================
-- RAG Interna Revizija — Supabase SQL Setup
-- Pokrenuti jednom u SQL Editoru na Supabase dashboardu
-- ============================================================

-- Korak 1: Aktiviraj pgvector ekstenziju
CREATE EXTENSION IF NOT EXISTS vector;

-- Korak 2: Kreiraj tabelu za chunkove dokumenata
CREATE TABLE IF NOT EXISTS dokumenti (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    naziv_dokumenta  TEXT NOT NULL,
    kategorija       TEXT NOT NULL,
    izvor            TEXT,
    godina           TEXT,
    tip_dokumenta    TEXT,
    napomena         TEXT,
    datum_uploada    TEXT,
    chunk_index      INTEGER,
    ukupno_chunkova  INTEGER,
    tekst            TEXT NOT NULL,
    embedding        VECTOR(1024)  -- BAAI/bge-m3 dimenzija
);

-- Korak 3: Indeks za brzu cosine similarity pretragu
CREATE INDEX IF NOT EXISTS dokumenti_embedding_idx
    ON dokumenti
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- Korak 4: Indeks za metadata filtriranje
CREATE INDEX IF NOT EXISTS dokumenti_kategorija_idx
    ON dokumenti (kategorija);

CREATE INDEX IF NOT EXISTS dokumenti_naziv_idx
    ON dokumenti (naziv_dokumenta);

-- Korak 5: Funkcija za similarity pretragu (sa metadata filterom)
CREATE OR REPLACE FUNCTION pretrazi_dokumente(
    upit_embedding     VECTOR(1024),
    kategorija_filter  TEXT    DEFAULT NULL,
    top_k              INTEGER DEFAULT 5,
    score_threshold    FLOAT   DEFAULT 0.35
)
RETURNS TABLE (
    id               UUID,
    tekst            TEXT,
    naziv_dokumenta  TEXT,
    kategorija       TEXT,
    izvor            TEXT,
    godina           TEXT,
    tip_dokumenta    TEXT,
    napomena         TEXT,
    chunk_index      INTEGER,
    ukupno_chunkova  INTEGER,
    similarity       FLOAT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        id,
        tekst,
        naziv_dokumenta,
        kategorija,
        izvor,
        godina,
        tip_dokumenta,
        napomena,
        chunk_index,
        ukupno_chunkova,
        1 - (embedding <=> upit_embedding) AS similarity
    FROM dokumenti
    WHERE
        (kategorija_filter IS NULL OR kategorija = kategorija_filter)
        AND 1 - (embedding <=> upit_embedding) >= score_threshold
    ORDER BY embedding <=> upit_embedding
    LIMIT top_k;
$$;

-- Korak 6: Funkcija za listanje jedinstvenih dokumenata
CREATE OR REPLACE FUNCTION lista_dokumenata_unique()
RETURNS TABLE (
    naziv_dokumenta  TEXT,
    kategorija       TEXT,
    izvor            TEXT,
    godina           TEXT,
    tip_dokumenta    TEXT,
    napomena         TEXT,
    datum_uploada    TEXT,
    chunkova         BIGINT
)
LANGUAGE SQL STABLE
AS $$
    SELECT
        naziv_dokumenta,
        MAX(kategorija)      AS kategorija,
        MAX(izvor)           AS izvor,
        MAX(godina)          AS godina,
        MAX(tip_dokumenta)   AS tip_dokumenta,
        MAX(napomena)        AS napomena,
        MAX(datum_uploada)   AS datum_uploada,
        COUNT(*)             AS chunkova
    FROM dokumenti
    GROUP BY naziv_dokumenta
    ORDER BY MAX(datum_uploada) DESC, naziv_dokumenta;
$$;

-- Korak 7: Row Level Security (RLS) — sigurnost
ALTER TABLE dokumenti ENABLE ROW LEVEL SECURITY;

-- Dozvoli sve operacije sa anon ključem (za Streamlit app)
CREATE POLICY "Dozvoli sve za anon" ON dokumenti
    FOR ALL
    TO anon
    USING (true)
    WITH CHECK (true);

-- Provjera instalacije
SELECT 'Supabase setup završen!' AS status;
