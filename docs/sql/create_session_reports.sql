CREATE TABLE IF NOT EXISTS public.session_reports (
    report_sk SERIAL PRIMARY KEY,
    session_fingerprint TEXT UNIQUE NOT NULL,
    session_sk INTEGER,
    data DATE,
    selecao TEXT NOT NULL,
    genero TEXT,
    contexto TEXT NOT NULL,
    jogo TEXT,
    report_title TEXT,
    report_txt TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_session_reports_selecao_data
    ON public.session_reports (selecao, data DESC);

CREATE INDEX IF NOT EXISTS idx_session_reports_contexto
    ON public.session_reports (contexto);
