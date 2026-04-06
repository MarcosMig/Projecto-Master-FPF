CREATE TABLE IF NOT EXISTS public.selecoes (
    selection_sk SERIAL PRIMARY KEY,
    codigo TEXT UNIQUE NOT NULL,
    escalao TEXT,
    genero TEXT,
    ativo BOOLEAN DEFAULT true,
    sort_order INTEGER,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

INSERT INTO public.selecoes (codigo, escalao, genero, ativo, sort_order)
VALUES
    ('AA M', 'AA', 'M', true, 1),
    ('AA F', 'AA', 'F', true, 2),
    ('U23 M', 'U23', 'M', true, 3),
    ('U23 F', 'U23', 'F', true, 4),
    ('U21 M', 'U21', 'M', true, 5),
    ('U21 F', 'U21', 'F', true, 6),
    ('U20 M', 'U20', 'M', true, 7),
    ('U20 F', 'U20', 'F', true, 8),
    ('U19 M', 'U19', 'M', true, 9),
    ('U19 F', 'U19', 'F', true, 10),
    ('U18 M', 'U18', 'M', true, 11),
    ('U18 F', 'U18', 'F', true, 12),
    ('U17 M', 'U17', 'M', true, 13),
    ('U17 F', 'U17', 'F', true, 14),
    ('U16 M', 'U16', 'M', true, 15),
    ('U16 F', 'U16', 'F', true, 16),
    ('U15 M', 'U15', 'M', true, 17),
    ('U15 F', 'U15', 'F', true, 18),
    ('U14 M', 'U14', 'M', true, 19),
    ('U14 F', 'U14', 'F', true, 20),
    ('U13 M', 'U13', 'M', true, 21),
    ('U13 F', 'U13', 'F', true, 22)
ON CONFLICT (codigo) DO UPDATE
SET
    escalao = EXCLUDED.escalao,
    genero = EXCLUDED.genero,
    ativo = EXCLUDED.ativo,
    sort_order = EXCLUDED.sort_order,
    updated_at = NOW();

INSERT INTO public.selecoes (codigo, escalao, genero, ativo, sort_order)
SELECT DISTINCT
    TRIM(a.selecao) AS codigo,
    SPLIT_PART(TRIM(a.selecao), ' ', 1) AS escalao,
    CASE
        WHEN RIGHT(TRIM(a.selecao), 1) IN ('M', 'F') THEN RIGHT(TRIM(a.selecao), 1)
        ELSE NULL
    END AS genero,
    true AS ativo,
    1000 + ROW_NUMBER() OVER (ORDER BY TRIM(a.selecao)) AS sort_order
FROM public.athletes a
WHERE COALESCE(TRIM(a.selecao), '') <> ''
ON CONFLICT (codigo) DO NOTHING;

SELECT
    s.codigo,
    s.escalao,
    s.genero,
    COUNT(a.athlete_sk) AS total_atletas
FROM public.selecoes s
LEFT JOIN public.athletes a
    ON a.selecao = s.codigo
GROUP BY s.codigo, s.escalao, s.genero
ORDER BY s.sort_order NULLS LAST, s.codigo;
