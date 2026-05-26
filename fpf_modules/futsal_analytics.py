from __future__ import annotations

from datetime import date

import pandas as pd

from .futsal_parquet_store import (
    PHYSICAL_COLUMNS,
    latest_records_by_athlete,
    read_athlete_pathway,
    read_athletes,
    read_physical_records,
    read_technical_records,
)


METRIC_CATALOG = [
    {"source": "physical", "key": "peso_kg", "label": "Peso (Composicao Corporal)", "group": "Antropometria", "unit": "kg", "direction": "higher"},
    {"source": "physical", "key": "altura_cm", "label": "Altura (Estrutura)", "group": "Antropometria", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "envergadura_cm", "label": "Envergadura (Estrutura)", "group": "Antropometria", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "comprimento_perna_cm", "label": "Comprimento Perna (Estrutura)", "group": "Antropometria", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "altura_sentada_cm", "label": "Altura Sentada (Estrutura)", "group": "Antropometria", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "salto_maturacional", "label": "Salto Maturacional (Maturacao)", "group": "Antropometria", "unit": "anos", "direction": "higher"},
    {"source": "physical", "key": "sprint_10m_s", "label": "Sprint 10m (Aceleracao)", "group": "Fisico", "unit": "s", "direction": "lower"},
    {"source": "physical", "key": "sprint_20m_s", "label": "Sprint 20m (Velocidade)", "group": "Fisico", "unit": "s", "direction": "lower"},
    {"source": "physical", "key": "teste_505_esq_s", "label": "505 Esq (Agilidade)", "group": "Fisico", "unit": "s", "direction": "lower"},
    {"source": "physical", "key": "teste_505_dir_s", "label": "505 Dir (Agilidade)", "group": "Fisico", "unit": "s", "direction": "lower"},
    {"source": "physical", "key": "sj_altura_cm", "label": "SJ Altura (Potencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "cmj_altura_cm", "label": "CMJ Altura (Potencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "dj_altura_cm", "label": "DJ Altura (Potencia Reativa)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "dj_rsi", "label": "DJ RSI (Reatividade)", "group": "Fisico", "unit": "", "direction": "higher"},
    {"source": "physical", "key": "dj_rsi_mod_mps", "label": "DJ RSI Mod (Reatividade)", "group": "Fisico", "unit": "m/s", "direction": "higher"},
    {"source": "physical", "key": "dj_contacto_ms", "label": "DJ Contacto (Reatividade)", "group": "Fisico", "unit": "ms", "direction": "lower"},
    {"source": "physical", "key": "j10_rsi_10_5", "label": "10J RSI 10-5 (Resistencia Reativa)", "group": "Fisico", "unit": "", "direction": "higher"},
    {"source": "physical", "key": "j10_cmj_cm", "label": "10J CMJ (Potencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "j10_media_saltos_cm", "label": "10J Media (Resistencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "j10_maximo_cm", "label": "10J Maximo (Potencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "j10_minimo_cm", "label": "10J Minimo (Resistencia)", "group": "Fisico", "unit": "cm", "direction": "higher"},
    {"source": "physical", "key": "indice_fadiga_10j_pct", "label": "Indice Fadiga 10J (Resistencia)", "group": "Fisico", "unit": "%", "direction": "lower"},
    {"source": "technical", "key": "um_x_um_ofensivo_score", "label": "1x1 Ofensivo (Tecnica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "um_x_um_defensivo_score", "label": "1x1 Defensivo (Tecnica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "lateralidade_score", "label": "Lateralidade (Tecnica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "imprevisibilidade_score", "label": "Imprevisibilidade (Tatica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "leitura_jogo_score", "label": "Leitura de Jogo (Tatica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "dominio_espaco_score", "label": "Dominio do Espaco (Tatica)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "reposicao_pe_score", "label": "Reposicao com o Pe (Tecnica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "reposicao_mao_score", "label": "Reposicao com a Mao (Tecnica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "tomada_decisao_score", "label": "Tomada de Decisao (Tatica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "comunicacao_score", "label": "Comunicacao (Tatica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "posicionamento_prontidao_score", "label": "Posicionamento (Tatica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "defesa_membros_superiores_score", "label": "Defesa Membros Superiores (Tecnica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "defesa_membros_inferiores_score", "label": "Defesa Membros Inferiores (Tecnica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "defesa_6m_ocupa_espaco_score", "label": "Defesa 6m (Tatica GR)", "group": "Tecnico | Tatica", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "espirito_equipa_score", "label": "Espirito de Equipa (Psicologico)", "group": "Psicologico", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "controlo_emocional_score", "label": "Controlo Emocional (Psicologico)", "group": "Psicologico", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "tenacidade_resiliencia_score", "label": "Tenacidade | Resiliencia (Psicologico)", "group": "Psicologico", "unit": "score", "direction": "higher"},
    {"source": "technical", "key": "atencao_concentracao_score", "label": "Atencao | Concentracao (Psicologico)", "group": "Psicologico", "unit": "score", "direction": "higher"},
]

METRIC_CATALOG_DF = pd.DataFrame(METRIC_CATALOG)
META_COLUMNS = [
    "record_id",
    "inserted_at",
    "atleta_id",
    "nome",
    "genero",
    "selecao",
    "posicao",
    "data_nascimento",
    "data_avaliacao",
    "ano_avaliacao",
    "idade_avaliacao",
    "escalao_avaliacao",
    "estado_maturacional",
    "salto_maturacional",
    "record_source",
]

ANTHROPOMETRY_PHYSICAL_COLUMNS = [
    "peso_kg",
    "altura_cm",
    "envergadura_cm",
    "comprimento_perna_cm",
    "altura_sentada_cm",
    "salto_maturacional",
    "estado_maturacional",
]

PHYSICAL_TEST_COLUMNS = [
    "sprint_10m_s",
    "sprint_20m_s",
    "teste_505_esq_s",
    "teste_505_dir_s",
    "sj_altura_cm",
    "cmj_altura_cm",
    "dj_caixa_m",
    "dj_altura_cm",
    "dj_rsi",
    "dj_rsi_mod_mps",
    "dj_contacto_ms",
    "jump_1_cm",
    "jump_2_cm",
    "jump_3_cm",
    "jump_4_cm",
    "jump_5_cm",
    "jump_6_cm",
    "jump_7_cm",
    "jump_8_cm",
    "jump_9_cm",
    "jump_10_cm",
    "contact_1_ms",
    "contact_2_ms",
    "contact_3_ms",
    "contact_4_ms",
    "contact_5_ms",
    "contact_6_ms",
    "contact_7_ms",
    "contact_8_ms",
    "contact_9_ms",
    "contact_10_ms",
    "j10_rsi_10_5",
    "j10_cmj_cm",
    "j10_media_saltos_cm",
    "j10_maximo_cm",
    "j10_minimo_cm",
    "indice_fadiga_10j_pct",
]

ANTHROPOMETRY_SOURCE_TYPES = {"manual_anthropometry", "selection_anthropometry_excel"}
PHYSICAL_TEST_SOURCE_TYPES = {"manual_physical_tests", "selection_physical_tests_excel", "selection_excel", "team_report", "upload"}

PROFILE_SCORE_METRICS = {
    "ANT": {
        "metrics": [
            {"row_key": "fis_peso_kg", "metric_key": "peso_kg"},
            {"row_key": "fis_altura_cm", "metric_key": "altura_cm"},
            {"row_key": "fis_envergadura_cm", "metric_key": "envergadura_cm"},
            {"row_key": "fis_comprimento_perna_cm", "metric_key": "comprimento_perna_cm"},
            {"row_key": "fis_altura_sentada_cm", "metric_key": "altura_sentada_cm"},
            {"row_key": "fis_salto_maturacional", "metric_key": "salto_maturacional"},
        ],
        "scopes": [["genero", "escalao_avaliacao"], ["genero"], []],
    },
    "FIS": {
        "metrics": [
            {"row_key": "fis_sprint_10m_s", "metric_key": "sprint_10m_s"},
            {"row_key": "fis_sprint_20m_s", "metric_key": "sprint_20m_s"},
            {"row_key": "fis_teste_505_esq_s", "metric_key": "teste_505_esq_s"},
            {"row_key": "fis_teste_505_dir_s", "metric_key": "teste_505_dir_s"},
            {"row_key": "fis_sj_altura_cm", "metric_key": "sj_altura_cm"},
            {"row_key": "fis_cmj_altura_cm", "metric_key": "cmj_altura_cm"},
            {"row_key": "fis_dj_altura_cm", "metric_key": "dj_altura_cm"},
            {"row_key": "fis_dj_rsi", "metric_key": "dj_rsi"},
            {"row_key": "fis_j10_rsi_10_5", "metric_key": "j10_rsi_10_5"},
            {"row_key": "fis_j10_media_saltos_cm", "metric_key": "j10_media_saltos_cm"},
            {"row_key": "fis_indice_fadiga_10j_pct", "metric_key": "indice_fadiga_10j_pct"},
        ],
        "scopes": [["genero", "escalao_avaliacao"], ["genero"], []],
    },
    "PSI": {
        "metrics": [
            {"row_key": "tec_espirito_equipa_score", "metric_key": "espirito_equipa_score"},
            {"row_key": "tec_controlo_emocional_score", "metric_key": "controlo_emocional_score"},
            {"row_key": "tec_tenacidade_resiliencia_score", "metric_key": "tenacidade_resiliencia_score"},
            {"row_key": "tec_atencao_concentracao_score", "metric_key": "atencao_concentracao_score"},
        ],
        "scopes": [["genero", "selecao", "escalao_avaliacao"], ["genero", "escalao_avaliacao"], ["genero"], []],
    },
}

FIELD_PLAYER_PROFILE_SCORE_METRICS = {
    "TEC": {
        "metrics": [
            {"row_key": "tec_um_x_um_ofensivo_score", "metric_key": "um_x_um_ofensivo_score"},
            {"row_key": "tec_um_x_um_defensivo_score", "metric_key": "um_x_um_defensivo_score"},
            {"row_key": "tec_lateralidade_score", "metric_key": "lateralidade_score"},
        ],
        "scopes": [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
    },
    "TAT": {
        "metrics": [
            {"row_key": "tec_imprevisibilidade_score", "metric_key": "imprevisibilidade_score"},
            {"row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score"},
            {"row_key": "tec_dominio_espaco_score", "metric_key": "dominio_espaco_score"},
        ],
        "scopes": [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
    },
}

GOALKEEPER_PROFILE_SCORE_METRICS = {
    "TEC": {
        "metrics": [
            {"row_key": "tec_reposicao_pe_score", "metric_key": "reposicao_pe_score"},
            {"row_key": "tec_reposicao_mao_score", "metric_key": "reposicao_mao_score"},
            {"row_key": "tec_defesa_membros_superiores_score", "metric_key": "defesa_membros_superiores_score"},
            {"row_key": "tec_defesa_membros_inferiores_score", "metric_key": "defesa_membros_inferiores_score"},
        ],
        "scopes": [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
    },
    "TAT": {
        "metrics": [
            {"row_key": "tec_tomada_decisao_score", "metric_key": "tomada_decisao_score"},
            {"row_key": "tec_comunicacao_score", "metric_key": "comunicacao_score"},
            {"row_key": "tec_posicionamento_prontidao_score", "metric_key": "posicionamento_prontidao_score"},
            {"row_key": "tec_defesa_6m_ocupa_espaco_score", "metric_key": "defesa_6m_ocupa_espaco_score"},
            {"row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score"},
        ],
        "scopes": [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
    },
}


def _clean_date(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _clean_number(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_text_value(value) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    return str(value).strip()


def _calculate_age(birth_date, reference_date: date | None = None) -> int | None:
    birth = _clean_date(birth_date)
    ref = _clean_date(reference_date) or date.today()
    if not birth:
        return None
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def _derive_escalao_from_age(age: int | None) -> str:
    if age is None or age < 0:
        return ""
    if age <= 11:
        return "Benjamins"
    if age <= 13:
        return "Infantis"
    if age <= 15:
        return "Iniciados"
    if age <= 17:
        return "Juvenis"
    if age <= 19:
        return "Juniores"
    if age <= 23:
        return "Sub-23"
    return "Seniores"


def _merge_athlete_meta(records_df: pd.DataFrame) -> pd.DataFrame:
    athletes_df = read_athletes()
    if records_df.empty:
        return pd.DataFrame(columns=META_COLUMNS)
    athlete_meta = athletes_df[
        ["atleta_id", "nome", "genero", "selecao", "posicao", "data_nascimento"]
    ].copy()
    merged = records_df.merge(athlete_meta, on="atleta_id", how="left")
    merged["data_avaliacao"] = pd.to_datetime(merged["data_avaliacao"], errors="coerce")
    merged["inserted_at"] = pd.to_datetime(merged["inserted_at"], errors="coerce")
    merged["ano_avaliacao"] = merged["data_avaliacao"].dt.year
    merged["idade_avaliacao"] = merged.apply(
        lambda row: _calculate_age(row.get("data_nascimento"), row.get("data_avaliacao")),
        axis=1,
    )
    merged["escalao_avaliacao"] = merged["idade_avaliacao"].map(_derive_escalao_from_age)
    for col in ["estado_maturacional", "salto_maturacional"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    return merged


def _build_long_metrics(records_df: pd.DataFrame, source: str) -> pd.DataFrame:
    if records_df.empty:
        return pd.DataFrame(columns=META_COLUMNS + ["metric_key", "metric_value", "metric_label", "metric_group", "metric_unit", "direction"])

    merged = _merge_athlete_meta(records_df)
    merged["record_source"] = source
    metrics_df = METRIC_CATALOG_DF[METRIC_CATALOG_DF["source"] == source].copy()
    value_columns = metrics_df["key"].tolist()
    long_df = merged.melt(
        id_vars=META_COLUMNS,
        value_vars=value_columns,
        var_name="metric_key",
        value_name="metric_value",
    )
    long_df["metric_value"] = pd.to_numeric(long_df["metric_value"], errors="coerce")
    long_df = long_df.dropna(subset=["metric_value"]).copy()
    long_df = long_df.merge(metrics_df, left_on="metric_key", right_on="key", how="left")
    long_df = long_df.rename(
        columns={
            "label": "metric_label",
            "group": "metric_group",
            "unit": "metric_unit",
        }
    )
    long_df = long_df.drop(columns=["key", "source"])
    return long_df


def build_metric_history() -> pd.DataFrame:
    physical_long = _build_long_metrics(read_physical_records(), "physical")
    technical_long = _build_long_metrics(read_technical_records(), "technical")
    dataframes = [df for df in [physical_long, technical_long] if not df.empty]
    if not dataframes:
        return pd.DataFrame(columns=META_COLUMNS + ["metric_key", "metric_value", "metric_label", "metric_group", "metric_unit", "direction"])
    combined = pd.concat(dataframes, ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(
        ["data_avaliacao", "inserted_at", "nome", "metric_label"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)


def build_database_summary() -> dict:
    athletes_df = read_athletes()
    physical_df = read_physical_records()
    technical_df = read_technical_records()
    history_df = build_metric_history()
    years = pd.concat(
        [
            pd.to_datetime(physical_df.get("data_avaliacao"), errors="coerce").dt.year,
            pd.to_datetime(technical_df.get("data_avaliacao"), errors="coerce").dt.year,
        ],
        ignore_index=True,
    ).dropna()
    selections = athletes_df["selecao"].fillna("").astype(str).str.strip()
    return {
        "n_atletas": int(len(athletes_df)),
        "n_atletas_ativas": int(athletes_df["ativo"].fillna(False).astype(bool).sum()) if "ativo" in athletes_df.columns else 0,
        "n_avaliacoes_fisicas": int(len(physical_df)),
        "n_avaliacoes_tecnicas": int(len(technical_df)),
        "n_registos_metricos": int(len(history_df)),
        "n_anos": int(years.nunique()),
        "n_selecoes": int(selections[selections.ne("")].nunique()),
        "n_testes": int(METRIC_CATALOG_DF["metric_key"].nunique()) if "metric_key" in METRIC_CATALOG_DF.columns else int(METRIC_CATALOG_DF["key"].nunique()),
    }


def latest_metric_records(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return history_df
    work_df = history_df.sort_values(
        ["metric_key", "atleta_id", "data_avaliacao", "inserted_at"],
        ascending=[True, True, False, False],
        na_position="last",
    )
    return work_df.drop_duplicates(subset=["metric_key", "atleta_id"], keep="first").reset_index(drop=True)


def compute_reference_stats(metric_df: pd.DataFrame) -> dict:
    if metric_df.empty:
        return {
            "n_avaliacoes": 0,
            "n_atletas": 0,
            "media": None,
            "desvio_padrao": None,
            "p10": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "top_10_cutoff": None,
        }
    values = pd.to_numeric(metric_df["metric_value"], errors="coerce").dropna()
    direction = str(metric_df["direction"].iloc[0])
    return {
        "n_avaliacoes": int(len(metric_df)),
        "n_atletas": int(metric_df["atleta_id"].nunique()),
        "media": float(values.mean()) if not values.empty else None,
        "desvio_padrao": float(values.std(ddof=0)) if not values.empty else None,
        "p10": float(values.quantile(0.10)) if not values.empty else None,
        "p25": float(values.quantile(0.25)) if not values.empty else None,
        "p50": float(values.quantile(0.50)) if not values.empty else None,
        "p75": float(values.quantile(0.75)) if not values.empty else None,
        "p90": float(values.quantile(0.90)) if not values.empty else None,
        "top_10_cutoff": float(values.quantile(0.10 if direction == "lower" else 0.90)) if not values.empty else None,
    }


def score_metric_records(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return metric_df.copy()
    work_df = metric_df.copy()
    work_df["metric_value"] = pd.to_numeric(work_df["metric_value"], errors="coerce")
    direction = str(work_df["direction"].iloc[0])
    if direction == "lower":
        work_df["performance_value"] = -work_df["metric_value"]
    else:
        work_df["performance_value"] = work_df["metric_value"]
    perf = work_df["performance_value"]
    mean_perf = perf.mean()
    std_perf = perf.std(ddof=0)
    work_df["z_score"] = (perf - mean_perf) / std_perf if pd.notna(std_perf) and std_perf not in (0, 0.0) else pd.NA
    work_df["percentil_historico"] = perf.rank(method="average", pct=True) * 100
    return work_df.sort_values(
        ["percentil_historico", "metric_value"],
        ascending=[False, True if direction == "lower" else False],
        na_position="last",
    ).reset_index(drop=True)


def build_yearly_trend(metric_df: pd.DataFrame) -> pd.DataFrame:
    if metric_df.empty:
        return pd.DataFrame(columns=["ano_avaliacao", "media", "mediana", "n_avaliacoes", "n_atletas"])
    trend_df = (
        metric_df.dropna(subset=["ano_avaliacao"])
        .groupby("ano_avaliacao", dropna=True)
        .agg(
            media=("metric_value", "mean"),
            mediana=("metric_value", "median"),
            n_avaliacoes=("metric_value", "size"),
            n_atletas=("atleta_id", "nunique"),
        )
        .reset_index()
        .sort_values("ano_avaliacao")
    )
    return trend_df


def _latest_physical_snapshot(
    physical_df: pd.DataFrame,
    atleta_id: str,
    columns: list[str] | None = None,
    source_types: set[str] | None = None,
) -> dict:
    if physical_df is None or physical_df.empty:
        return {}
    work_df = physical_df[physical_df["atleta_id"].astype(str) == str(atleta_id)].copy()
    if work_df.empty:
        return {}
    if source_types is not None and "source_type" in work_df.columns:
        work_df = work_df[work_df["source_type"].astype(str).isin(source_types)].copy()
    if work_df.empty:
        return {}
    work_df["data_avaliacao"] = pd.to_datetime(work_df["data_avaliacao"], errors="coerce")
    work_df["inserted_at"] = pd.to_datetime(work_df["inserted_at"], errors="coerce")
    work_df = work_df.sort_values(["data_avaliacao", "inserted_at"], ascending=[False, False], na_position="last")
    target_columns = columns or PHYSICAL_COLUMNS
    snapshot: dict[str, object] = {"atleta_id": atleta_id, "data_avaliacao": work_df.iloc[0].get("data_avaliacao")}
    for col in target_columns:
        if col not in work_df.columns:
            continue
        series = work_df[col]
        valid_mask = series.notna()
        if series.dtype == "object":
            valid_mask = valid_mask & series.astype(str).str.strip().ne("")
        valid_rows = work_df[valid_mask]
        snapshot[col] = None if valid_rows.empty else valid_rows.iloc[0].get(col)
    return snapshot


def _build_latest_physical_merge_df(physical_df: pd.DataFrame, athlete_ids: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    for atleta_id in athlete_ids:
        anth_snapshot = _latest_physical_snapshot(
            physical_df,
            atleta_id,
            columns=ANTHROPOMETRY_PHYSICAL_COLUMNS,
            source_types=ANTHROPOMETRY_SOURCE_TYPES,
        )
        tests_snapshot = _latest_physical_snapshot(
            physical_df,
            atleta_id,
            columns=PHYSICAL_TEST_COLUMNS,
            source_types=PHYSICAL_TEST_SOURCE_TYPES,
        )
        row = {
            "atleta_id": atleta_id,
            "fis_anth_data_avaliacao": anth_snapshot.get("data_avaliacao"),
            "fis_tests_data_avaliacao": tests_snapshot.get("data_avaliacao"),
        }
        for col in ANTHROPOMETRY_PHYSICAL_COLUMNS:
            row[f"fis_{col}"] = anth_snapshot.get(col)
        for col in PHYSICAL_TEST_COLUMNS:
            row[f"fis_{col}"] = tests_snapshot.get(col)
        rows.append(row)
    return pd.DataFrame(rows)


def build_latest_athlete_profiles() -> pd.DataFrame:
    athletes_df = read_athletes().copy()
    if athletes_df.empty:
        return athletes_df
    athletes_df["ativo"] = athletes_df["ativo"].fillna(False).astype(bool)
    athletes_df["idade"] = athletes_df["data_nascimento"].map(_calculate_age)
    athletes_df["escalao"] = athletes_df["idade"].map(_derive_escalao_from_age)

    latest_physical = _build_latest_physical_merge_df(read_physical_records(), athletes_df["atleta_id"].astype(str).tolist())
    latest_technical = latest_records_by_athlete(read_technical_records())
    if not latest_physical.empty:
        athletes_df = athletes_df.merge(latest_physical, on="atleta_id", how="left")
    if not latest_technical.empty:
        latest_technical = latest_technical.add_prefix("tec_").rename(columns={"tec_atleta_id": "atleta_id"})
        athletes_df = athletes_df.merge(latest_technical, on="atleta_id", how="left")
    pathway_df = read_athlete_pathway()
    if not pathway_df.empty:
        latest_pathway = pathway_df.copy()
        latest_pathway["ano"] = pd.to_numeric(latest_pathway["ano"], errors="coerce")
        latest_pathway["data_referencia"] = pd.to_datetime(latest_pathway["data_referencia"], errors="coerce", dayfirst=True)
        latest_pathway["inserted_at"] = pd.to_datetime(latest_pathway["inserted_at"], errors="coerce")
        latest_pathway = latest_pathway.sort_values(
            ["atleta_id", "ano", "data_referencia", "inserted_at"],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop_duplicates(subset=["atleta_id"], keep="first")
        latest_pathway = latest_pathway.rename(
            columns={
                "selecao": "selecao_atual",
                "estado": "estado_atual",
                "data_referencia": "data_referencia_atual",
            }
        )
        athletes_df = athletes_df.merge(
            latest_pathway[["atleta_id", "selecao_atual", "estado_atual", "data_referencia_atual"]],
            on="atleta_id",
            how="left",
        )
        athletes_df["selecao"] = athletes_df["selecao_atual"].where(
            athletes_df["selecao_atual"].notna() & (athletes_df["selecao_atual"].astype(str) != ""),
            athletes_df["selecao"],
        )
    else:
        athletes_df["estado_atual"] = ""
        athletes_df["selecao_atual"] = athletes_df["selecao"]
    athletes_df["escalao_atual"] = athletes_df["escalao"]
    return athletes_df


def metric_comparison(metric_key: str, value, athlete_row: pd.Series, history_df: pd.DataFrame, scope_sets: list[list[str]]) -> dict | None:
    number = _clean_number(value)
    if number is None or history_df.empty:
        return None
    metric_df = history_df[history_df["metric_key"] == metric_key].copy()
    if metric_df.empty:
        return None

    field_values = {
        "genero": _clean_text_value(athlete_row.get("genero")),
        "selecao": _clean_text_value(athlete_row.get("selecao")),
        "escalao_avaliacao": _clean_text_value(athlete_row.get("escalao")),
        "posicao": _clean_text_value(athlete_row.get("posicao")),
    }

    chosen_df = metric_df
    for scope_fields in scope_sets:
        scoped_df = metric_df.copy()
        valid_scope = True
        for field in scope_fields:
            field_value = field_values.get(field, "")
            if not field_value:
                valid_scope = False
                break
            scoped_df = scoped_df[scoped_df[field].astype(str) == field_value].copy()
        if valid_scope and len(scoped_df) >= 4 and scoped_df["atleta_id"].nunique() >= 3:
            chosen_df = scoped_df
            break

    values = pd.to_numeric(chosen_df["metric_value"], errors="coerce").dropna()
    if values.empty:
        return None
    direction = _clean_text_value(chosen_df["direction"].iloc[0])
    if direction == "lower":
        perf_values = -values
        perf_value = -number
    else:
        perf_values = values
        perf_value = number
    percentile = float((perf_values <= perf_value).mean() * 100)
    return {
        "percentile": percentile,
        "direction": direction,
        "n_avaliacoes": int(len(chosen_df)),
        "n_atletas": int(chosen_df["atleta_id"].nunique()),
    }


def compute_profile_scores_for_row(athlete_row: pd.Series, history_df: pd.DataFrame) -> dict[str, float | None]:
    is_goalkeeper = _clean_text_value(athlete_row.get("posicao")) == "GR"
    score_map = dict(PROFILE_SCORE_METRICS)
    score_map.update(GOALKEEPER_PROFILE_SCORE_METRICS if is_goalkeeper else FIELD_PLAYER_PROFILE_SCORE_METRICS)

    results: dict[str, float | None] = {}
    for score_key, config in score_map.items():
        percentiles: list[float] = []
        for metric in config["metrics"]:
            comparison = metric_comparison(
                metric["metric_key"],
                athlete_row.get(metric["row_key"]),
                athlete_row,
                history_df,
                config["scopes"],
            )
            if comparison is not None and comparison.get("percentile") is not None:
                percentiles.append(float(comparison["percentile"]))
        results[score_key] = None if not percentiles else sum(percentiles) / len(percentiles)
    return results


def build_profile_scores_dataframe() -> pd.DataFrame:
    profiles_df = build_latest_athlete_profiles()
    if profiles_df.empty:
        return profiles_df
    history_df = build_metric_history()
    score_rows: list[dict] = []
    for _, row in profiles_df.iterrows():
        scores = compute_profile_scores_for_row(row, history_df)
        score_rows.append(
            {
                "atleta_id": row.get("atleta_id"),
                "ANT": scores.get("ANT"),
                "FIS": scores.get("FIS"),
                "TEC": scores.get("TEC"),
                "TAT": scores.get("TAT"),
                "PSI": scores.get("PSI"),
            }
        )
    scores_df = pd.DataFrame(score_rows)
    return profiles_df.merge(scores_df, on="atleta_id", how="left")
