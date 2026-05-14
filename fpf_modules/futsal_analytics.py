from __future__ import annotations

from datetime import date

import pandas as pd

from .futsal_parquet_store import read_athletes, read_physical_records, read_technical_records


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


def _clean_date(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


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
