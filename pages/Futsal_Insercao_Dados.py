from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO
import unicodedata

import pandas as pd
import streamlit as st

from fpf_modules.futsal_parquet_store import (
    PHYSICAL_COLUMNS,
    TECHNICAL_COLUMNS,
    append_records,
    read_athletes,
)


REPORT_PHYSICAL_COLUMN_MAP = {
    "Peso(kg)": "peso_kg",
    "Peso corporal (kg)": "peso_kg",
    "Altura (cm)": "altura_cm",
    "Altura Sentada (cm)": "altura_sentada_cm",
    "Altura sentada (cm)": "altura_sentada_cm",
    "Envergadura (cm)": "envergadura_cm",
    "Comprimento Perna (cm)": "comprimento_perna_cm",
    "Sprint 10m(s)": "sprint_10m_s",
    "Sprint 20m (s)": "sprint_20m_s",
    "Teste 5-0-5 (esq)": "teste_505_esq_s",
    "Teste 5-0-5 (dir)": "teste_505_dir_s",
    "SJ altura (cm)": "sj_altura_cm",
    "CMJ altura (cm)": "cmj_altura_cm",
    "DJ caixa (m)": "dj_caixa_m",
    "DJ altura (cm)": "dj_altura_cm",
    "DJ RSI": "dj_rsi",
    "DJ RSI mod (m/s)": "dj_rsi_mod_mps",
    "DJ contacto (ms)": "dj_contacto_ms",
    "Jump 1 (cm)": "jump_1_cm",
    "Jump 2 (cm)": "jump_2_cm",
    "Jump 3 (cm)": "jump_3_cm",
    "Jump 4 (cm)": "jump_4_cm",
    "Jump 5 (cm)": "jump_5_cm",
    "Jump 6 (cm)": "jump_6_cm",
    "Jump 7 (cm)": "jump_7_cm",
    "Jump 8 (cm)": "jump_8_cm",
    "Jump 9 (cm)": "jump_9_cm",
    "Jump 10 (cm)": "jump_10_cm",
    "Contact 1 (ms)": "contact_1_ms",
    "Contact 2 (ms)": "contact_2_ms",
    "Contact 3 (ms)": "contact_3_ms",
    "Contact 4 (ms)": "contact_4_ms",
    "Contact 5 (ms)": "contact_5_ms",
    "Contact 6 (ms)": "contact_6_ms",
    "Contact 7 (ms)": "contact_7_ms",
    "Contact 8 (ms)": "contact_8_ms",
    "Contact 9 (ms)": "contact_9_ms",
    "Contact 10 (ms)": "contact_10_ms",
    "10J RSI 10-5": "j10_rsi_10_5",
    "10J média saltos (cm)": "j10_media_saltos_cm",
    "10J mÃ©dia saltos (cm)": "j10_media_saltos_cm",
    "10J CMJ (cm)": "j10_cmj_cm",
    "10J máximo (cm)": "j10_maximo_cm",
    "10J mÃ¡ximo (cm)": "j10_maximo_cm",
    "10J mínimo (cm)": "j10_minimo_cm",
    "10J mÃ­nimo (cm)": "j10_minimo_cm",
    "Índice fadiga 10J (%)": "indice_fadiga_10j_pct",
    "Ãndice fadiga 10J (%)": "indice_fadiga_10j_pct",
}

REPORT_TTP_COLUMN_MAP = {
    "ID": "atleta_id",
    "ID Atleta": "atleta_id",
    "NOME JOGADOR": "nome_jogador",
    "1x1 ofensivo": "um_x_um_ofensivo_score",
    "1x1 defensivo": "um_x_um_defensivo_score",
    "Lateralidade": "lateralidade_score",
    "Imprevisibilidade": "imprevisibilidade_score",
    "Leitura de jogo": "leitura_jogo_score",
    "Domínio do Espaço": "dominio_espaco_score",
    "Espírito de equipa": "espirito_equipa_score",
    "Controlo emocional": "controlo_emocional_score",
    "Tenacidade / Resiliência": "tenacidade_resiliencia_score",
    "Atenção /concentração": "atencao_concentracao_score",
    "Reposição com o pé": "reposicao_pe_score",
    "Reposicao com o pe": "reposicao_pe_score",
    "Reposição com a mão": "reposicao_mao_score",
    "Reposicao com a mao": "reposicao_mao_score",
    "Tomada de decisão": "tomada_decisao_score",
    "Tomada de decisao": "tomada_decisao_score",
    "Comunicação": "comunicacao_score",
    "Comunicacao": "comunicacao_score",
    "Posicionamento": "posicionamento_prontidao_score",
    "Defesa Membros Inferiores": "defesa_membros_inferiores_score",
    "Defesa Membros Superiores": "defesa_membros_superiores_score",
    "Defesa 6m": "defesa_6m_ocupa_espaco_score",
    "Tenacidade": "tenacidade_resiliencia_score",
    "Atenção": "atencao_concentracao_score",
}

PHYSICAL_ANTHROPOMETRY_MODEL_COLUMNS = [
    "ID",
    "Nome",
    "PosiÃ§Ã£o",
    "Peso(kg)",
    "Altura (cm)",
    "Altura Sentada (cm)",
    "Envergadura (cm)",
    "Comprimento Perna (cm)",
]

PHYSICAL_TEST_MODEL_COLUMNS = [
    "ID",
    "Nome",
    "Posição",
    "Peso(kg)",
    "Altura (cm)",
    "Altura Sentada (cm)",
    "Envergadura (cm)",
    "Comprimento Perna (cm)",
    "Sprint 10m(s)",
    "Sprint 20m (s)",
    "Teste 5-0-5 (esq)",
    "Teste 5-0-5 (dir)",
    "SJ altura (cm)",
    "CMJ altura (cm)",
    "DJ caixa (m)",
    "DJ altura (cm)",
    "DJ RSI",
    "DJ RSI mod (m/s)",
    "DJ contacto (ms)",
    "Jump 1 (cm)",
    "Jump 2 (cm)",
    "Jump 3 (cm)",
    "Jump 4 (cm)",
    "Jump 5 (cm)",
    "Jump 6 (cm)",
    "Jump 7 (cm)",
    "Jump 8 (cm)",
    "Jump 9 (cm)",
    "Jump 10 (cm)",
    "Contact 1 (ms)",
    "Contact 2 (ms)",
    "Contact 3 (ms)",
    "Contact 4 (ms)",
    "Contact 5 (ms)",
    "Contact 6 (ms)",
    "Contact 7 (ms)",
    "Contact 8 (ms)",
    "Contact 9 (ms)",
    "Contact 10 (ms)",
    "10J RSI 10-5",
    "10J CMJ (cm)",
    "10J média saltos (cm)",
    "10J máximo (cm)",
    "10J mínimo (cm)",
    "Índice fadiga 10J (%)",
]

PHYSICAL_MODEL_COLUMNS = PHYSICAL_TEST_MODEL_COLUMNS

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

TTP_MODEL_COLUMNS = [
    "ID",
    "NOME JOGADOR",
    "1x1 ofensivo",
    "1x1 defensivo",
    "Lateralidade",
    "Imprevisibilidade",
    "Leitura de jogo",
    "Domínio do Espaço",
    "Espírito de equipa",
    "Controlo emocional",
    "Tenacidade / Resiliência",
    "Atenção /concentração",
]

TTP_GR_MODEL_COLUMNS = [
    "ID",
    "NOME JOGADOR",
    "Reposição com o pé",
    "Reposição com a mão",
    "Tomada de decisão",
    "Comunicação",
    "Posicionamento",
    "Defesa Membros Inferiores",
    "Defesa Membros Superiores",
    "Defesa 6m",
    "Leitura de jogo",
    "Espírito de equipa",
    "Controlo emocional",
    "Tenacidade",
    "Atenção",
]

TTP_SCORE_COLUMNS = [
    "um_x_um_ofensivo_score",
    "um_x_um_defensivo_score",
    "lateralidade_score",
    "imprevisibilidade_score",
    "leitura_jogo_score",
    "dominio_espaco_score",
    "reposicao_pe_score",
    "reposicao_mao_score",
    "tomada_decisao_score",
    "comunicacao_score",
    "posicionamento_prontidao_score",
    "defesa_membros_inferiores_score",
    "defesa_membros_superiores_score",
    "defesa_6m_ocupa_espaco_score",
    "espirito_equipa_score",
    "controlo_emocional_score",
    "tenacidade_resiliencia_score",
    "atencao_concentracao_score",
]

PHYSICAL_TEMPLATE_HINT_COLUMNS = set(REPORT_PHYSICAL_COLUMN_MAP.keys()) | {"ID", "Nome", "Posição", "Posicao"}
PHYSICAL_ANTHRO_HINT_COLUMNS = {"ID", "Nome", "Posição", "PosiÃ§Ã£o", "Posicao", "Peso(kg)", "Peso corporal (kg)", "Altura (cm)", "Altura Sentada (cm)", "Altura sentada (cm)", "Envergadura (cm)", "Comprimento Perna (cm)"}
PHYSICAL_TEST_HINT_COLUMNS = {"ID", "Nome", "Posição", "PosiÃ§Ã£o", "Posicao", "Sprint 10m(s)", "Sprint 20m (s)", "Teste 5-0-5 (esq)", "Teste 5-0-5 (dir)", "SJ altura (cm)", "CMJ altura (cm)", "DJ caixa (m)", "DJ altura (cm)", "DJ RSI", "DJ RSI mod (m/s)", "DJ contacto (ms)", "Jump 1 (cm)", "Jump 10 (cm)", "Contact 1 (ms)", "Contact 10 (ms)", "10J RSI 10-5", "10J CMJ (cm)", "10J mÃ©dia saltos (cm)", "10J mÃ¡ximo (cm)", "10J mÃ­nimo (cm)", "Ãndice fadiga 10J (%)"}
TTP_TEMPLATE_HINT_COLUMNS = set(REPORT_TTP_COLUMN_MAP.keys()) | {"ID", "NOME JOGADOR"}


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _normalize_name_key(value) -> str:
    text = _clean_text_value(value)
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    compact = " ".join(ascii_text.lower().split())
    return compact


def _name_tokens(value) -> list[str]:
    normalized = _normalize_name_key(value)
    return [token for token in normalized.split(" ") if token]


def _names_match_reasonable(db_name, upload_name) -> bool:
    db_tokens = _name_tokens(db_name)
    upload_tokens = _name_tokens(upload_name)
    if not db_tokens or not upload_tokens:
        return True
    if db_tokens == upload_tokens:
        return True

    db_set = set(db_tokens)
    upload_set = set(upload_tokens)

    if len(db_tokens) >= 2 and len(upload_tokens) >= 2:
        if db_tokens[0] == upload_tokens[0] and db_tokens[-1] == upload_tokens[-1]:
            return True

    overlap = db_set & upload_set
    if len(overlap) >= 2 and (db_set.issubset(upload_set) or upload_set.issubset(db_set)):
        return True

    return False


def _clean_date(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _clean_number(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return None


def _normalize_length_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    meter_mask = numeric.notna() & (numeric > 0) & (numeric < 3)
    numeric.loc[meter_mask] = numeric.loc[meter_mask] * 100
    return numeric


def _calculate_decimal_age(birth_date, reference_date) -> float | None:
    birth = _clean_date(birth_date)
    ref = _clean_date(reference_date)
    if not birth or not ref:
        return None
    return (ref - birth).days / 365.25


def _calculate_maturity_offset(genero, birth_date, reference_date, peso_kg, altura_cm, altura_sentada_cm) -> float | None:
    age = _calculate_decimal_age(birth_date, reference_date)
    weight = _clean_number(peso_kg)
    stature = _clean_number(altura_cm)
    sitting_height = _clean_number(altura_sentada_cm)
    sex = _clean_text_value(genero).lower()
    if age is None or weight is None or stature is None or sitting_height is None:
        return None
    leg_length = stature - sitting_height
    if leg_length <= 0:
        return None
    weight_height_ratio = (weight / stature) * 100 if stature else None
    if weight_height_ratio is None:
        return None
    if sex == "masculino":
        return (
            -9.236
            + (0.0002708 * (leg_length * sitting_height))
            - (0.001663 * (age * leg_length))
            + (0.007216 * (age * sitting_height))
            + (0.02292 * weight_height_ratio)
        )
    return (
        -9.376
        + (0.0001882 * (leg_length * sitting_height))
        + (0.0022 * (age * leg_length))
        + (0.005841 * (age * sitting_height))
        - (0.002658 * (age * weight))
        + (0.07693 * weight_height_ratio)
    )


def _classify_maturity_offset(maturity_offset: float | None) -> str:
    value = _clean_number(maturity_offset)
    if value is None:
        return ""
    if value < -1:
        return "Pre-PHV"
    if value <= 1:
        return "Circa-PHV"
    return "Post-PHV"


def _read_table_upload(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    file_name = str(getattr(uploaded_file, "name", "") or "").lower()
    if file_name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded_file)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Para ler ficheiros Excel (.xlsx/.xls) nesta página, a app precisa da biblioteca 'openpyxl' instalada."
            ) from exc
    raw_bytes = uploaded_file.getvalue()
    for sep in [",", ";", "\t"]:
        try:
            return pd.read_csv(StringIO(raw_bytes.decode("utf-8-sig")), sep=sep)
        except Exception:
            continue
    raise RuntimeError("Nao foi possivel ler o ficheiro.")


def _validate_athlete_ids(df_upload: pd.DataFrame, athletes_df: pd.DataFrame) -> None:
    athlete_ids = set(athletes_df["atleta_id"].astype(str).str.strip().tolist()) if not athletes_df.empty else set()
    upload_ids = set(df_upload["atleta_id"].fillna("").astype(str).str.strip().tolist())
    missing = sorted([athlete_id for athlete_id in upload_ids if athlete_id and athlete_id not in athlete_ids])
    if missing:
        raise RuntimeError(f"Os seguintes atleta_id nao existem na ficha mestre: {', '.join(missing[:15])}")


def _validate_athlete_name_match(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, upload_name_column: str) -> None:
    if upload_name_column not in df_upload.columns:
        return
    compare_df = df_upload[["atleta_id", upload_name_column]].copy()
    compare_df[upload_name_column] = compare_df[upload_name_column].fillna("").astype(str).str.strip()
    compare_df = compare_df[compare_df[upload_name_column].ne("")].copy()
    if compare_df.empty:
        return

    athlete_names = athletes_df[["atleta_id", "nome"]].copy()
    athlete_names["nome_base"] = athlete_names["nome"].map(_normalize_name_key)
    compare_df["nome_upload"] = compare_df[upload_name_column].map(_normalize_name_key)
    compare_df = compare_df.merge(athlete_names[["atleta_id", "nome", "nome_base"]], on="atleta_id", how="left")

    mismatch_mask = ~compare_df.apply(
        lambda row: _names_match_reasonable(row.get("nome"), row.get(upload_name_column)),
        axis=1,
    )
    mismatches = compare_df[mismatch_mask].copy()
    if mismatches.empty:
        return

    messages: list[str] = []
    for row_idx, row in mismatches.head(12).iterrows():
        line_number = int(row_idx) + 2
        messages.append(
            f"linha {line_number}: ID {row.get('atleta_id')} | ficheiro '{_clean_text_value(row.get(upload_name_column))}' | BD '{_clean_text_value(row.get('nome'))}'"
        )
    raise RuntimeError(
        "Foram encontradas divergencias entre o ID e o nome da atleta no ficheiro. "
        "Corrige antes de importar: " + " ; ".join(messages)
    )


def _validate_ttp_scale(df_upload: pd.DataFrame, score_columns: list[str]) -> None:
    invalid_messages: list[str] = []
    for col in score_columns:
        if col not in df_upload.columns:
            continue
        numeric = pd.to_numeric(df_upload[col], errors="coerce")
        invalid_mask = numeric.notna() & ~numeric.between(1, 7)
        if invalid_mask.any():
            invalid_rows = (df_upload.index[invalid_mask] + 2).tolist()[:10]
            invalid_messages.append(f"{col}: linhas {', '.join(str(row) for row in invalid_rows)}")
    if invalid_messages:
        raise RuntimeError(
            "Foram encontrados valores fora da escala 1-7 no lote TTP. Corrige estas colunas/linhas: "
            + " | ".join(invalid_messages)
        )


def _validate_template_family(upload_columns: list[str], expected_kind: str) -> None:
    cleaned_columns = {str(col).strip() for col in upload_columns}
    physical_hits = len(cleaned_columns & PHYSICAL_TEMPLATE_HINT_COLUMNS)
    anthropometry_hits = len(cleaned_columns & PHYSICAL_ANTHRO_HINT_COLUMNS)
    physical_test_hits = len(cleaned_columns & PHYSICAL_TEST_HINT_COLUMNS)
    ttp_hits = len(cleaned_columns & TTP_TEMPLATE_HINT_COLUMNS)

    if expected_kind == "physical":
        if physical_hits < 4:
            raise RuntimeError(
                "O ficheiro carregado nao parece ser um modelo de dados fisicos valido. "
                "Confirma se selecionaste a tab correta e se o ficheiro corresponde ao modelo fisico."
            )
        if ttp_hits >= 5 and ttp_hits > physical_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Tecnico | Tatica | Psicologico, nao a Dados Fisicos. "
                "Carrega este ficheiro na tab correta."
            )
    elif expected_kind == "ttp":
        if ttp_hits < 4:
            raise RuntimeError(
                "O ficheiro carregado nao parece ser um modelo TTP valido. "
                "Confirma se selecionaste a tab correta e se o ficheiro corresponde ao modelo tecnico | tatica | psicologico."
            )
        if physical_hits >= 5 and physical_hits > ttp_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Dados Fisicos, nao a Tecnico | Tatica | Psicologico. "
                "Carrega este ficheiro na tab correta."
            )
    elif expected_kind == "anthropometry":
        if anthropometry_hits < 4:
            raise RuntimeError(
                "O ficheiro carregado nao parece ser um modelo antropometrico valido. "
                "Confirma se selecionaste a subtab correta e se o ficheiro corresponde ao modelo antropometrico."
            )
        if physical_test_hits >= 5 and physical_test_hits > anthropometry_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Testes Fisicos, nao a Dados Antropometricos. "
                "Carrega este ficheiro na subtab correta."
            )
        if ttp_hits >= 5 and ttp_hits > anthropometry_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Tecnico | Tatica | Psicologico, nao a Dados Antropometricos. "
                "Carrega este ficheiro na tab correta."
            )
    elif expected_kind == "physical_tests":
        if physical_test_hits < 4:
            raise RuntimeError(
                "O ficheiro carregado nao parece ser um modelo de testes fisicos valido. "
                "Confirma se selecionaste a subtab correta e se o ficheiro corresponde ao modelo de testes."
            )
        if anthropometry_hits >= 5 and anthropometry_hits > physical_test_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Dados Antropometricos, nao a Testes Fisicos. "
                "Carrega este ficheiro na subtab correta."
            )
        if ttp_hits >= 5 and ttp_hits > physical_test_hits:
            raise RuntimeError(
                "O ficheiro carregado parece pertencer a Tecnico | Tatica | Psicologico, nao a Testes Fisicos. "
                "Carrega este ficheiro na tab correta."
            )


def _prepare_bulk_anthropometry_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")
    work_df = df_upload.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]
    _validate_template_family(work_df.columns.tolist(), "anthropometry")
    if "ID" not in work_df.columns:
        raise RuntimeError("O modelo antropometrico tem de incluir a coluna 'ID'.")
    work_df["atleta_id"] = work_df["ID"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nenhuma linha valida foi encontrada no modelo.")
    rename_map = {source: target for source, target in REPORT_PHYSICAL_COLUMN_MAP.items() if source in work_df.columns}
    work_df = work_df.rename(columns=rename_map)
    work_df["data_avaliacao"] = pd.to_datetime(data_avaliacao).date()
    target_columns = [
        "peso_kg",
        "altura_cm",
        "altura_sentada_cm",
        "envergadura_cm",
        "comprimento_perna_cm",
        "salto_maturacional",
        "estado_maturacional",
    ]
    for col in PHYSICAL_COLUMNS:
        if col not in work_df.columns:
            work_df[col] = pd.NA
    for col in target_columns:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    for col in ["altura_cm", "altura_sentada_cm", "envergadura_cm", "comprimento_perna_cm"]:
        if col in work_df.columns:
            work_df[col] = _normalize_length_series(work_df[col])
    _validate_athlete_ids(work_df[["atleta_id"]].copy(), athletes_df)
    _validate_athlete_name_match(work_df, athletes_df, "Nome")
    athlete_meta = athletes_df[["atleta_id", "data_nascimento", "genero"]].copy()
    work_df = work_df.merge(athlete_meta, on="atleta_id", how="left")
    work_df["salto_maturacional"] = work_df.apply(
        lambda row: _calculate_maturity_offset(
            row.get("genero"),
            row.get("data_nascimento"),
            row.get("data_avaliacao"),
            row.get("peso_kg"),
            row.get("altura_cm"),
            row.get("altura_sentada_cm"),
        ),
        axis=1,
    )
    work_df["estado_maturacional"] = work_df["salto_maturacional"].map(_classify_maturity_offset)
    return work_df[["atleta_id", "data_avaliacao"] + PHYSICAL_COLUMNS].copy()


def _prepare_bulk_physical_test_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")
    work_df = df_upload.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]
    _validate_template_family(work_df.columns.tolist(), "physical_tests")
    if "ID" not in work_df.columns:
        raise RuntimeError("O modelo de testes fisicos tem de incluir a coluna 'ID'.")
    work_df["atleta_id"] = work_df["ID"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nenhuma linha valida foi encontrada no modelo.")
    rename_map = {source: target for source, target in REPORT_PHYSICAL_COLUMN_MAP.items() if source in work_df.columns}
    work_df = work_df.rename(columns=rename_map)
    work_df["data_avaliacao"] = pd.to_datetime(data_avaliacao).date()
    for col in PHYSICAL_COLUMNS:
        if col not in work_df.columns:
            work_df[col] = pd.NA
    for col in PHYSICAL_TEST_COLUMNS:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    _validate_athlete_ids(work_df[["atleta_id"]].copy(), athletes_df)
    _validate_athlete_name_match(work_df, athletes_df, "Nome")
    return work_df[["atleta_id", "data_avaliacao"] + PHYSICAL_COLUMNS].copy()


def _prepare_bulk_physical_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    return _prepare_bulk_physical_test_records(df_upload, athletes_df, data_avaliacao)


def _prepare_bulk_ttp_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")
    work_df = df_upload.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]
    _validate_template_family(work_df.columns.tolist(), "ttp")
    if "ID" not in work_df.columns and "ID Atleta" not in work_df.columns:
        raise RuntimeError("O modelo tecnico|tatica|psicologico tem de incluir a coluna 'ID'.")
    rename_map = {source: target for source, target in REPORT_TTP_COLUMN_MAP.items() if source in work_df.columns}
    work_df = work_df.rename(columns=rename_map)
    work_df["atleta_id"] = work_df["atleta_id"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nenhuma linha valida foi encontrada no modelo.")
    work_df["data_avaliacao"] = pd.to_datetime(data_avaliacao).date()
    for col in TECHNICAL_COLUMNS:
        if col not in work_df.columns:
            work_df[col] = pd.NA
    numeric_columns = [col for col in TTP_SCORE_COLUMNS if col in TECHNICAL_COLUMNS]
    for col in numeric_columns:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    if "observacoes" in work_df.columns:
        work_df["observacoes"] = work_df["observacoes"].fillna("").astype(str)
    _validate_ttp_scale(work_df, numeric_columns)
    _validate_athlete_ids(work_df[["atleta_id"]].copy(), athletes_df)
    _validate_athlete_name_match(work_df, athletes_df, "nome_jogador")
    return work_df[["atleta_id", "data_avaliacao"] + TECHNICAL_COLUMNS].copy()


def _build_model_excel(columns: list[str], sheet_name: str) -> bytes:
    output = BytesIO()
    try:
        with pd.ExcelWriter(output) as writer:
            pd.DataFrame(columns=columns).to_excel(writer, sheet_name=sheet_name, index=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Para gerar o modelo Excel (.xlsx), a app precisa da biblioteca 'openpyxl' instalada."
        ) from exc
    return output.getvalue()


def _build_model_csv(columns: list[str]) -> bytes:
    return pd.DataFrame(columns=columns).to_csv(index=False).encode("utf-8-sig")


success_message = st.session_state.pop("futsal_bulk_insert_success", "")
if success_message:
    st.success(success_message)

athletes_df = read_athletes()
if athletes_df.empty:
    st.info("Ainda nao existem atletas registadas.")
    st.stop()

tab_anthropometry, tab_physical, tab_ttp = st.tabs(
    ["Avaliacoes Antropometricas", "Testes Fisicos", "Avaliacoes Tecnico | Tatica | Psicologico"]
)

with tab_anthropometry:
    try:
        excel_model_anth = _build_model_excel(PHYSICAL_ANTHROPOMETRY_MODEL_COLUMNS, "Antropometria")
    except RuntimeError as exc:
        st.warning(str(exc))
        st.download_button(
            "Descarregar modelo antropometrico em CSV",
            data=_build_model_csv(PHYSICAL_ANTHROPOMETRY_MODEL_COLUMNS),
            file_name="modelo_insercao_dados_antropometricos_futsal.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            "Descarregar modelo antropometrico em Excel",
            data=excel_model_anth,
            file_name="modelo_insercao_dados_antropometricos_futsal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    anth_date_col, _ = st.columns([1.2, 3.8])
    data_avaliacao_anth = anth_date_col.date_input("Data da avaliacao do lote antropometrico", value=date.today(), format="DD/MM/YYYY", key="bulk_date_anthropometry")
    anth_upload_col, _ = st.columns([2.2, 2.8])
    uploaded_anth_file = anth_upload_col.file_uploader(
        "Carregar modelo Excel antropometrico da seleÃ§Ã£o",
        type=["xlsx", "xls", "csv"],
        key="futsal_bulk_selection_upload_anthropometry",
    )

    if uploaded_anth_file is not None:
        try:
            raw_anth_df = _read_table_upload(uploaded_anth_file)
            prepared_anth_df = _prepare_bulk_anthropometry_records(raw_anth_df, athletes_df, data_avaliacao_anth)
        except Exception as exc:
            st.error(str(exc))
        else:
            preview_anth_df = prepared_anth_df.merge(
                athletes_df[["atleta_id", "nome", "posicao"]],
                on="atleta_id",
                how="left",
            )
            st.markdown("**PrÃ©-visualizaÃ§Ã£o antropomÃ©trica**")
            anth_preview_columns = ["atleta_id", "nome", "posicao", "data_avaliacao", "peso_kg", "altura_cm", "altura_sentada_cm", "envergadura_cm", "comprimento_perna_cm", "salto_maturacional", "estado_maturacional"]
            st.dataframe(preview_anth_df[anth_preview_columns], use_container_width=True, hide_index=True)
            st.caption(f"Linhas prontas a importar: {len(prepared_anth_df)}")
            if st.button("Importar dados antropometricos da seleÃ§Ã£o", type="primary", key="import_bulk_anthropometry_button"):
                result = append_records("physical", prepared_anth_df, source_type="selection_anthropometry_excel", source_file=uploaded_anth_file.name)
                st.session_state["futsal_bulk_insert_success"] = (
                    f"OperaÃ§Ã£o concluÃ­da. Lote antropometrico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}"
                )
                st.rerun()

with tab_physical:
    try:
        excel_model = _build_model_excel(PHYSICAL_TEST_MODEL_COLUMNS, "Testes_Fisicos")
    except RuntimeError as exc:
        st.warning(str(exc))
        st.download_button(
            "Descarregar modelo fisico em CSV",
            data=_build_model_csv(PHYSICAL_TEST_MODEL_COLUMNS),
            file_name="modelo_insercao_testes_fisicos_futsal.csv",
            mime="text/csv",
        )
    else:
        st.download_button(
            "Descarregar modelo fisico em Excel",
            data=excel_model,
            file_name="modelo_insercao_testes_fisicos_futsal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    physical_date_col, _ = st.columns([1.2, 3.8])
    data_avaliacao_fisica = physical_date_col.date_input("Data da avaliacao do lote de testes", value=date.today(), format="DD/MM/YYYY", key="bulk_date_physical")
    physical_upload_col, _ = st.columns([2.2, 2.8])
    uploaded_physical_file = physical_upload_col.file_uploader(
        "Carregar modelo Excel fisico da seleção",
        type=["xlsx", "xls", "csv"],
        key="futsal_bulk_selection_upload_physical",
    )

    if uploaded_physical_file is not None:
        try:
            raw_df = _read_table_upload(uploaded_physical_file)
            prepared_df = _prepare_bulk_physical_test_records(raw_df, athletes_df, data_avaliacao_fisica)
        except Exception as exc:
            st.error(str(exc))
        else:
            preview_df = prepared_df.merge(
                athletes_df[["atleta_id", "nome", "posicao"]],
                on="atleta_id",
                how="left",
            )
            st.markdown("**Pré-visualização**")
            preview_columns = ["atleta_id", "nome", "posicao", "data_avaliacao"] + PHYSICAL_TEST_COLUMNS
            st.dataframe(preview_df[preview_columns], use_container_width=True, hide_index=True)
            st.caption(f"Linhas prontas a importar: {len(prepared_df)}")
            if st.button("Importar dados fisicos da seleção", type="primary", key="import_bulk_selection_button"):
                result = append_records("physical", prepared_df, source_type="selection_physical_tests_excel", source_file=uploaded_physical_file.name)
                st.session_state["futsal_bulk_insert_success"] = (
                    f"Operação concluída. Lote físico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}"
                )
                st.rerun()

with tab_ttp:
    try:
        excel_model_ttp = _build_model_excel(TTP_MODEL_COLUMNS, "Avaliacao_TTP")
        excel_model_ttp_gr = _build_model_excel(TTP_GR_MODEL_COLUMNS, "Avaliacao_TTP_GR")
    except RuntimeError as exc:
        st.warning(str(exc))
        dl1, dl2, _ = st.columns([1.25, 1, 2.35])
        dl1.download_button(
            "Descarregar modelo TTP campo em CSV",
            data=_build_model_csv(TTP_MODEL_COLUMNS),
            file_name="modelo_insercao_dados_ttp_futsal.csv",
            mime="text/csv",
        )
        dl2.download_button(
            "Descarregar modelo TTP GR em CSV",
            data=_build_model_csv(TTP_GR_MODEL_COLUMNS),
            file_name="modelo_insercao_dados_ttp_gr_futsal.csv",
            mime="text/csv",
        )
    else:
        dl1, dl2, _ = st.columns([1.25, 1, 2.35])
        dl1.download_button(
            "Descarregar modelo TTP campo em Excel",
            data=excel_model_ttp,
            file_name="modelo_insercao_dados_ttp_futsal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        dl2.download_button(
            "Descarregar modelo TTP GR em Excel",
            data=excel_model_ttp_gr,
            file_name="modelo_insercao_dados_ttp_gr_futsal.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    ttp_date_col, _ = st.columns([1.2, 3.8])
    data_avaliacao_ttp = ttp_date_col.date_input("Data da avaliacao do lote TTP", value=date.today(), format="DD/MM/YYYY", key="bulk_date_ttp")
    ttp_upload_col, _ = st.columns([2.2, 2.8])
    uploaded_ttp_file = ttp_upload_col.file_uploader(
        "Carregar modelo Excel tecnico | tatica | psicologico",
        type=["xlsx", "xls", "csv"],
        key="futsal_bulk_selection_upload_ttp",
    )

    if uploaded_ttp_file is not None:
        try:
            raw_ttp_df = _read_table_upload(uploaded_ttp_file)
            prepared_ttp_df = _prepare_bulk_ttp_records(raw_ttp_df, athletes_df, data_avaliacao_ttp)
        except Exception as exc:
            st.error(str(exc))
        else:
            preview_ttp_df = prepared_ttp_df.merge(
                athletes_df[["atleta_id", "nome", "posicao"]],
                on="atleta_id",
                how="left",
            )
            st.markdown("**Pré-visualização**")
            base_preview_columns = [
                "atleta_id",
                "nome",
                "posicao",
                "data_avaliacao",
            ]
            ttp_preview_columns = [
                "um_x_um_ofensivo_score",
                "um_x_um_defensivo_score",
                "lateralidade_score",
                "imprevisibilidade_score",
                "leitura_jogo_score",
                "dominio_espaco_score",
                "reposicao_pe_score",
                "reposicao_mao_score",
                "tomada_decisao_score",
                "comunicacao_score",
                "posicionamento_prontidao_score",
                "defesa_membros_inferiores_score",
                "defesa_membros_superiores_score",
                "defesa_6m_ocupa_espaco_score",
                "espirito_equipa_score",
                "controlo_emocional_score",
                "tenacidade_resiliencia_score",
                "atencao_concentracao_score",
            ]
            visible_ttp_columns = [
                col for col in ttp_preview_columns
                if col in preview_ttp_df.columns and preview_ttp_df[col].notna().any()
            ]
            preview_columns = base_preview_columns + visible_ttp_columns
            st.dataframe(preview_ttp_df[preview_columns], use_container_width=True, hide_index=True)
            st.caption(f"Linhas prontas a importar: {len(prepared_ttp_df)}")
            if st.button("Importar dados TTP da seleção", type="primary", key="import_bulk_ttp_button"):
                result = append_records("technical", prepared_ttp_df, source_type="selection_ttp_excel", source_file=uploaded_ttp_file.name)
                st.session_state["futsal_bulk_insert_success"] = (
                    f"Operação concluída. Lote tecnico | tatica | psicologico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}"
                )
                st.rerun()
