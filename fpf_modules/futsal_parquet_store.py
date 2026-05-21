from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .constants import CLEANDATA_DIR


FUTSAL_DIR = Path(CLEANDATA_DIR) / "futsal"
FUTSAL_PHOTOS_DIR = FUTSAL_DIR / "athlete_photos"
ATHLETES_PATH = FUTSAL_DIR / "athletes_master.parquet"
ATHLETE_HISTORY_PATH = FUTSAL_DIR / "athlete_history.parquet"
ATHLETE_PATHWAY_PATH = FUTSAL_DIR / "athlete_pathway.parquet"
PHYSICAL_RECORDS_PATH = FUTSAL_DIR / "physical_assessments.parquet"
TECHNICAL_RECORDS_PATH = FUTSAL_DIR / "technical_psychological_assessments.parquet"

ATHLETE_ID_PREFIX = "FTS"

ATHLETE_COLUMNS = [
    "atleta_id",
    "nome",
    "data_nascimento",
    "genero",
    "selecao",
    "posicao",
    "foto_path",
    "ativo",
    "created_at",
    "updated_at",
]

ATHLETE_HISTORY_COLUMNS = [
    "event_id",
    "inserted_at",
    "event_type",
    "atleta_id",
    "nome",
    "data_nascimento",
    "genero",
    "selecao",
    "posicao",
    "ativo",
    "descricao",
]

ATHLETE_PATHWAY_COLUMNS = [
    "entry_id",
    "inserted_at",
    "atleta_id",
    "ano",
    "selecao",
    "epoca",
    "escalao",
    "estado",
    "data_referencia",
    "observacoes",
]

RECORD_META_COLUMNS = [
    "record_id",
    "batch_id",
    "inserted_at",
    "source_type",
    "source_file",
    "atleta_id",
    "data_avaliacao",
]

PHYSICAL_COLUMNS = [
    "peso_kg",
    "altura_cm",
    "envergadura_cm",
    "comprimento_perna_cm",
    "altura_sentada_cm",
    "salto_maturacional",
    "estado_maturacional",
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

TECHNICAL_COLUMNS = [
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
    "defesa_membros_superiores_score",
    "defesa_membros_inferiores_score",
    "defesa_6m_ocupa_espaco_score",
    "espirito_equipa_score",
    "controlo_emocional_score",
    "tenacidade_resiliencia_score",
    "atencao_concentracao_score",
    "observacoes",
]

PHYSICAL_LENGTH_COLUMNS = [
    "altura_cm",
    "altura_sentada_cm",
    "envergadura_cm",
    "comprimento_perna_cm",
]


def ensure_store() -> None:
    FUTSAL_DIR.mkdir(parents=True, exist_ok=True)
    FUTSAL_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    for path, columns in [
        (ATHLETES_PATH, ATHLETE_COLUMNS),
        (ATHLETE_HISTORY_PATH, ATHLETE_HISTORY_COLUMNS),
        (ATHLETE_PATHWAY_PATH, ATHLETE_PATHWAY_COLUMNS),
        (PHYSICAL_RECORDS_PATH, RECORD_META_COLUMNS + PHYSICAL_COLUMNS),
        (TECHNICAL_RECORDS_PATH, RECORD_META_COLUMNS + TECHNICAL_COLUMNS),
    ]:
        if not path.exists():
            pd.DataFrame(columns=columns).to_parquet(path, index=False)


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_store()
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_parquet(path)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns].copy()


def _write_parquet(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    ensure_store()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df[columns].to_parquet(path, index=False)


def _normalize_length_columns(df: pd.DataFrame) -> pd.DataFrame:
    work_df = df.copy()
    for col in PHYSICAL_LENGTH_COLUMNS:
        if col not in work_df.columns:
            continue
        numeric = pd.to_numeric(work_df[col], errors="coerce")
        meter_mask = numeric.notna() & (numeric > 0) & (numeric < 3)
        numeric.loc[meter_mask] = numeric.loc[meter_mask] * 100
        work_df[col] = numeric
    return work_df


def _clean_date_value(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _clean_number_value(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return None


def _calculate_decimal_age(birth_date, reference_date) -> float | None:
    birth = _clean_date_value(birth_date)
    ref = _clean_date_value(reference_date)
    if not birth or not ref:
        return None
    return (ref - birth).days / 365.25


def _calculate_maturity_offset(genero, birth_date, reference_date, peso_kg, altura_cm, altura_sentada_cm) -> float | None:
    age = _calculate_decimal_age(birth_date, reference_date)
    weight = _clean_number_value(peso_kg)
    stature = _clean_number_value(altura_cm)
    sitting_height = _clean_number_value(altura_sentada_cm)
    sex = "" if genero is None or pd.isna(genero) else str(genero).strip().lower()
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
    value = _clean_number_value(maturity_offset)
    if value is None:
        return ""
    if value < -1:
        return "Pre-PHV"
    if value <= 1:
        return "Circa-PHV"
    return "Post-PHV"


def _repair_physical_records(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work_df = _normalize_length_columns(df)
    athletes_df = read_athletes()
    if athletes_df.empty:
        return work_df

    athlete_meta = athletes_df[["atleta_id", "data_nascimento", "genero"]].copy()
    merged = work_df.merge(athlete_meta, on="atleta_id", how="left", suffixes=("", "_ath"))
    calc_mask = (
        merged["peso_kg"].notna()
        & merged["altura_cm"].notna()
        & merged["altura_sentada_cm"].notna()
        & merged["data_nascimento"].notna()
    )
    merged.loc[calc_mask, "salto_maturacional"] = merged.loc[calc_mask].apply(
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
    merged.loc[calc_mask, "estado_maturacional"] = merged.loc[calc_mask, "salto_maturacional"].map(_classify_maturity_offset)
    return merged[RECORD_META_COLUMNS + PHYSICAL_COLUMNS].copy()


def read_athletes() -> pd.DataFrame:
    return _read_parquet(ATHLETES_PATH, ATHLETE_COLUMNS)


def save_athletes(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), ATHLETES_PATH, ATHLETE_COLUMNS)


def read_athlete_history() -> pd.DataFrame:
    return _read_parquet(ATHLETE_HISTORY_PATH, ATHLETE_HISTORY_COLUMNS)


def save_athlete_history(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), ATHLETE_HISTORY_PATH, ATHLETE_HISTORY_COLUMNS)


def append_athlete_history_event(record: dict) -> None:
    df = read_athlete_history()
    payload = {col: record.get(col) for col in ATHLETE_HISTORY_COLUMNS}
    payload["event_id"] = payload.get("event_id") or uuid.uuid4().hex
    payload["inserted_at"] = payload.get("inserted_at") or datetime.now(timezone.utc).isoformat()
    df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    save_athlete_history(df)


def read_athlete_pathway() -> pd.DataFrame:
    return _read_parquet(ATHLETE_PATHWAY_PATH, ATHLETE_PATHWAY_COLUMNS)


def save_athlete_pathway(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), ATHLETE_PATHWAY_PATH, ATHLETE_PATHWAY_COLUMNS)


def append_athlete_pathway_entry(record: dict) -> None:
    df = read_athlete_pathway()
    payload = {col: record.get(col) for col in ATHLETE_PATHWAY_COLUMNS}
    payload["entry_id"] = payload.get("entry_id") or uuid.uuid4().hex
    payload["inserted_at"] = payload.get("inserted_at") or datetime.now(timezone.utc).isoformat()
    df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    save_athlete_pathway(df)


def update_athlete_pathway_entry(entry_id: str, updates: dict) -> None:
    entry_id = str(entry_id or "").strip()
    if not entry_id:
        raise RuntimeError("Entry ID invalido.")
    df = read_athlete_pathway()
    mask = df["entry_id"].astype(str) == entry_id
    if not mask.any():
        raise RuntimeError("Registo de percurso nao encontrado.")
    for key, value in updates.items():
        if key in df.columns:
            df.loc[mask, key] = value
    save_athlete_pathway(df.copy())


def delete_athlete_pathway_entry(entry_id: str) -> None:
    entry_id = str(entry_id or "").strip()
    if not entry_id:
        raise RuntimeError("Entry ID invalido.")
    df = read_athlete_pathway()
    save_athlete_pathway(df[df["entry_id"].astype(str) != entry_id].copy())


def read_physical_records() -> pd.DataFrame:
    raw_df = _read_parquet(PHYSICAL_RECORDS_PATH, RECORD_META_COLUMNS + PHYSICAL_COLUMNS)
    repaired_df = _repair_physical_records(raw_df)
    if not raw_df.equals(repaired_df):
        save_physical_records(repaired_df)
    return repaired_df


def save_physical_records(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), PHYSICAL_RECORDS_PATH, RECORD_META_COLUMNS + PHYSICAL_COLUMNS)


def read_technical_records() -> pd.DataFrame:
    return _read_parquet(TECHNICAL_RECORDS_PATH, RECORD_META_COLUMNS + TECHNICAL_COLUMNS)


def save_technical_records(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), TECHNICAL_RECORDS_PATH, RECORD_META_COLUMNS + TECHNICAL_COLUMNS)


def generate_athlete_id(athletes_df: pd.DataFrame) -> str:
    if athletes_df is None or athletes_df.empty or "atleta_id" not in athletes_df.columns:
        return f"{ATHLETE_ID_PREFIX}-00001"
    ids = athletes_df["atleta_id"].fillna("").astype(str).str.extract(rf"{ATHLETE_ID_PREFIX}-(\d+)", expand=False).dropna()
    next_number = 1 if ids.empty else int(ids.astype(int).max()) + 1
    return f"{ATHLETE_ID_PREFIX}-{next_number:05d}"


def upsert_athlete(record: dict) -> None:
    df = read_athletes()
    now = datetime.now(timezone.utc).isoformat()
    payload = {col: record.get(col) for col in ATHLETE_COLUMNS}
    payload["updated_at"] = now
    atleta_id = str(payload.get("atleta_id") or "").strip()
    if not atleta_id:
        raise RuntimeError("O atleta tem de ter um ID.")
    old_row = df.loc[df["atleta_id"].astype(str) == atleta_id].head(1)
    is_new = old_row.empty
    payload["created_at"] = payload.get("created_at") or (
        old_row["created_at"].iloc[0] if not old_row.empty else now
    )
    if not old_row.empty:
        df = df[df["atleta_id"].astype(str) != atleta_id].copy()
    df = pd.concat([df, pd.DataFrame([payload])], ignore_index=True)
    df["ativo"] = df["ativo"].fillna(False).astype(bool)
    save_athletes(
        df.sort_values(
            ["ativo", "nome", "atleta_id"],
            ascending=[False, True, True],
            na_position="last",
        )
    )
    append_athlete_history_event(
        {
            "event_type": "create_athlete" if is_new else "update_athlete",
            "atleta_id": atleta_id,
            "nome": payload.get("nome"),
            "data_nascimento": payload.get("data_nascimento"),
            "genero": payload.get("genero"),
            "selecao": payload.get("selecao"),
            "posicao": payload.get("posicao"),
            "ativo": payload.get("ativo"),
            "descricao": "Criacao da atleta" if is_new else "Atualizacao da ficha da atleta",
        }
    )


def append_records(kind: str, df_new: pd.DataFrame, source_type: str, source_file: str = "") -> dict:
    if df_new is None or df_new.empty:
        return {"batch_id": "", "inserted": 0}

    batch_id = uuid.uuid4().hex[:12]
    inserted_at = datetime.now(timezone.utc).isoformat()
    work_df = df_new.copy()
    work_df["record_id"] = [uuid.uuid4().hex for _ in range(len(work_df))]
    work_df["batch_id"] = batch_id
    work_df["inserted_at"] = inserted_at
    work_df["source_type"] = source_type
    work_df["source_file"] = source_file

    if kind == "physical":
        current_df = read_physical_records()
        combined = pd.concat([current_df, work_df], ignore_index=True)
        save_physical_records(combined)
    elif kind == "technical":
        current_df = read_technical_records()
        combined = pd.concat([current_df, work_df], ignore_index=True)
        save_technical_records(combined)
    else:
        raise RuntimeError("Tipo de registo invalido.")

    return {"batch_id": batch_id, "inserted": len(work_df)}


def delete_record(kind: str, record_id: str) -> None:
    record_id = str(record_id or "").strip()
    if not record_id:
        raise RuntimeError("Record ID invalido.")
    if kind == "physical":
        df = read_physical_records()
        save_physical_records(df[df["record_id"].astype(str) != record_id].copy())
    elif kind == "technical":
        df = read_technical_records()
        save_technical_records(df[df["record_id"].astype(str) != record_id].copy())
    else:
        raise RuntimeError("Tipo de registo invalido.")


def update_record(kind: str, record_id: str, updates: dict) -> None:
    record_id = str(record_id or "").strip()
    if not record_id:
        raise RuntimeError("Record ID invalido.")

    if kind == "physical":
        df = read_physical_records()
        save_fn = save_physical_records
    elif kind == "technical":
        df = read_technical_records()
        save_fn = save_technical_records
    else:
        raise RuntimeError("Tipo de registo invalido.")

    mask = df["record_id"].astype(str) == record_id
    if not mask.any():
        raise RuntimeError("Registo nao encontrado.")

    for key, value in updates.items():
        if key in df.columns:
            df.loc[mask, key] = value
    save_fn(df.copy())


def delete_athlete(atleta_id: str) -> None:
    atleta_id = str(atleta_id or "").strip()
    if not atleta_id:
        raise RuntimeError("Atleta ID invalido.")
    history_df = read_athlete_history()
    save_athlete_history(history_df[history_df["atleta_id"].astype(str) != atleta_id].copy())
    pathway_df = read_athlete_pathway()
    save_athlete_pathway(pathway_df[pathway_df["atleta_id"].astype(str) != atleta_id].copy())
    athletes_df = read_athletes()
    save_athletes(athletes_df[athletes_df["atleta_id"].astype(str) != atleta_id].copy())
    physical_df = read_physical_records()
    save_physical_records(physical_df[physical_df["atleta_id"].astype(str) != atleta_id].copy())
    technical_df = read_technical_records()
    save_technical_records(technical_df[technical_df["atleta_id"].astype(str) != atleta_id].copy())


def latest_records_by_athlete(records_df: pd.DataFrame) -> pd.DataFrame:
    if records_df is None or records_df.empty:
        return pd.DataFrame(columns=records_df.columns if records_df is not None else [])
    work_df = records_df.copy()
    work_df["data_avaliacao"] = pd.to_datetime(work_df["data_avaliacao"], errors="coerce")
    work_df["inserted_at"] = pd.to_datetime(work_df["inserted_at"], errors="coerce")
    work_df = work_df.sort_values(["atleta_id", "data_avaliacao", "inserted_at"], ascending=[True, False, False], na_position="last")
    return work_df.drop_duplicates(subset=["atleta_id"], keep="first").reset_index(drop=True)


def save_photo(atleta_id: str, uploaded_file) -> str:
    ensure_store()
    extension = uploaded_file.name.rsplit(".", 1)[-1].lower() if "." in uploaded_file.name else "jpg"
    path = FUTSAL_PHOTOS_DIR / f"{atleta_id}.{extension}"
    path.write_bytes(uploaded_file.getvalue())
    return str(path)


def delete_photo(photo_path: str) -> None:
    path = Path(str(photo_path or "").strip())
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass
