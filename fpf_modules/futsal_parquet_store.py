from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .constants import CLEANDATA_DIR


FUTSAL_DIR = Path(CLEANDATA_DIR) / "futsal"
FUTSAL_PHOTOS_DIR = FUTSAL_DIR / "athlete_photos"
ATHLETES_PATH = FUTSAL_DIR / "athletes_master.parquet"
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
    "passe_score",
    "remate_score",
    "drible_score",
    "controlo_bola_score",
    "decisao_score",
    "concentracao_score",
    "lideranca_score",
    "resiliencia_score",
    "competitividade_score",
    "observacoes",
]


def ensure_store() -> None:
    FUTSAL_DIR.mkdir(parents=True, exist_ok=True)
    FUTSAL_PHOTOS_DIR.mkdir(parents=True, exist_ok=True)
    for path, columns in [
        (ATHLETES_PATH, ATHLETE_COLUMNS),
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


def read_athletes() -> pd.DataFrame:
    return _read_parquet(ATHLETES_PATH, ATHLETE_COLUMNS)


def save_athletes(df: pd.DataFrame) -> None:
    _write_parquet(df.copy(), ATHLETES_PATH, ATHLETE_COLUMNS)


def read_physical_records() -> pd.DataFrame:
    return _read_parquet(PHYSICAL_RECORDS_PATH, RECORD_META_COLUMNS + PHYSICAL_COLUMNS)


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


def delete_athlete(atleta_id: str) -> None:
    atleta_id = str(atleta_id or "").strip()
    if not atleta_id:
        raise RuntimeError("Atleta ID invalido.")
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
