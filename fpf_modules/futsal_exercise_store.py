from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from .constants import CLEANDATA_DIR
from .supabase_manager import delete_table_rows, insert_or_update_table, read_table


FUTSAL_DIR = Path(CLEANDATA_DIR) / "futsal"
EXERCISES_PATH = FUTSAL_DIR / "exercise_library.parquet"
EXERCISE_OBJECTS_PATH = FUTSAL_DIR / "exercise_objects.parquet"

EXERCISE_COLUMNS = [
    "exercise_id",
    "created_at",
    "updated_at",
    "titulo",
    "categoria",
    "objetivo",
    "contexto",
    "estrutura_funcional",
    "espaco",
    "numero_jogadores",
    "forma",
    "capacidade_motora",
    "duracao",
    "intensidade",
    "condicionantes",
    "comportamentos_ofensivos",
    "comportamentos_defensivos",
    "frequencia",
    "recuperacao",
    "densidade",
    "volume",
    "descricao",
    "observacoes",
    "canvas_json",
]

EXERCISE_OBJECT_COLUMNS = [
    "object_id",
    "exercise_id",
    "ordem",
    "tipo",
    "rotulo",
    "cor",
    "x_pct",
    "y_pct",
    "tamanho",
]

SUPABASE_TABLES = {
    "exercises": ("futsal_exercises", EXERCISE_COLUMNS, ["exercise_id"]),
    "exercise_objects": ("futsal_exercise_objects", EXERCISE_OBJECT_COLUMNS, ["object_id"]),
}


def _selected_backend() -> str:
    env_backend = str(os.getenv("FPF_FUTSAL_BACKEND", "")).strip().lower()
    if env_backend:
        return env_backend
    try:
        return str(st.secrets.get("futsal", {}).get("backend") or "parquet").strip().lower()
    except Exception:
        return "parquet"


BACKEND = _selected_backend()


def ensure_exercise_store() -> None:
    FUTSAL_DIR.mkdir(parents=True, exist_ok=True)
    for path, columns in [
        (EXERCISES_PATH, EXERCISE_COLUMNS),
        (EXERCISE_OBJECTS_PATH, EXERCISE_OBJECT_COLUMNS),
    ]:
        if not path.exists():
            pd.DataFrame(columns=columns).to_parquet(path, index=False)


def _read_parquet(path: Path, columns: list[str]) -> pd.DataFrame:
    ensure_exercise_store()
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_parquet(path)
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    return df[columns].copy()


def _write_parquet(df: pd.DataFrame, path: Path, columns: list[str]) -> None:
    ensure_exercise_store()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df[columns].to_parquet(path, index=False)


def _ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work_df = df.copy() if df is not None else pd.DataFrame()
    for col in columns:
        if col not in work_df.columns:
            work_df[col] = pd.NA
    return work_df[columns].copy()


def _read_supabase_table(table_key: str) -> pd.DataFrame:
    table_name, columns, _ = SUPABASE_TABLES[table_key]
    df = read_table(table_name)
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    return _ensure_columns(df, columns)


def _sync_supabase_table(table_key: str, df: pd.DataFrame) -> None:
    table_name, columns, pk_columns = SUPABASE_TABLES[table_key]
    clean_df = _ensure_columns(df, columns)
    existing_df = _read_supabase_table(table_key)
    pk = pk_columns[0]
    existing_ids = {
        str(value).strip()
        for value in existing_df.get(pk, pd.Series(dtype="object")).tolist()
        if str(value).strip()
    }
    new_ids = {
        str(value).strip()
        for value in clean_df.get(pk, pd.Series(dtype="object")).tolist()
        if str(value).strip()
    }
    for removed_id in sorted(existing_ids - new_ids):
        delete_table_rows(table_name, {pk: removed_id})
    if not clean_df.empty:
        stats = insert_or_update_table(table_name, clean_df, pk_columns=pk_columns)
        if not stats.get("success", False):
            raise RuntimeError(stats.get("error") or f"Falha ao sincronizar {table_name}.")


def read_exercises_parquet() -> pd.DataFrame:
    return _read_parquet(EXERCISES_PATH, EXERCISE_COLUMNS)


def read_exercise_objects_parquet() -> pd.DataFrame:
    return _read_parquet(EXERCISE_OBJECTS_PATH, EXERCISE_OBJECT_COLUMNS)


def read_exercises() -> pd.DataFrame:
    if BACKEND == "supabase":
        return _read_supabase_table("exercises")
    return read_exercises_parquet()


def read_exercise_objects() -> pd.DataFrame:
    if BACKEND == "supabase":
        return _read_supabase_table("exercise_objects")
    return read_exercise_objects_parquet()


def save_exercises(df: pd.DataFrame) -> None:
    if BACKEND == "supabase":
        _sync_supabase_table("exercises", df)
        return
    _write_parquet(df.copy(), EXERCISES_PATH, EXERCISE_COLUMNS)


def save_exercise_objects(df: pd.DataFrame) -> None:
    if BACKEND == "supabase":
        _sync_supabase_table("exercise_objects", df)
        return
    _write_parquet(df.copy(), EXERCISE_OBJECTS_PATH, EXERCISE_OBJECT_COLUMNS)


def upsert_exercise(exercise_payload: dict, objects_df: pd.DataFrame) -> str:
    exercises_df = read_exercises()
    all_objects_df = read_exercise_objects()

    exercise_id = str(exercise_payload.get("exercise_id") or "").strip() or uuid.uuid4().hex
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {col: exercise_payload.get(col) for col in EXERCISE_COLUMNS}
    payload["exercise_id"] = exercise_id
    payload["created_at"] = payload.get("created_at") or now_iso
    payload["updated_at"] = now_iso

    mask = exercises_df["exercise_id"].astype(str) == exercise_id
    if mask.any():
        payload["created_at"] = exercises_df.loc[mask, "created_at"].iloc[0]
        exercises_df = exercises_df.loc[~mask].copy()

    exercises_df = pd.concat([exercises_df, pd.DataFrame([payload])], ignore_index=True)
    save_exercises(exercises_df)

    all_objects_df = all_objects_df.loc[all_objects_df["exercise_id"].astype(str) != exercise_id].copy()
    clean_objects = objects_df.copy()
    clean_objects["exercise_id"] = exercise_id
    if "object_id" not in clean_objects.columns:
        clean_objects["object_id"] = [uuid.uuid4().hex for _ in range(len(clean_objects))]
    clean_objects["object_id"] = clean_objects["object_id"].fillna("").astype(str)
    empty_mask = clean_objects["object_id"].eq("")
    if empty_mask.any():
        clean_objects.loc[empty_mask, "object_id"] = [uuid.uuid4().hex for _ in range(int(empty_mask.sum()))]
    for col in EXERCISE_OBJECT_COLUMNS:
        if col not in clean_objects.columns:
            clean_objects[col] = pd.NA
    all_objects_df = pd.concat([all_objects_df, clean_objects[EXERCISE_OBJECT_COLUMNS]], ignore_index=True)
    save_exercise_objects(all_objects_df)
    return exercise_id


def delete_exercise(exercise_id: str) -> None:
    exercise_id = str(exercise_id or "").strip()
    if not exercise_id:
        raise RuntimeError("Exercise ID invalido.")
    if BACKEND == "supabase":
        delete_table_rows("futsal_exercise_objects", {"exercise_id": exercise_id})
        delete_table_rows("futsal_exercises", {"exercise_id": exercise_id})
        return
    exercises_df = read_exercises_parquet()
    objects_df = read_exercise_objects_parquet()
    save_exercises(exercises_df.loc[exercises_df["exercise_id"].astype(str) != exercise_id].copy())
    save_exercise_objects(objects_df.loc[objects_df["exercise_id"].astype(str) != exercise_id].copy())
