"""
Module responsible for saving and loading data to parquet files.
"""

import os
import pandas as pd

from .constants import CAMPOS_DIR, CLEANDATA_DIR


def save_field_to_parquet(df_new, filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """
    Appends new field data to the parquet database.
    Deduplicates by 'session_id' to avoid repeated entries.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Exists read old data and add new row
    if os.path.exists(filename):
        df_old = pd.read_parquet(filename)
        df_final = pd.concat([df_old, df_new]).drop_duplicates(
            subset=['BL_lat', 'BR_lat', 'TL_lat', 'TR_lat'], keep='last'  # keep='last' so new data overwrites old
        )

    else:
        df_final = df_new

    df_final.to_parquet(filename, index=False)



def read_field_from_parquet(filename=f"{CAMPOS_DIR}/fields_database.parquet"):
    """
    Appends new field data to the parquet database.
    Deduplicates by 'session_id' to avoid repeated entries.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Exists read old data and add new row
    if os.path.exists(filename):
        df = pd.read_parquet(filename)

    else:
        return pd.DataFrame()

    return df


# -------------------------------
# Persistência parquet analítica
# -------------------------------

def append_dedup_parquet(df_new, filename, subset_keys):
    """Append em parquet com deduplicação por chave composta."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    if df_new is None or df_new.empty:
        return pd.DataFrame()

    if os.path.exists(filename):
        df_old = pd.read_parquet(filename)
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=subset_keys, keep="last")
    else:
        df_final = df_new.copy()

    df_final.to_parquet(filename, index=False)
    return df_final



def resolve_athlete_sk(df_metrics, base_dir=CLEANDATA_DIR, genero=None):
    """Resolve athlete_sk persistente a partir de atleta_id."""
    path = os.path.join(base_dir, "athletes.parquet")
    os.makedirs(base_dir, exist_ok=True)

    atletas = pd.Series(dtype="object")
    if df_metrics is not None and not df_metrics.empty and "atleta_id" in df_metrics.columns:
        atletas = pd.Series(df_metrics["atleta_id"].dropna().astype(str).unique())

    if os.path.exists(path):
        df_dim = pd.read_parquet(path)
    else:
        df_dim = pd.DataFrame(columns=["athlete_sk", "atleta_id", "genero", "ativo", "created_at", "updated_at"])

    if not df_dim.empty:
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str)

    existing = dict(zip(df_dim.get("atleta_id", pd.Series(dtype="object")), df_dim.get("athlete_sk", pd.Series(dtype="int64"))))
    next_id = 1 if df_dim.empty else int(pd.to_numeric(df_dim["athlete_sk"], errors="coerce").max()) + 1

    now_ts = pd.Timestamp.utcnow()
    new_rows = []
    for aid in atletas.tolist():
        if aid not in existing:
            existing[aid] = next_id
            new_rows.append(
                {
                    "athlete_sk": next_id,
                    "atleta_id": aid,
                    "genero": genero,
                    "ativo": True,
                    "created_at": now_ts,
                    "updated_at": now_ts,
                }
            )
            next_id += 1

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_dim = pd.concat([df_dim, df_new], ignore_index=True)
        df_dim.to_parquet(path, index=False)

    return existing



def resolve_session_sk(session_fingerprint, session_payload, base_dir=CLEANDATA_DIR):
    """Resolve session_sk persistente a partir do fingerprint da sessão."""
    path = os.path.join(base_dir, "sessions.parquet")
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame()

    if not df.empty and "session_fingerprint" in df.columns:
        mask = df["session_fingerprint"].astype(str) == str(session_fingerprint)
        if mask.any():
            return int(df.loc[mask, "session_sk"].iloc[0])

    next_id = 1 if df.empty else int(pd.to_numeric(df["session_sk"], errors="coerce").max()) + 1

    payload = dict(session_payload)
    payload["session_sk"] = next_id
    if "created_at" not in payload:
        payload["created_at"] = pd.Timestamp.utcnow()

    df_new = pd.DataFrame([payload])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_parquet(path, index=False)

    return next_id
