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
        # Ensure consistent dtypes between old and new data
        for col in df_new.columns:
            if col in df_old.columns:
                if df_old[col].dtype != df_new[col].dtype:
                    # Convert new data to match old data dtype
                    if df_old[col].dtype == 'datetime64[ns, UTC]' or df_old[col].dtype == 'datetime64[ns]':
                        df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
                    elif df_old[col].dtype == 'object':
                        df_new[col] = df_new[col].astype(str)
                    elif df_old[col].dtype == 'int64':
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce').astype('Int64')
                    elif df_old[col].dtype == 'float64':
                        df_new[col] = pd.to_numeric(df_new[col], errors='coerce')
                    elif df_old[col].dtype == 'bool':
                        df_new[col] = df_new[col].astype(bool)
        
        df_final = pd.concat([df_old, df_new], ignore_index=True)
        df_final = df_final.drop_duplicates(subset=subset_keys, keep="last")
    else:
        df_final = df_new.copy()

    df_final.to_parquet(filename, index=False)
    return df_final



def resolve_athlete_sk(df_metrics, base_dir=CLEANDATA_DIR, genero=None, athlete_profiles=None, selecao=None):
    """Resolve athlete_sk persistente a partir de atleta_id."""
    path = os.path.join(base_dir, "athletes.parquet")
    os.makedirs(base_dir, exist_ok=True)

    atletas = pd.Series(dtype="object")
    if df_metrics is not None and not df_metrics.empty and "atleta_id" in df_metrics.columns:
        atletas = pd.Series(df_metrics["atleta_id"].dropna().astype(str).unique())

    if os.path.exists(path):
        df_dim = pd.read_parquet(path)
    else:
        df_dim = pd.DataFrame(columns=[
            "athlete_sk", "atleta_id", "nome", "data_nascimento", "posicao",
            "pe_preferencial", "altura_cm", "peso_kg", "escalao", "selecao",
            "genero", "ativo", "created_at", "updated_at"
        ])

    if not df_dim.empty:
        df_dim["atleta_id"] = df_dim["atleta_id"].astype(str)

    profiles_map = {}
    if athlete_profiles is not None and not athlete_profiles.empty and "atleta_id" in athlete_profiles.columns:
        df_profiles = athlete_profiles.copy()
        df_profiles["atleta_id"] = df_profiles["atleta_id"].astype(str)
        profiles_map = df_profiles.set_index("atleta_id").to_dict(orient="index")

    existing = dict(zip(df_dim.get("atleta_id", pd.Series(dtype="object")), df_dim.get("athlete_sk", pd.Series(dtype="int64"))))
    next_id = 1 if df_dim.empty else int(pd.to_numeric(df_dim["athlete_sk"], errors="coerce").max()) + 1

    now_ts = pd.Timestamp.utcnow()
    new_rows = []
    for aid in atletas.tolist():
        profile = profiles_map.get(aid, {})
        if aid not in existing:
            existing[aid] = next_id
            new_rows.append(
                {
                    "athlete_sk": next_id,
                    "atleta_id": aid,
                    "nome": profile.get("nome"),
                    "data_nascimento": profile.get("data_nascimento"),
                    "posicao": profile.get("posicao"),
                    "pe_preferencial": profile.get("pe_preferencial"),
                    "altura_cm": profile.get("altura_cm"),
                    "peso_kg": profile.get("peso_kg"),
                    "escalao": profile.get("escalao"),
                    "selecao": profile.get("selecao", selecao),
                    "genero": profile.get("genero", genero),
                    "ativo": True,
                    "created_at": now_ts,
                    "updated_at": now_ts,
                }
            )
            next_id += 1
        elif profile:
            idx = df_dim["atleta_id"].astype(str) == aid
            for col in ["nome", "data_nascimento", "posicao", "pe_preferencial", "altura_cm", "peso_kg", "escalao", "selecao"]:
                if col in df_dim.columns and profile.get(col) is not None and not pd.isna(profile.get(col)):
                    df_dim.loc[idx, col] = profile.get(col)
            if "genero" in df_dim.columns and genero is not None:
                df_dim.loc[idx, "genero"] = df_dim.loc[idx, "genero"].fillna(profile.get("genero", genero))
            if "updated_at" in df_dim.columns:
                df_dim.loc[idx, "updated_at"] = now_ts

    if new_rows:
        df_new = pd.DataFrame(new_rows)
        df_dim = pd.concat([df_dim, df_new], ignore_index=True)
    if new_rows or profiles_map:
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


# ==== DuckDB helpers for analytic schema =====
from .duckdb_utils import connect as _ddb_connect, initialize_schema as _ddb_init, insert_metrics as _ddb_insert


def resolve_game_sk(game_payload, base_dir=CLEANDATA_DIR):
    """Resolve game_sk persistente a partir dos dados de um jogo.

    `game_payload` deve conter os campos únicos que identificam uma partida
    (por exemplo, data + adversário). A função devolve a chave surrogate e
    grava um parquet "games.parquet" no `base_dir`.
    """
    path = os.path.join(base_dir, "games.parquet")
    os.makedirs(base_dir, exist_ok=True)

    if os.path.exists(path):
        df = pd.read_parquet(path)
    else:
        df = pd.DataFrame()

    # se já existir manifestação idêntica retorna a chave
    if not df.empty:
        mask = pd.Series([True] * len(df))
        for k, v in game_payload.items():
            if k in df.columns:
                mask &= df[k] == v
        if mask.any():
            return int(df.loc[mask, "game_sk"].iloc[0])

    next_id = 1 if df.empty else int(pd.to_numeric(df["game_sk"], errors="coerce").max()) + 1

    rec = dict(game_payload)
    rec["game_sk"] = next_id
    if "game_date" in rec and pd.isna(rec.get("game_date")):
        rec["game_date"] = pd.NaT

    df_new = pd.DataFrame([rec])
    df = pd.concat([df, df_new], ignore_index=True)
    df.to_parquet(path, index=False)

    return next_id


def ensure_duckdb(db_file=None):
    """Return an open DuckDB connection with schema initialised.

    Args:
        db_file: optional path to duckdb file; defaults to CLEANDATA_DIR/fpf.duckdb.
    """
    if db_file is None:
        db_file = os.path.join(CLEANDATA_DIR, "fpf.duckdb")
    con = _ddb_connect(db_file)
    _ddb_init(con)
    return con


def write_metrics(metrics_df, db_file=None):
    """Persist a metrics dataframe into the DuckDB analytics schema."""
    con = ensure_duckdb(db_file)
    _ddb_insert(con, metrics_df)
    # optional: close connection
    con.close()
    return None


def write_session_data(
    df_perf, df_qc, df_samples, df_athlete_session,
    db_file=None
):
    """Persist performance, quality, samples, and athlete_session DataFrames to DuckDB.
    
    This is the main integration point for the analytic pipeline.
    Uses upsert logic (update if exists via PK, insert if new) to avoid duplicates.
    
    Returns:
        dict with integration stats
    """
    from .duckdb_utils import create_analytics_tables, insert_table
    
    try:
        con = ensure_duckdb(db_file)
    except ImportError as e:
        raise ImportError(
            f"DuckDB não está instalado. Execute no terminal:\n"
            f"pip install duckdb\n\n"
            f"Erro original: {str(e)}"
        )
    
    create_analytics_tables(con)
    
    stats = {
        'performance_metrics': None,
        'quality_metrics': None,
        'samples': None,
        'athlete_session': None,
    }
    
    # Insert into analytics tables with upsert logic
    if df_perf is not None and not df_perf.empty:
        stats['performance_metrics'] = insert_table(
            con, "performance_metrics", df_perf,
            pk_columns=["session_sk", "athlete_sk", "phase_id"]
        )
    
    if df_qc is not None and not df_qc.empty:
        stats['quality_metrics'] = insert_table(
            con, "quality_metrics", df_qc,
            pk_columns=["session_sk", "athlete_sk", "phase_id"]
        )
    
    if df_samples is not None and not df_samples.empty:
        stats['samples'] = insert_table(
            con, "samples", df_samples,
            pk_columns=["session_sk", "athlete_sk", "phase_id", "time"]
        )
    
    if df_athlete_session is not None and not df_athlete_session.empty:
        stats['athlete_session'] = insert_table(
            con, "athlete_session", df_athlete_session,
            pk_columns=["session_sk", "athlete_sk"]
        )
    
    con.close()
    return stats
