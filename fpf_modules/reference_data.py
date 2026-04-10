from __future__ import annotations

import pandas as pd

from .constants import SELECOES_OPCOES
from .supabase_manager import initialize_schema, read_field_reference, read_selections_reference, read_table


ATHLETE_REFERENCE_COLUMNS = [
    "athlete_sk",
    "atleta_id",
    "nome",
    "data_nascimento",
    "posicao",
    "foto_url",
    "genero",
    "ativo",
]


def load_selection_reference(active_only: bool = True) -> list[str]:
    """Load selection options used by forms and filters."""
    try:
        df = read_selections_reference(active_only=active_only)
    except Exception:
        return [str(opt).strip() for opt in SELECOES_OPCOES if str(opt).strip()]

    if df is None or df.empty or "codigo" not in df.columns:
        return [str(opt).strip() for opt in SELECOES_OPCOES if str(opt).strip()]

    options = df["codigo"].astype(str).str.strip().tolist()
    options = [opt for opt in options if opt]
    return options or [str(opt).strip() for opt in SELECOES_OPCOES if str(opt).strip()]


def load_field_reference() -> pd.DataFrame:
    """Load saved fields reference data from the active persistence layer."""
    return read_field_reference()


def load_active_athletes_by_selection(selecao: str = "") -> pd.DataFrame:
    """Load active athlete reference data, optionally filtered by selection."""
    try:
        initialize_schema()
        df = read_table("athletes")
    except Exception:
        return pd.DataFrame(columns=ATHLETE_REFERENCE_COLUMNS)
    if df is None or df.empty:
        return pd.DataFrame(columns=ATHLETE_REFERENCE_COLUMNS)

    for col in ATHLETE_REFERENCE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ATHLETE_REFERENCE_COLUMNS].copy()
    if df.empty:
        return df

    df["atleta_id"] = df["atleta_id"].astype(str).str.strip()
    if "selecao" not in df.columns:
        df["selecao"] = pd.Series([""] * len(df), dtype="string")
    else:
        df["selecao"] = df["selecao"].astype("string").fillna("").str.strip()
    df["ativo"] = df["ativo"].fillna(True).astype(bool)
    df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce").dt.date

    df = df[df["atleta_id"].ne("") & df["ativo"]]
    if selecao:
        selecao_norm = str(selecao).strip()
        genero = ""
        if selecao_norm == "Masculino":
            genero = "M"
        elif selecao_norm == "Feminino":
            genero = "F"

        df_filtered = df[df["selecao"].eq(selecao_norm)] if "selecao" in df.columns else pd.DataFrame()
        if df_filtered.empty and not genero:
            genero_parts = selecao_norm.split()
            genero = genero_parts[-1] if genero_parts and genero_parts[-1] in ["M", "F"] else ""
        if df_filtered.empty and genero:
            df_filtered = df[df["genero"].astype("string").fillna("").str.strip().eq(genero)]
        df = df_filtered

    return (
        df.sort_values(["nome", "atleta_id"], na_position="last")
        .drop_duplicates(subset=["atleta_id"], keep="last")
        .reset_index(drop=True)
    )
