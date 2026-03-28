from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .constants import CLEANDATA_DIR, SELECOES_OPCOES


SELECOES_PATH = Path(CLEANDATA_DIR) / "selecoes.parquet"
SELECOES_COLUMNS = [
    "selection_sk",
    "codigo",
    "escalao",
    "genero",
    "ativo",
    "sort_order",
    "created_at",
    "updated_at",
]


def _parse_selection_label(label: str) -> tuple[str, str]:
    value = str(label).strip()
    if not value:
        return "", ""
    parts = value.rsplit(" ", 1)
    if len(parts) == 2 and parts[1] in {"M", "F"}:
        return parts[0], parts[1]
    return value, ""


def _default_selections_df() -> pd.DataFrame:
    now_ts = pd.Timestamp.utcnow()
    rows = []
    for idx, codigo in enumerate(SELECOES_OPCOES, start=1):
        escalao, genero = _parse_selection_label(codigo)
        rows.append(
            {
                "selection_sk": idx,
                "codigo": codigo,
                "escalao": escalao,
                "genero": genero,
                "ativo": True,
                "sort_order": idx,
                "created_at": now_ts,
                "updated_at": now_ts,
            }
        )
    return pd.DataFrame(rows, columns=SELECOES_COLUMNS)


def ensure_selections_table(base_dir: str = CLEANDATA_DIR) -> pd.DataFrame:
    os.makedirs(base_dir, exist_ok=True)
    path = Path(base_dir) / "selecoes.parquet"
    if path.exists():
        try:
            df = pd.read_parquet(path)
        except Exception:
            df = _default_selections_df()
    else:
        df = _default_selections_df()

    for col in SELECOES_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[SELECOES_COLUMNS].copy()
    if df.empty:
        df = _default_selections_df()

    df["codigo"] = df["codigo"].astype(str).str.strip()
    df = df[df["codigo"].ne("")].drop_duplicates(subset=["codigo"], keep="last")
    df["escalao"] = df["escalao"].astype("string").fillna("").str.strip()
    df["genero"] = df["genero"].astype("string").fillna("").str.strip()
    df["ativo"] = df["ativo"].fillna(True).astype(bool)
    df["selection_sk"] = pd.to_numeric(df["selection_sk"], errors="coerce")
    if df["selection_sk"].isna().any():
        df["selection_sk"] = range(1, len(df) + 1)
    df["selection_sk"] = df["selection_sk"].astype(int)
    df["sort_order"] = pd.to_numeric(df["sort_order"], errors="coerce")
    if df["sort_order"].isna().any():
        df["sort_order"] = range(1, len(df) + 1)
    df["sort_order"] = df["sort_order"].astype(int)
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce").fillna(pd.Timestamp.utcnow())
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce").fillna(pd.Timestamp.utcnow())
    df = df.sort_values(["sort_order", "codigo"], na_position="last").reset_index(drop=True)
    df.to_parquet(path, index=False)
    return df


def load_selections_df(base_dir: str = CLEANDATA_DIR, active_only: bool = True) -> pd.DataFrame:
    df = ensure_selections_table(base_dir=base_dir)
    if active_only:
        df = df[df["ativo"]].copy()
    return df.sort_values(["sort_order", "codigo"], na_position="last").reset_index(drop=True)


def load_selection_options(base_dir: str = CLEANDATA_DIR, active_only: bool = True) -> list[str]:
    df = load_selections_df(base_dir=base_dir, active_only=active_only)
    options = df["codigo"].astype(str).str.strip().tolist()
    return [opt for opt in options if opt]
