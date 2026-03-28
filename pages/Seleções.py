from pathlib import Path

import pandas as pd
import streamlit as st

from fpf_modules.constants import CLEANDATA_DIR
from fpf_modules.selections import SELECOES_COLUMNS, ensure_selections_table


SELECOES_PATH = Path(CLEANDATA_DIR) / "selecoes.parquet"


def _save_selections(df: pd.DataFrame) -> None:
    save_df = df.copy()
    for col in SELECOES_COLUMNS:
        if col not in save_df.columns:
            save_df[col] = pd.NA

    save_df = save_df[SELECOES_COLUMNS].copy()
    save_df["codigo"] = save_df["codigo"].astype(str).str.strip()
    save_df = save_df[save_df["codigo"].ne("")].drop_duplicates(subset=["codigo"], keep="last")
    save_df["escalao"] = save_df["escalao"].astype("string").fillna("").str.strip()
    save_df["genero"] = save_df["genero"].astype("string").fillna("").str.strip()
    save_df["ativo"] = save_df["ativo"].fillna(True).astype(bool)
    save_df["selection_sk"] = pd.to_numeric(save_df["selection_sk"], errors="coerce")
    save_df["sort_order"] = pd.to_numeric(save_df["sort_order"], errors="coerce")

    if save_df["selection_sk"].isna().any():
        next_ids = []
        current_max = int(save_df["selection_sk"].dropna().max()) if save_df["selection_sk"].dropna().any() else 0
        for value in save_df["selection_sk"]:
            if pd.isna(value):
                current_max += 1
                next_ids.append(current_max)
            else:
                next_ids.append(int(value))
        save_df["selection_sk"] = next_ids
    else:
        save_df["selection_sk"] = save_df["selection_sk"].astype(int)

    if save_df["sort_order"].isna().any():
        save_df["sort_order"] = range(1, len(save_df) + 1)
    save_df["sort_order"] = save_df["sort_order"].astype(int)
    save_df["created_at"] = pd.to_datetime(save_df["created_at"], errors="coerce").fillna(pd.Timestamp.utcnow())
    save_df["updated_at"] = pd.Timestamp.utcnow()
    save_df = save_df.sort_values(["sort_order", "codigo"], na_position="last").reset_index(drop=True)
    save_df.to_parquet(SELECOES_PATH, index=False)


st.title("Seleções")
st.caption("Tabela mestre de seleções usada pelos formulários e filtros da aplicação.")

selecoes_df = ensure_selections_table()

edited_df = st.data_editor(
    selecoes_df[SELECOES_COLUMNS],
    hide_index=True,
    use_container_width=True,
    num_rows="dynamic",
    column_config={
        "selection_sk": st.column_config.NumberColumn("ID Interno", disabled=True),
        "codigo": st.column_config.TextColumn("Código", required=True),
        "escalao": st.column_config.TextColumn("Escalão"),
        "genero": st.column_config.SelectboxColumn("Género", options=["", "M", "F"]),
        "ativo": st.column_config.CheckboxColumn("Ativa"),
        "sort_order": st.column_config.NumberColumn("Ordem", min_value=1, step=1),
        "created_at": st.column_config.DatetimeColumn("Criada", disabled=True),
        "updated_at": st.column_config.DatetimeColumn("Atualizada", disabled=True),
    },
    key="selections_editor",
)

if st.button("Guardar seleções", type="primary"):
    _save_selections(edited_df)
    st.success("Tabela de seleções atualizada com sucesso.")
    st.rerun()
