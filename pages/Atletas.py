import os
from pathlib import Path

import pandas as pd
import streamlit as st

from fpf_modules.constants import CLEANDATA_DIR
from fpf_modules.selections import load_selection_options


ATHLETES_PATH = Path(CLEANDATA_DIR) / "athletes.parquet"
ATHLETE_COLUMNS = [
    "athlete_sk",
    "atleta_id",
    "nome",
    "numero_camisola",
    "data_nascimento",
    "posicao",
    "pe_preferencial",
    "altura_cm",
    "peso_kg",
    "escalao",
    "selecao",
    "genero",
    "ativo",
    "created_at",
    "updated_at",
]
ATHLETE_POSITIONS = ["", "GR", "DD", "DE", "DC", "MD", "ME", "MC", "MDC", "MAC", "ED", "EE", "AV", "PL"]
ATHLETE_FEET = ["", "Direito", "Esquerdo", "Ambidestro"]
SELECTION_OPTIONS = load_selection_options()


def _empty_athletes_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ATHLETE_COLUMNS)


def _selection_to_genero(selecao: str) -> str:
    parts = str(selecao).split()
    return parts[-1] if parts and parts[-1] in ["M", "F"] else ""


def _selection_to_escalao(selecao: str) -> str:
    value = str(selecao).strip()
    if not value:
        return ""
    parts = value.rsplit(" ", 1)
    return parts[0] if len(parts) == 2 else value


def _load_athletes() -> pd.DataFrame:
    if not ATHLETES_PATH.exists():
        return _empty_athletes_df()

    df = pd.read_parquet(ATHLETES_PATH)
    for col in ATHLETE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ATHLETE_COLUMNS].copy()
    if not df.empty:
        df["atleta_id"] = df["atleta_id"].astype(str)
        df["numero_camisola"] = pd.to_numeric(df["numero_camisola"], errors="coerce")
        df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce").dt.date
        df["altura_cm"] = pd.to_numeric(df["altura_cm"], errors="coerce")
        df["peso_kg"] = pd.to_numeric(df["peso_kg"], errors="coerce")
        df["ativo"] = df["ativo"].fillna(True).astype(bool)
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    return df.sort_values(["ativo", "nome", "atleta_id"], ascending=[False, True, True], na_position="last")


def _save_athletes(df: pd.DataFrame) -> None:
    os.makedirs(CLEANDATA_DIR, exist_ok=True)
    save_df = df.copy()
    save_df["atleta_id"] = save_df["atleta_id"].astype(str).str.strip()
    save_df = save_df[save_df["atleta_id"].ne("")].drop_duplicates(subset=["atleta_id"], keep="last")
    save_df["numero_camisola"] = pd.to_numeric(save_df["numero_camisola"], errors="coerce")
    save_df["data_nascimento"] = pd.to_datetime(save_df["data_nascimento"], errors="coerce").dt.date
    save_df["altura_cm"] = pd.to_numeric(save_df["altura_cm"], errors="coerce")
    save_df["peso_kg"] = pd.to_numeric(save_df["peso_kg"], errors="coerce")
    save_df["ativo"] = save_df["ativo"].fillna(True).astype(bool)
    save_df["selecao"] = save_df["selecao"].astype(str).str.strip()
    save_df["genero"] = save_df["selecao"].map(_selection_to_genero)
    save_df["escalao"] = save_df["selecao"].map(_selection_to_escalao)
    save_df.to_parquet(ATHLETES_PATH, index=False)


def _next_athlete_sk(df: pd.DataFrame) -> int:
    if df.empty or "athlete_sk" not in df.columns:
        return 1
    values = pd.to_numeric(df["athlete_sk"], errors="coerce").dropna()
    return 1 if values.empty else int(values.max()) + 1


def _editor_view(df: pd.DataFrame) -> pd.DataFrame:
    display_cols = [
        "atleta_id",
        "nome",
        "numero_camisola",
        "data_nascimento",
        "posicao",
        "pe_preferencial",
        "altura_cm",
        "peso_kg",
        "selecao",
        "ativo",
    ]
    return st.data_editor(
        df[display_cols],
        hide_index=True,
        use_container_width=True,
        column_order=display_cols,
        num_rows="dynamic",
        column_config={
            "atleta_id": st.column_config.TextColumn("Atleta ID", required=True),
            "nome": st.column_config.TextColumn("Nome"),
            "numero_camisola": st.column_config.NumberColumn("Nº Camisola", min_value=1, max_value=99, step=1),
            "data_nascimento": st.column_config.DateColumn("Nascimento", format="DD/MM/YYYY"),
            "posicao": st.column_config.SelectboxColumn("Posição", options=ATHLETE_POSITIONS),
            "pe_preferencial": st.column_config.SelectboxColumn("Pé Preferencial", options=ATHLETE_FEET),
            "altura_cm": st.column_config.NumberColumn("Altura (cm)", min_value=0, max_value=260, step=1),
            "peso_kg": st.column_config.NumberColumn("Peso (kg)", min_value=0, max_value=200, step=1),
            "selecao": st.column_config.SelectboxColumn("Seleção", options=[""] + SELECTION_OPTIONS),
            "ativo": st.column_config.CheckboxColumn("Ativo"),
        },
        key="athletes_editor",
    )


def _merge_edited_rows(original_df: pd.DataFrame, edited_df: pd.DataFrame) -> pd.DataFrame:
    now_ts = pd.Timestamp.utcnow()
    original_meta = original_df.set_index("atleta_id")[["created_at", "updated_at"]].to_dict(orient="index") if not original_df.empty else {}

    merged = edited_df.copy()
    merged["atleta_id"] = merged["atleta_id"].astype(str).str.strip()
    merged = merged[merged["atleta_id"].ne("")].drop_duplicates(subset=["atleta_id"], keep="last")
    merged["created_at"] = merged["atleta_id"].map(lambda aid: original_meta.get(aid, {}).get("created_at", now_ts))
    merged["updated_at"] = now_ts
    merged["genero"] = merged["selecao"].map(_selection_to_genero)
    merged["escalao"] = merged["selecao"].map(_selection_to_escalao)

    for col in ATHLETE_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    return merged[ATHLETE_COLUMNS].copy()


st.title("Atletas")
st.caption("Consulta, edita e cria fichas de atleta para enriquecer a base analítica.")

athletes_df = _load_athletes()
tab_view, tab_insert = st.tabs(["Visualizar", "Inserir"])

with tab_view:
    st.subheader("Fichas registadas")
    edited_df = _editor_view(athletes_df if not athletes_df.empty else _empty_athletes_df())

    if st.button("Guardar alterações", type="primary", key="save_athletes_table"):
        final_df = _merge_edited_rows(athletes_df, edited_df)
        duplicate_ids = final_df["atleta_id"].duplicated(keep=False)
        if duplicate_ids.any():
            st.error("Existem atleta_id duplicados. Corrige-os antes de guardar.")
        else:
            _save_athletes(final_df)
            st.success("Fichas de atletas atualizadas com sucesso.")
            st.rerun()

with tab_insert:
    st.subheader("Novo atleta")
    with st.form("new_athlete_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        atleta_id = col1.text_input("Atleta ID")
        nome = col2.text_input("Nome")

        col3, col4, col5 = st.columns(3)
        data_nascimento = col3.date_input("Data de nascimento", value=None, format="DD/MM/YYYY")
        posicao = col4.selectbox("Posição", ATHLETE_POSITIONS)
        numero_camisola = col5.number_input("Nº Camisola", min_value=1, max_value=99, step=1, value=None)

        col6, col7, col8 = st.columns(3)
        pe_preferencial = col6.selectbox("Pé Preferencial", ATHLETE_FEET)
        altura_cm = col7.number_input("Altura (cm)", min_value=0, max_value=260, step=1, value=None)
        peso_kg = col8.number_input("Peso (kg)", min_value=0, max_value=200, step=1, value=None)

        col9, col10 = st.columns(2)
        selecao = col9.selectbox("Seleção", [""] + SELECTION_OPTIONS)
        ativo = col10.checkbox("Ativo", value=True)

        submitted = st.form_submit_button("Inserir atleta", type="primary")

    if submitted:
        atleta_id = str(atleta_id).strip()
        if not atleta_id:
            st.error("O campo Atleta ID é obrigatório.")
        elif not athletes_df.empty and athletes_df["atleta_id"].astype(str).eq(atleta_id).any():
            st.error("Já existe um atleta com esse Atleta ID.")
        else:
            now_ts = pd.Timestamp.utcnow()
            new_row = pd.DataFrame(
                [{
                    "athlete_sk": _next_athlete_sk(athletes_df),
                    "atleta_id": atleta_id,
                    "nome": nome or pd.NA,
                    "numero_camisola": numero_camisola,
                    "data_nascimento": data_nascimento,
                    "posicao": posicao,
                    "pe_preferencial": pe_preferencial,
                    "altura_cm": altura_cm,
                    "peso_kg": peso_kg,
                    "escalao": _selection_to_escalao(selecao),
                    "selecao": selecao,
                    "genero": _selection_to_genero(selecao),
                    "ativo": ativo,
                    "created_at": now_ts,
                    "updated_at": now_ts,
                }]
            )
            final_df = pd.concat([athletes_df, new_row], ignore_index=True)
            _save_athletes(final_df[ATHLETE_COLUMNS])
            st.success("Novo atleta criado com sucesso.")
            st.rerun()
