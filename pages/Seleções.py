import pandas as pd
import streamlit as st

from fpf_modules.selections import SELECOES_COLUMNS, ensure_selections_table
from fpf_modules.supabase_manager import (
    initialize_schema,
    read_selections_reference,
    read_table,
    save_selections_reference,
    sync_selections_reference,
)


ATHLETE_COLUMNS = [
    "atleta_id",
    "nome",
    "posicao",
    "data_nascimento",
    "genero",
    "ativo",
    "selecao",
]

REPORT_COLUMNS = [
    "session_fingerprint",
    "data",
    "selecao",
    "genero",
    "contexto",
    "jogo",
    "report_title",
    "report_txt",
    "updated_at",
]


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _format_birth_date(value) -> str:
    if pd.isna(value) or value in ("", None):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _format_date(value) -> str:
    if pd.isna(value) or value in ("", None):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _load_selections() -> pd.DataFrame:
    try:
        df = sync_selections_reference(include_default=True)
    except Exception:
        df = ensure_selections_table()

    for col in SELECOES_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[SELECOES_COLUMNS].copy()


def _load_athletes() -> pd.DataFrame:
    try:
        initialize_schema()
        df = read_table("athletes")
    except Exception:
        return pd.DataFrame(columns=ATHLETE_COLUMNS)

    if df is None or df.empty:
        return pd.DataFrame(columns=ATHLETE_COLUMNS)

    for col in ATHLETE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ATHLETE_COLUMNS].copy()
    for col in ["atleta_id", "nome", "posicao", "genero", "selecao"]:
        df[col] = df[col].map(_clean_text_value)
    df["ativo"] = df["ativo"].fillna(True).astype(bool)
    df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce")
    return df.sort_values(["selecao", "nome", "atleta_id"], na_position="last").reset_index(drop=True)


def _load_reports() -> pd.DataFrame:
    try:
        initialize_schema()
        df = read_table("session_reports")
    except Exception:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    if df is None or df.empty:
        return pd.DataFrame(columns=REPORT_COLUMNS)

    for col in REPORT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[REPORT_COLUMNS].copy()
    for col in ["session_fingerprint", "selecao", "genero", "contexto", "jogo", "report_title", "report_txt"]:
        df[col] = df[col].map(_clean_text_value)
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    return df.sort_values(["data", "updated_at"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _build_selection_athletes_df(athletes_df: pd.DataFrame, codigo: str) -> pd.DataFrame:
    if athletes_df.empty:
        return pd.DataFrame(columns=["Nome", "Posicao", "Nascimento", "Genero", "Ativo"])

    view_df = athletes_df[athletes_df["selecao"].eq(codigo)].copy()
    if view_df.empty:
        return pd.DataFrame(columns=["Nome", "Posicao", "Nascimento", "Genero", "Ativo"])

    genero_map = {"M": "Masculino", "F": "Feminino"}
    view_df["Genero"] = view_df["genero"].map(lambda value: genero_map.get(value, value or "-"))
    view_df["Ativo"] = view_df["ativo"].map(lambda value: "Sim" if bool(value) else "Nao")
    view_df["Nascimento"] = view_df["data_nascimento"].map(_format_birth_date)

    return (
        view_df.rename(columns={"nome": "Nome", "posicao": "Posicao"})[
            ["Nome", "Posicao", "Nascimento", "Genero", "Ativo"]
        ]
        .sort_values(["Nome"], na_position="last")
        .reset_index(drop=True)
    )


def _build_selection_reports_df(reports_df: pd.DataFrame, codigo: str) -> pd.DataFrame:
    if reports_df.empty:
        return pd.DataFrame(columns=["Data", "Contexto", "Jogo", "Atualizado"])

    view_df = reports_df[reports_df["selecao"].eq(codigo)].copy()
    if view_df.empty:
        return pd.DataFrame(columns=["Data", "Contexto", "Jogo", "Atualizado"])

    view_df["Data"] = view_df["data"].map(_format_date)
    view_df["Contexto"] = view_df["contexto"].replace("", "-")
    view_df["Jogo"] = view_df["jogo"].replace("", "-")
    view_df["Atualizado"] = view_df["updated_at"].map(_format_date)
    return view_df[["Data", "Contexto", "Jogo", "Atualizado"]].reset_index(drop=True)


def _render_report_details(reports_df: pd.DataFrame, codigo: str) -> None:
    selection_reports_df = reports_df[reports_df["selecao"].eq(codigo)].copy()
    if selection_reports_df.empty:
        st.caption("Sem jogos ou treinos registados para esta selecao.")
        return

    st.dataframe(
        _build_selection_reports_df(reports_df, codigo),
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("**Relatórios guardados**")
    for _, row in selection_reports_df.iterrows():
        date_txt = _format_date(row.get("data"))
        contexto_txt = _clean_text_value(row.get("contexto")) or "Sessao"
        jogo_txt = _clean_text_value(row.get("jogo"))
        label_parts = [date_txt, contexto_txt]
        if jogo_txt:
            label_parts.append(jogo_txt)
        report_label = " | ".join(label_parts)

        with st.expander(report_label, expanded=False):
            meta_left, meta_right = st.columns(2)
            meta_left.markdown(f"**Data:** {date_txt}")
            meta_left.markdown(f"**Contexto:** {contexto_txt}")
            meta_right.markdown(f"**Selecao:** {_clean_text_value(row.get('selecao')) or '-'}")
            meta_right.markdown(f"**Jogo:** {jogo_txt or '-'}")
            st.code(_clean_text_value(row.get("report_txt")) or "Sem relatorio guardado.", language="text")


def _render_selection_expanders(selecoes_df: pd.DataFrame, athletes_df: pd.DataFrame, reports_df: pd.DataFrame) -> None:
    if selecoes_df.empty:
        st.info("Sem selecoes registadas.")
        return

    for _, row in selecoes_df.iterrows():
        codigo = _clean_text_value(row.get("codigo"))
        if not codigo:
            continue

        athletes_view_df = _build_selection_athletes_df(athletes_df, codigo)
        selection_reports_df = reports_df[reports_df["selecao"].eq(codigo)].copy() if not reports_df.empty else pd.DataFrame()
        expander_label = codigo

        with st.expander(expander_label, expanded=False):
            with st.expander("Jogos | Treinos", expanded=False):
                _render_report_details(reports_df, codigo)

            with st.expander("Atletas", expanded=False):
                if athletes_view_df.empty:
                    st.caption("Sem atletas afetos a esta selecao.")
                else:
                    st.dataframe(athletes_view_df, hide_index=True, use_container_width=True)


st.title("Seleções")
st.caption("Consulta cada selecao com os respetivos jogos, treinos e atletas afetos.")

selecoes_df = _load_selections()
athletes_df = _load_athletes()
reports_df = _load_reports()

tab_view, tab_manage = st.tabs(["Visualizar", "Gerir"])

with tab_view:
    st.subheader("Seleções Registadas")
    st.caption(
        f"Seleções: {len(selecoes_df)} | "
        f"Atletas com seleção atribuída: {int(athletes_df['selecao'].ne('').sum()) if not athletes_df.empty else 0} | "
        f"Registos guardados: {len(reports_df)}"
    )
    _render_selection_expanders(selecoes_df, athletes_df, reports_df)

with tab_manage:
    st.subheader("Tabela Mestre")
    edited_df = st.data_editor(
        selecoes_df,
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

    col_save, col_sync = st.columns(2)

    with col_save:
        if st.button("Guardar seleções", type="primary", use_container_width=True):
            save_selections_reference(edited_df)
            st.success("Tabela de seleções atualizada na base de dados.")
            st.rerun()

    with col_sync:
        if st.button("Sincronizar com atletas", use_container_width=True):
            sync_selections_reference(include_default=True)
            st.success("Seleções sincronizadas com os códigos existentes nos atletas.")
            st.rerun()

    try:
        current_df = read_selections_reference(active_only=False)
        st.caption(f"Seleções na base de dados: {len(current_df)}")
    except Exception:
        pass
