import pandas as pd
import streamlit as st

from fpf_modules.reference_data import load_selection_reference
from fpf_modules.supabase_manager import initialize_schema, read_table


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _format_date(value) -> str:
    if pd.isna(value) or value in ("", None):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _load_session_reports() -> pd.DataFrame:
    try:
        initialize_schema()
        df = read_table("session_reports")
    except Exception:
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    for col in ["session_fingerprint", "selecao", "genero", "contexto", "jogo", "report_title", "report_txt"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].map(_clean_text_value)

    if "data" not in df.columns:
        df["data"] = pd.NaT
    df["data"] = pd.to_datetime(df["data"], errors="coerce")

    if "updated_at" not in df.columns:
        df["updated_at"] = pd.NaT
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")

    return df.sort_values(["data", "updated_at"], ascending=[False, False], na_position="last").reset_index(drop=True)


def _build_summary_df(selection_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = selection_df.copy()
    summary_df["Data"] = summary_df["data"].map(_format_date)
    summary_df["Contexto"] = summary_df["contexto"].replace("", "-")
    summary_df["Jogo"] = summary_df["jogo"].replace("", "-")
    summary_df["Atualizado"] = summary_df["updated_at"].map(_format_date)
    return summary_df[["Data", "Contexto", "Jogo", "Atualizado"]].reset_index(drop=True)


def _render_selection_section(selection_code: str, selection_df: pd.DataFrame) -> None:
    expander_label = f"{selection_code} ({len(selection_df)})"
    with st.expander(expander_label, expanded=False):
        st.markdown("**Sessões registadas**")
        st.dataframe(_build_summary_df(selection_df), hide_index=True, use_container_width=True)

        st.markdown("**Relatórios guardados**")
        for _, row in selection_df.iterrows():
            date_txt = _format_date(row.get("data"))
            contexto_txt = _clean_text_value(row.get("contexto")) or "Sessao"
            jogo_txt = _clean_text_value(row.get("jogo"))
            title_parts = [date_txt, contexto_txt]
            if jogo_txt:
                title_parts.append(jogo_txt)
            report_label = " | ".join(title_parts)

            with st.expander(report_label, expanded=False):
                meta_left, meta_right = st.columns(2)
                meta_left.markdown(f"**Data:** {date_txt}")
                meta_left.markdown(f"**Contexto:** {contexto_txt}")
                meta_right.markdown(f"**Selecao:** {_clean_text_value(row.get('selecao')) or '-'}")
                meta_right.markdown(f"**Jogo:** {jogo_txt or '-'}")
                st.code(_clean_text_value(row.get("report_txt")) or "Sem relatorio guardado.", language="text")


st.title("Jogos | Treinos")
st.caption("Consulta os jogos e treinos publicados por selecao e abre o relatorio tecnico guardado em cada sessao.")

reports_df = _load_session_reports()
selection_options = ["Todas"] + load_selection_reference(active_only=False)
context_options = ["Todos", "Jogo", "Treino"]

filter_col1, filter_col2 = st.columns(2)
selected_selection = filter_col1.selectbox("Selecao", options=selection_options)
selected_context = filter_col2.selectbox("Contexto", options=context_options)

view_df = reports_df.copy()
if not view_df.empty and selected_selection != "Todas":
    view_df = view_df[view_df["selecao"].eq(selected_selection)].copy()
if not view_df.empty and selected_context != "Todos":
    view_df = view_df[view_df["contexto"].eq(selected_context)].copy()

if view_df.empty:
    st.info(
        "Sem registos disponíveis. Depois de publicares uma sessão, o relatório técnico ficará guardado aqui."
    )
else:
    for selection_code in view_df["selecao"].drop_duplicates().tolist():
        selection_df = view_df[view_df["selecao"].eq(selection_code)].copy()
        _render_selection_section(selection_code, selection_df)
