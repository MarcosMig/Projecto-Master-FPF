import pandas as pd
import streamlit as st

from fpf_modules.supabase_manager import (
    delete_table_rows,
    initialize_schema,
    read_field_reference,
    save_field_reference,
)


FIELD_DISPLAY_COLUMNS = [
    "field_fingerprint",
    "estadio",
    "campo_local",
    "cidade",
    "pais",
    "clat",
    "clon",
    "dist_x",
    "dist_y",
]


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _format_number(value, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _load_fields() -> pd.DataFrame:
    initialize_schema()
    df = read_field_reference()
    if df is None or df.empty:
        return pd.DataFrame(columns=FIELD_DISPLAY_COLUMNS)

    for col in FIELD_DISPLAY_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    for col in ["estadio", "campo_local", "cidade", "pais", "field_fingerprint"]:
        df[col] = df[col].map(_clean_text_value)

    for col in ["clat", "clon", "dist_x", "dist_y"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df.sort_values(
        by=["estadio", "campo_local", "cidade", "pais"],
        ascending=[True, True, True, True],
        na_position="last",
    ).reset_index(drop=True)


def _save_field_details(
    row: pd.Series,
    estadio: str,
    campo_local: str,
    cidade: str,
    pais: str,
) -> None:
    updated = row.copy()
    updated["estadio"] = _clean_text_value(estadio)
    updated["campo_local"] = _clean_text_value(campo_local)
    updated["cidade"] = _clean_text_value(cidade)
    updated["pais"] = _clean_text_value(pais)
    save_field_reference(pd.DataFrame([updated]))


def _delete_field(field_fingerprint: str) -> None:
    stats = delete_table_rows("fields", {"field_fingerprint": field_fingerprint})
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao eliminar campo.")


def _render_field_card(row: pd.Series) -> None:
    fingerprint = _clean_text_value(row.get("field_fingerprint"))
    title = _clean_text_value(row.get("estadio")) or _clean_text_value(row.get("campo_local")) or "Campo sem nome"
    subtitle_parts = [part for part in [
        _clean_text_value(row.get("campo_local")),
        _clean_text_value(row.get("cidade")),
        _clean_text_value(row.get("pais")),
    ] if part]
    subtitle = " | ".join(subtitle_parts) if subtitle_parts else "Sem local definido"
    edit_key = f"edit_field_{fingerprint}"

    with st.expander(title, expanded=False):
        st.caption(subtitle)

        info_col1, info_col2 = st.columns(2)
        info_col1.markdown(f"**Cidade:** {_clean_text_value(row.get('cidade')) or '-'}")
        info_col1.markdown(f"**País:** {_clean_text_value(row.get('pais')) or '-'}")
        info_col1.markdown(f"**Local:** {_clean_text_value(row.get('campo_local')) or '-'}")
        info_col2.markdown(f"**Latitude:** {_format_number(row.get('clat'), 6)}")
        info_col2.markdown(f"**Longitude:** {_format_number(row.get('clon'), 6)}")
        info_col2.markdown(
            f"**Dimensões:** {_format_number(row.get('dist_x'), 1)} x {_format_number(row.get('dist_y'), 1)} m"
        )
        st.caption(f"Fingerprint: {fingerprint or '-'}")

        if st.session_state.get(edit_key, False):
            st.divider()
            st.markdown("**Editar campo**")
            with st.form(f"edit_field_form_{fingerprint}"):
                form_col1, form_col2 = st.columns(2)
                estadio_edit = form_col1.text_input("Nome do campo", value=_clean_text_value(row.get("estadio")))
                campo_local_edit = form_col2.text_input("Local", value=_clean_text_value(row.get("campo_local")))
                form_col3, form_col4 = st.columns(2)
                cidade_edit = form_col3.text_input("Cidade", value=_clean_text_value(row.get("cidade")))
                pais_edit = form_col4.text_input("País", value=_clean_text_value(row.get("pais")))
                save_edit = st.form_submit_button("Guardar alterações", type="primary")

            if save_edit:
                _save_field_details(
                    row,
                    estadio_edit,
                    campo_local_edit,
                    cidade_edit,
                    pais_edit,
                )
                st.session_state[edit_key] = False
                st.success("Campo atualizado com sucesso.")
                st.rerun()

        st.divider()
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Editar campo", key=f"btn_edit_field_{fingerprint}"):
            st.session_state[edit_key] = not st.session_state.get(edit_key, False)
            st.rerun()
        if action_col2.button("Eliminar campo", key=f"btn_delete_field_{fingerprint}"):
            _delete_field(fingerprint)
            st.success("Campo eliminado com sucesso.")
            st.rerun()


st.title("Campos")
st.caption("Consulta, edita e elimina os campos guardados na base de dados.")

fields_df = _load_fields()

search_col1, search_col2 = st.columns([1.5, 1.0])
search_text = search_col1.text_input("Pesquisar", placeholder="Nome, cidade, país ou local")
country_filter = search_col2.text_input("Filtrar por país", placeholder="Ex.: Portugal")

view_df = fields_df.copy()
if search_text:
    search_value = search_text.strip().lower()
    mask = (
        view_df["estadio"].str.lower().str.contains(search_value, na=False)
        | view_df["campo_local"].str.lower().str.contains(search_value, na=False)
        | view_df["cidade"].str.lower().str.contains(search_value, na=False)
        | view_df["pais"].str.lower().str.contains(search_value, na=False)
    )
    view_df = view_df[mask].copy()

if country_filter:
    view_df = view_df[view_df["pais"].str.lower().eq(country_filter.strip().lower())].copy()

st.caption(f"Campos registados: {len(view_df)}")

if view_df.empty:
    st.info("Sem campos registados para este filtro.")
else:
    for _, field_row in view_df.iterrows():
        _render_field_card(field_row)
