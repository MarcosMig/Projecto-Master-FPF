import pandas as pd
import streamlit as st

from fpf_modules.supabase_manager import (
    delete_public_storage_url,
    delete_table_rows,
    initialize_schema,
    insert_or_update_table,
    read_table,
    upload_image_to_storage,
)


ATHLETE_COLUMNS = [
    "athlete_sk",
    "atleta_id",
    "nome",
    "foto_url",
    "data_nascimento",
    "posicao",
    "genero",
    "ativo",
]
ATHLETE_UPLOAD_COLUMNS = [
    "atleta_id",
    "nome",
    "foto_url",
    "data_nascimento",
    "posicao",
    "genero",
    "ativo",
]
ATHLETE_POSITIONS = ["", "GR", "DD", "DE", "DC", "MD", "ME", "MC", "MDC", "MAC", "ED", "EE", "AV", "PL"]
GENDER_OPTIONS = ["Masculino", "Feminino"]
ATHLETE_PHOTO_BUCKET = "athlete-photos"
ATHLETE_PHOTO_UPLOADER_KEY = "athlete_photo_uploader"


def _empty_athletes_df() -> pd.DataFrame:
    return pd.DataFrame(columns=ATHLETE_COLUMNS)


def _clean_text_value(value):
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _split_full_name(full_name: str) -> tuple[str, str]:
    value = _clean_text_value(full_name)
    if not value:
        return "", ""
    parts = value.split()
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], " ".join(parts[1:])


def _compose_full_name(nome: str, sobrenome: str) -> str:
    return " ".join(part for part in [_clean_text_value(nome), _clean_text_value(sobrenome)] if part).strip()


def _genero_label_to_code(value: str) -> str:
    mapping = {
        "Masculino": "M",
        "Feminino": "F",
        "M": "M",
        "F": "F",
    }
    return mapping.get(_clean_text_value(value), "")


def _genero_code_to_label(value: str) -> str:
    mapping = {
        "M": "Masculino",
        "F": "Feminino",
    }
    return mapping.get(_clean_text_value(value), "")


def _generate_internal_atleta_id(df: pd.DataFrame) -> str:
    if df is None or df.empty or "atleta_id" not in df.columns:
        return "ATH-00001"

    existing_ids = df["atleta_id"].map(_clean_text_value)
    numeric_suffixes = (
        existing_ids.str.extract(r"ATH-(\d+)", expand=False)
        .dropna()
        .astype(int)
    )
    next_number = 1 if numeric_suffixes.empty else int(numeric_suffixes.max()) + 1
    return f"ATH-{next_number:05d}"


def _load_athletes() -> pd.DataFrame:
    initialize_schema()
    df = read_table("athletes")
    if df is None or df.empty:
        return _empty_athletes_df()

    for col in ATHLETE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[ATHLETE_COLUMNS].copy()
    if not df.empty:
        for col in ["atleta_id", "nome", "posicao", "genero", "foto_url"]:
            df[col] = df[col].map(_clean_text_value)
        df["data_nascimento"] = pd.to_datetime(df["data_nascimento"], errors="coerce").dt.date
        df["ativo"] = df["ativo"].fillna(True).astype(bool)

    return df.sort_values(["ativo", "nome", "atleta_id"], ascending=[False, True, True], na_position="last")


def _save_athletes(df: pd.DataFrame) -> None:
    initialize_schema()
    save_df = df.copy()
    for col in ["atleta_id", "nome", "posicao", "genero", "foto_url"]:
        save_df[col] = save_df[col].map(_clean_text_value)
    save_df = save_df[save_df["atleta_id"].ne("")].drop_duplicates(subset=["atleta_id"], keep="last")
    save_df["data_nascimento"] = pd.to_datetime(save_df["data_nascimento"], errors="coerce").dt.date
    save_df["ativo"] = save_df["ativo"].fillna(True).astype(bool)
    save_df["genero"] = save_df["genero"].map(_genero_label_to_code)

    stats = insert_or_update_table(
        "athletes",
        save_df[ATHLETE_UPLOAD_COLUMNS],
        pk_columns=["atleta_id"],
    )
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao guardar atletas na base de dados.")


def _format_birth_date(value) -> str:
    if pd.isna(value) or value in ("", None):
        return "-"
    try:
        return pd.to_datetime(value).strftime("%d/%m/%Y")
    except Exception:
        return str(value)


def _delete_athlete(atleta_id: str, foto_url: str) -> None:
    stats = delete_table_rows("athletes", {"atleta_id": atleta_id})
    if not stats.get("success", False):
        raise RuntimeError(stats.get("error") or "Falha ao eliminar atleta.")
    delete_public_storage_url(ATHLETE_PHOTO_BUCKET, foto_url)


def _save_athlete_photo(row: pd.Series, foto) -> None:
    atleta_id = _clean_text_value(row.get("atleta_id"))
    if not atleta_id:
        raise RuntimeError("Nao foi possivel identificar o atleta para atualizar a foto.")

    extension = foto.name.rsplit(".", 1)[-1].lower() if "." in foto.name else "jpg"
    object_path = f"{atleta_id}/{atleta_id}.{extension}"
    foto_url = upload_image_to_storage(
        ATHLETE_PHOTO_BUCKET,
        object_path,
        foto.getvalue(),
        foto.type,
    )

    updated_row = pd.DataFrame(
        [{
            "atleta_id": atleta_id,
            "nome": _clean_text_value(row.get("nome")),
            "foto_url": foto_url,
            "data_nascimento": row.get("data_nascimento"),
            "posicao": _clean_text_value(row.get("posicao")),
            "genero": _clean_text_value(row.get("genero")),
            "ativo": bool(row.get("ativo")),
        }]
    )
    _save_athletes(updated_row)


def _save_athlete_details(
    row: pd.Series,
    nome: str,
    sobrenome: str,
    data_nascimento,
    genero_label: str,
    posicao: str,
    ativo: bool,
) -> None:
    atleta_id = _clean_text_value(row.get("atleta_id"))
    if not atleta_id:
        raise RuntimeError("Nao foi possivel identificar o atleta para atualizar.")

    nome = _clean_text_value(nome)
    sobrenome = _clean_text_value(sobrenome)
    genero = _genero_label_to_code(genero_label)

    if not nome:
        raise RuntimeError("O campo Nome e obrigatorio.")
    if not sobrenome:
        raise RuntimeError("O campo Sobrenome e obrigatorio.")
    if not genero:
        raise RuntimeError("O campo Genero e obrigatorio.")

    updated_row = pd.DataFrame(
        [{
            "atleta_id": atleta_id,
            "nome": _compose_full_name(nome, sobrenome),
            "foto_url": _clean_text_value(row.get("foto_url")),
            "data_nascimento": data_nascimento,
            "posicao": _clean_text_value(posicao),
            "genero": genero,
            "ativo": bool(ativo),
        }]
    )
    _save_athletes(updated_row)


def _render_athlete_ficha(row: pd.Series) -> None:
    athlete_name = _clean_text_value(row.get("nome")) or _clean_text_value(row.get("atleta_id")) or "Atleta"
    with st.expander(athlete_name, expanded=False):
        left_col, right_col = st.columns([1.0, 1.8], gap="large")

        with left_col:
            foto_url = _clean_text_value(row.get("foto_url"))
            if foto_url:
                st.image(foto_url, use_container_width=True)
            else:
                st.caption("Sem foto registada.")
            foto_update = st.file_uploader(
                "Atualizar foto",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"update_photo_{_clean_text_value(row.get('atleta_id'))}",
            )
            if foto_update is not None:
                if st.button("Guardar foto", key=f"save_photo_{_clean_text_value(row.get('atleta_id'))}"):
                    _save_athlete_photo(row, foto_update)
                    st.success("Foto atualizada com sucesso.")
                    st.rerun()

        with right_col:
            nome, sobrenome = _split_full_name(row.get("nome"))
            genero_label = _genero_code_to_label(row.get("genero")) or "-"
            ativo_label = "Sim" if bool(row.get("ativo")) else "Nao"

            info_left, info_right = st.columns(2)
            info_left.markdown(f"**Nome:** {nome or '-'}")
            info_left.markdown(f"**Sobrenome:** {sobrenome or '-'}")
            info_left.markdown(f"**Nascimento:** {_format_birth_date(row.get('data_nascimento'))}")
            info_right.markdown(f"**Genero:** {genero_label}")
            info_right.markdown(f"**Posicao:** {_clean_text_value(row.get('posicao')) or '-'}")
            info_right.markdown(f"**Ativo:** {ativo_label}")

            st.caption(f"ID interno: {_clean_text_value(row.get('atleta_id')) or '-'}")

            action_col1, action_col2 = st.columns(2)
            edit_key = f"edit_mode_{_clean_text_value(row.get('atleta_id'))}"

            if action_col1.button("Editar atleta", key=f"edit_{_clean_text_value(row.get('atleta_id'))}"):
                st.session_state[edit_key] = not st.session_state.get(edit_key, False)
                st.rerun()

            if action_col2.button("Eliminar atleta", key=f"delete_{_clean_text_value(row.get('atleta_id'))}"):
                _delete_athlete(_clean_text_value(row.get("atleta_id")), _clean_text_value(row.get("foto_url")))
                st.success("Atleta eliminado com sucesso.")
                st.rerun()

            if st.session_state.get(edit_key, False):
                st.markdown("**Editar atleta**")
                default_nome, default_sobrenome = _split_full_name(row.get("nome"))
                with st.form(f"edit_athlete_form_{_clean_text_value(row.get('atleta_id'))}"):
                    form_col1, form_col2 = st.columns(2)
                    nome_edit = form_col1.text_input("Nome", value=default_nome)
                    sobrenome_edit = form_col2.text_input("Sobrenome", value=default_sobrenome)

                    form_col3, form_col4, form_col5 = st.columns(3)
                    data_edit = form_col3.date_input(
                        "Data de nascimento",
                        value=row.get("data_nascimento") if pd.notna(row.get("data_nascimento")) else None,
                        format="DD/MM/YYYY",
                    )
                    genero_edit = form_col4.selectbox(
                        "Genero",
                        GENDER_OPTIONS,
                        index=GENDER_OPTIONS.index(genero_label) if genero_label in GENDER_OPTIONS else 0,
                    )
                    posicao_edit = form_col5.selectbox(
                        "Posicao",
                        ATHLETE_POSITIONS,
                        index=ATHLETE_POSITIONS.index(_clean_text_value(row.get("posicao")))
                        if _clean_text_value(row.get("posicao")) in ATHLETE_POSITIONS else 0,
                    )
                    ativo_edit = st.checkbox("Ativo", value=bool(row.get("ativo")))

                    save_edit = st.form_submit_button("Guardar alterações", type="primary")

                if save_edit:
                    _save_athlete_details(
                        row,
                        nome_edit,
                        sobrenome_edit,
                        data_edit,
                        genero_edit,
                        posicao_edit,
                        ativo_edit,
                    )
                    st.session_state[edit_key] = False
                    st.success("Atleta atualizado com sucesso.")
                    st.rerun()


st.title("Atletas")
st.caption("Consulta e cria fichas base de atleta para enriquecer a base analitica.")

athletes_df = _load_athletes()
tab_view, tab_insert = st.tabs(["Visualizar", "Inserir"])

with tab_view:
    st.subheader("Atletas Registados")
    gender_filter_options = ["Todos", "Masculino", "Feminino"]
    selected_gender_label = st.selectbox(
        "Genero",
        options=gender_filter_options,
        key="athletes_view_gender_filter",
    )

    if athletes_df.empty or selected_gender_label == "Todos":
        view_df = athletes_df if not athletes_df.empty else _empty_athletes_df()
    else:
        gender_code = _genero_label_to_code(selected_gender_label)
        view_df = athletes_df[athletes_df["genero"].map(_clean_text_value).eq(gender_code)].copy()

    if view_df.empty:
        st.info("Sem atletas registados para este filtro.")
    else:
        for _, athlete_row in view_df.reset_index(drop=True).iterrows():
            _render_athlete_ficha(athlete_row)

with tab_insert:
    st.subheader("Novo atleta")
    with st.form("new_athlete_form", clear_on_submit=True):
        photo_col, details_col = st.columns([1.05, 1.95], gap="large")

        with photo_col:
            st.markdown("**Foto do atleta**")
            photo_preview = st.empty()
            foto = st.file_uploader(
                "Upload da foto",
                type=["png", "jpg", "jpeg", "webp"],
                key=ATHLETE_PHOTO_UPLOADER_KEY,
                label_visibility="collapsed",
            )
            if foto is not None:
                photo_preview.image(foto, use_container_width=True)
            else:
                photo_preview.caption("Sem foto carregada.")
            st.caption("Upload da foto")

        with details_col:
            st.markdown("**Dados do atleta**")
            col1, col2 = st.columns(2)
            nome = col1.text_input("Nome")
            sobrenome = col2.text_input("Sobrenome")

            col3, col4, col5 = st.columns(3)
            data_nascimento = col3.date_input("Data de nascimento", value=None, format="DD/MM/YYYY")
            genero_label = col4.selectbox("Genero", GENDER_OPTIONS)
            posicao = col5.selectbox("Posicao", ATHLETE_POSITIONS)

            ativo = st.checkbox("Ativo", value=True)

        submitted = st.form_submit_button("Inserir atleta", type="primary")

    if submitted:
        nome = _clean_text_value(nome)
        sobrenome = _clean_text_value(sobrenome)
        genero = _genero_label_to_code(genero_label)
        atleta_id = _generate_internal_atleta_id(athletes_df)

        if not nome:
            st.error("O campo Nome e obrigatorio.")
        elif not sobrenome:
            st.error("O campo Sobrenome e obrigatorio.")
        elif not genero:
            st.error("O campo Genero e obrigatorio.")
        elif foto is None:
            st.error("A foto do atleta e obrigatoria.")
        else:
            try:
                extension = foto.name.rsplit(".", 1)[-1].lower() if "." in foto.name else "jpg"
                object_path = f"{atleta_id}/{atleta_id}.{extension}"
                foto_url = upload_image_to_storage(
                    ATHLETE_PHOTO_BUCKET,
                    object_path,
                    foto.getvalue(),
                    foto.type,
                )
                new_row = pd.DataFrame(
                    [{
                        "atleta_id": atleta_id,
                        "nome": _compose_full_name(nome, sobrenome),
                        "foto_url": foto_url,
                        "data_nascimento": data_nascimento,
                        "posicao": posicao,
                        "genero": genero,
                        "ativo": ativo,
                    }]
                )
                final_df = pd.concat([athletes_df, new_row], ignore_index=True)
                _save_athletes(final_df)
            except RuntimeError as exc:
                st.error(str(exc))
            else:
                st.session_state.pop(ATHLETE_PHOTO_UPLOADER_KEY, None)
                st.success("Novo atleta criado com sucesso.")
                st.rerun()
