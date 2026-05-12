from __future__ import annotations

import re
import unicodedata
from datetime import date

import pandas as pd
import streamlit as st

from fpf_modules.futsal_parquet_store import generate_athlete_id, read_athletes, save_photo, upsert_athlete


ATHLETE_BIRTHDATE_MIN = date(1980, 1, 1)
ATHLETE_BIRTHDATE_MAX = date.today()
GENDER_OPTIONS = ["Masculino", "Feminino"]
POSITION_OPTIONS = ["", "GR", "Fixo", "Ala", "Pivot", "Universal"]
SELECTION_OPTIONS = ["", "S12", "S13", "S14", "S15", "S16", "S17"]


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _clean_date(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _calculate_age(birth_date, reference_date: date | None = None) -> int | None:
    birth = _clean_date(birth_date)
    if not birth:
        return None
    today = reference_date or date.today()
    return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))


def _derive_escalao_from_age(age: int | None) -> str:
    if age is None or age < 0:
        return ""
    if age <= 11:
        return "Benjamins"
    if age <= 13:
        return "Infantis"
    if age <= 15:
        return "Iniciados"
    if age <= 17:
        return "Juvenis"
    if age <= 19:
        return "Juniores"
    if age <= 23:
        return "Sub-23"
    return "Seniores"


def _extract_convocatoria_athletes(uploaded_file) -> list[str]:
    try:
        from pypdf import PdfReader
    except Exception as exc:
        raise RuntimeError(
            "A opcao de importar convocatoria PDF precisa da biblioteca 'pypdf' instalada no ambiente da app."
        ) from exc

    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    athlete_names: list[str] = []
    player_line_pattern = re.compile(r"^(?P<body>.+?)\s+(?P<count>[1-5])$")
    for line in lines:
        match = player_line_pattern.match(line)
        if not match:
            continue

        body = match.group("body").strip()
        count = int(match.group("count"))
        if not any(ch.islower() for ch in body):
            continue

        names_part = body.replace("(Gr)", "").strip()
        names_part = unicodedata.normalize("NFC", names_part)
        names_part = re.sub(
            r"(?<=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ])(?=[A-ZÁÀÂÃÉÊÍÓÔÕÚÇ][a-záàâãéêíóôõúç])",
            " ",
            names_part,
        )
        words = re.findall(r"[^\W\d_]+", names_part, flags=re.UNICODE)
        tokens = [word for word in words if word[:1].isupper() and not word.isupper()]
        if len(tokens) < 2:
            continue

        if count == 1:
            athlete_names.append(" ".join(tokens))
            continue

        if len(tokens) >= count * 2 and len(tokens) % count == 0:
            chunk_size = len(tokens) // count
            for start in range(0, len(tokens), chunk_size):
                athlete_names.append(" ".join(tokens[start:start + chunk_size]))
            continue

        if count == 2 and len(tokens) >= 4:
            athlete_names.append(" ".join(tokens[:2]))
            athlete_names.append(" ".join(tokens[2:4]))

    cleaned = []
    seen = set()
    for name in athlete_names:
        cleaned_name = _clean_text_value(name)
        if not cleaned_name:
            continue
        key = cleaned_name.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(cleaned_name)
    return cleaned


def _create_athletes_from_names(names: list[str], athletes_df: pd.DataFrame) -> dict:
    existing_names = {
        _clean_text_value(row.get("nome")).lower()
        for _, row in athletes_df.iterrows()
        if _clean_text_value(row.get("nome"))
    }
    created = 0
    skipped = 0
    current_df = read_athletes()
    for name in names:
        normalized = _clean_text_value(name)
        if not normalized:
            continue
        key = normalized.lower()
        if key in existing_names:
            skipped += 1
            continue
        athlete_id = generate_athlete_id(current_df)
        upsert_athlete(
            {
                "atleta_id": athlete_id,
                "nome": normalized,
                "data_nascimento": None,
                "genero": "",
                "selecao": "",
                "posicao": "",
                "foto_path": "",
                "ativo": True,
            }
        )
        current_df = read_athletes()
        existing_names.add(key)
        created += 1
    return {"created": created, "skipped": skipped}


def _render_create_athlete_form(athletes_df: pd.DataFrame) -> None:
    st.title("Criar Atleta")
    with st.form("futsal_create_athlete_form", clear_on_submit=True):
        athlete_id = generate_athlete_id(athletes_df)
        st.caption(f"ID atribuido: {athlete_id}")
        top_left, top_right = st.columns([0.8, 2.2], gap="large")
        with top_left:
            st.markdown("**Foto do atleta**")
            st.caption("Captura tipo passe.")
            foto = st.camera_input("Tirar foto", key="futsal_master_camera", label_visibility="collapsed")
        with top_right:
            line1_col1, line1_col2 = st.columns([1.1, 0.9])
            nome = line1_col1.text_input("Nome")
            line1_col2.caption("Introduz nome completo: primeiro e ultimo nome.")

            line2_col1, line2_col2, line2_col3 = st.columns(3)
            data_nascimento = line2_col1.date_input(
                "Data de nascimento",
                value=None,
                min_value=ATHLETE_BIRTHDATE_MIN,
                max_value=ATHLETE_BIRTHDATE_MAX,
                format="DD/MM/YYYY",
            )
            idade_preview = _calculate_age(data_nascimento)
            line2_col2.text_input("Idade", value="" if idade_preview is None else str(idade_preview), disabled=True)
            genero = line2_col3.selectbox("Genero", options=GENDER_OPTIONS)

            line3_col1, line3_col2, line3_col3 = st.columns(3)
            escalao_preview = _derive_escalao_from_age(idade_preview)
            line3_col1.text_input("Escalao", value=escalao_preview, disabled=True)
            selecao = line3_col2.selectbox("Selecao", options=SELECTION_OPTIONS)
            posicao = line3_col3.selectbox("Posicao", options=POSITION_OPTIONS)
        ativo = st.checkbox("Ativo", value=True)
        submitted = st.form_submit_button("Criar atleta", type="primary")

    if submitted:
        if not _clean_text_value(nome):
            st.error("O campo Nome e obrigatorio.")
            return
        photo_path = ""
        if foto is not None:
            photo_path = save_photo(athlete_id, foto)
        upsert_athlete(
            {
                "atleta_id": athlete_id,
                "nome": _clean_text_value(nome),
                "data_nascimento": _clean_date(data_nascimento),
                "genero": _clean_text_value(genero),
                "selecao": _clean_text_value(selecao),
                "posicao": _clean_text_value(posicao),
                "foto_path": photo_path,
                "ativo": bool(ativo),
            }
        )
        st.success("Atleta criado com sucesso.")
        st.rerun()

    st.divider()
    st.markdown("**Criacao em lote por convocatoria PDF**")
    st.caption("Cria atletas apenas com o campo Nome a partir da convocatoria. O resto da ficha pode ser preenchido depois.")
    uploaded_convocatoria = st.file_uploader(
        "Carregar convocatoria PDF",
        type=["pdf"],
        key="futsal_convocatoria_pdf",
    )
    if uploaded_convocatoria is not None and st.button("Criar lote de atletas", key="create_batch_from_pdf"):
        try:
            athlete_names = _extract_convocatoria_athletes(uploaded_convocatoria)
            if not athlete_names:
                raise RuntimeError("Nao foi possivel identificar atletas na convocatoria.")
            result = _create_athletes_from_names(athlete_names, athletes_df)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Lote criado. Novos atletas: {result['created']} | Ja existentes: {result['skipped']}")
            st.caption("Atletas identificados: " + ", ".join(athlete_names))
            st.rerun()


_render_create_athlete_form(read_athletes())
