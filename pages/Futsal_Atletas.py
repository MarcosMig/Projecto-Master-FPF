from __future__ import annotations

import re
import unicodedata
from datetime import date
from io import StringIO

import pandas as pd
import streamlit as st

from fpf_modules.futsal_parquet_store import (
    ATHLETE_COLUMNS,
    PHYSICAL_COLUMNS,
    TECHNICAL_COLUMNS,
    append_records,
    delete_athlete,
    delete_photo,
    delete_record,
    generate_athlete_id,
    latest_records_by_athlete,
    read_athletes,
    read_physical_records,
    read_technical_records,
    save_photo,
    upsert_athlete,
)


ATHLETE_BIRTHDATE_MIN = date(1980, 1, 1)
ATHLETE_BIRTHDATE_MAX = date.today()
GENDER_OPTIONS = ["Masculino", "Feminino"]
POSITION_OPTIONS = ["", "GR", "Fixo", "Ala", "Pivot", "Universal"]
SELECTION_OPTIONS = ["", "Sub-15", "Sub-17", "Sub-19", "Sub-21", "AA"]

MYJUMPLAB_IMPORT_COLUMNS = [
    "Date",
    "Team",
    "Name",
    "Body weight(kg)",
    "Push-off distance (hp0, in m)",
    "RSI 10-5",
    "CMJ (cm)",
    "Jump 1 (cm)",
    "Jump 2 (cm)",
    "Jump 3 (cm)",
    "Jump 4 (cm)",
    "Jump 5 (cm)",
    "Jump 6 (cm)",
    "Jump 7 (cm)",
    "Jump 8 (cm)",
    "Jump 9 (cm)",
    "Jump 10 (cm)",
    "Contact time 1 (ms)",
    "Contact time 2 (ms)",
    "Contact time 3 (ms)",
    "Contact time 4 (ms)",
    "Contact time 5 (ms)",
    "Contact time 6 (ms)",
    "Contact time 7 (ms)",
    "Contact time 8 (ms)",
    "Contact time 9 (ms)",
    "Contact time 10 (ms)",
]

PHYSICAL_UPLOAD_COLUMNS = [
    "atleta_id",
    "data_avaliacao",
    "peso_kg",
    "altura_cm",
    "envergadura_cm",
    "comprimento_perna_cm",
    "sprint_10m_s",
    "sprint_20m_s",
    "teste_505_esq_s",
    "teste_505_dir_s",
    "sj_altura_cm",
    "cmj_altura_cm",
    "dj_caixa_m",
    "dj_altura_cm",
    "dj_rsi",
    "dj_rsi_mod_mps",
    "dj_contacto_ms",
    "jump_1_cm",
    "jump_2_cm",
    "jump_3_cm",
    "jump_4_cm",
    "jump_5_cm",
    "jump_6_cm",
    "jump_7_cm",
    "jump_8_cm",
    "jump_9_cm",
    "jump_10_cm",
    "contact_1_ms",
    "contact_2_ms",
    "contact_3_ms",
    "contact_4_ms",
    "contact_5_ms",
    "contact_6_ms",
    "contact_7_ms",
    "contact_8_ms",
    "contact_9_ms",
    "contact_10_ms",
    "j10_rsi_10_5",
    "j10_cmj_cm",
    "j10_media_saltos_cm",
    "j10_maximo_cm",
    "j10_minimo_cm",
    "indice_fadiga_10j_pct",
]

TECHNICAL_UPLOAD_COLUMNS = [
    "atleta_id",
    "data_avaliacao",
    "passe_score",
    "remate_score",
    "drible_score",
    "controlo_bola_score",
    "decisao_score",
    "concentracao_score",
    "lideranca_score",
    "resiliencia_score",
    "competitividade_score",
    "observacoes",
]

REPORT_PHYSICAL_COLUMN_MAP = {
    "Peso corporal (kg)": "peso_kg",
    "Altura (cm)": "altura_cm",
    "Envergadura (cm)": "envergadura_cm",
    "Comprimento Perna (cm)": "comprimento_perna_cm",
    "Sprint 10m(s)": "sprint_10m_s",
    "Sprint 20m (s)": "sprint_20m_s",
    "Teste 5-0-5 (esq)": "teste_505_esq_s",
    "Teste 5-0-5 (dir)": "teste_505_dir_s",
    "SJ altura (cm)": "sj_altura_cm",
    "CMJ altura (cm)": "cmj_altura_cm",
    "DJ caixa (m)": "dj_caixa_m",
    "DJ altura (cm)": "dj_altura_cm",
    "DJ RSI": "dj_rsi",
    "DJ RSI mod (m/s)": "dj_rsi_mod_mps",
    "DJ contacto (ms)": "dj_contacto_ms",
    "Jump 1 (cm)": "jump_1_cm",
    "Jump 2 (cm)": "jump_2_cm",
    "Jump 3 (cm)": "jump_3_cm",
    "Jump 4 (cm)": "jump_4_cm",
    "Jump 5 (cm)": "jump_5_cm",
    "Jump 6 (cm)": "jump_6_cm",
    "Jump 7 (cm)": "jump_7_cm",
    "Jump 8 (cm)": "jump_8_cm",
    "Jump 9 (cm)": "jump_9_cm",
    "Jump 10 (cm)": "jump_10_cm",
    "Contact 1 (ms)": "contact_1_ms",
    "Contact 2 (ms)": "contact_2_ms",
    "Contact 3 (ms)": "contact_3_ms",
    "Contact 4 (ms)": "contact_4_ms",
    "Contact 5 (ms)": "contact_5_ms",
    "Contact 6 (ms)": "contact_6_ms",
    "Contact 7 (ms)": "contact_7_ms",
    "Contact 8 (ms)": "contact_8_ms",
    "Contact 9 (ms)": "contact_9_ms",
    "Contact 10 (ms)": "contact_10_ms",
    "10J RSI 10-5": "j10_rsi_10_5",
    "10J CMJ (cm)": "j10_cmj_cm",
    "10J média saltos (cm)": "j10_media_saltos_cm",
    "10J máximo (cm)": "j10_maximo_cm",
    "10J mínimo (cm)": "j10_minimo_cm",
    "Índice fadiga 10J (%)": "indice_fadiga_10j_pct",
}


def _clean_text_value(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"", "None", "nan", "NaT", "<NA>"} else text


def _clean_number(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return None


def _clean_integer(value):
    number = _clean_number(value)
    return None if number is None else int(number)


def _clean_date(value):
    if value in ("", None) or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    return None if pd.isna(parsed) else parsed.date()


def _format_date(value) -> str:
    cleaned = _clean_date(value)
    return cleaned.strftime("%d/%m/%Y") if cleaned else "-"


def _format_metric(value, suffix: str = "", decimals: int = 1) -> str:
    number = _clean_number(value)
    if number is None:
        return "-"
    return f"{number:.{decimals}f}{suffix}"


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


def _render_summary_metrics(athletes_df: pd.DataFrame, physical_df: pd.DataFrame, technical_df: pd.DataFrame) -> None:
    total_atletas = int(len(athletes_df))
    ativos = int(athletes_df["ativo"].fillna(False).sum()) if not athletes_df.empty else 0
    total_fisicas = int(len(physical_df))
    total_tecnicas = int(len(technical_df))
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Atletas", total_atletas)
    col2.metric("Ativos", ativos)
    col3.metric("Registos fisicos", total_fisicas)
    col4.metric("Registos tecnico/psico", total_tecnicas)


def _read_table_upload(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    file_name = str(getattr(uploaded_file, "name", "") or "").lower()
    if file_name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.getvalue()
    for sep in [";", ",", "\t"]:
        try:
            return pd.read_csv(StringIO(raw_bytes.decode("utf-8-sig")), sep=sep)
        except Exception:
            continue
    raise RuntimeError("Nao foi possivel ler o ficheiro.")


def _build_template_csv(columns: list[str]) -> bytes:
    return pd.DataFrame(columns=columns).to_csv(index=False).encode("utf-8-sig")


def _validate_athlete_ids(df_upload: pd.DataFrame, athletes_df: pd.DataFrame) -> None:
    athlete_ids = set(athletes_df["atleta_id"].astype(str).str.strip().tolist()) if not athletes_df.empty else set()
    upload_ids = set(df_upload["atleta_id"].fillna("").astype(str).str.strip().tolist())
    missing = sorted([athlete_id for athlete_id in upload_ids if athlete_id and athlete_id not in athlete_ids])
    if missing:
        raise RuntimeError(f"Os seguintes atleta_id nao existem na ficha mestre: {', '.join(missing[:15])}")


def _prepare_uploaded_records(df_upload: pd.DataFrame, allowed_columns: list[str]) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")
    if "atleta_id" not in df_upload.columns:
        raise RuntimeError("O ficheiro tem de incluir a coluna 'atleta_id'.")

    work_df = df_upload.copy()
    for col in allowed_columns:
        if col not in work_df.columns:
            work_df[col] = pd.NA

    work_df["atleta_id"] = work_df["atleta_id"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nao existem atleta_id validos no ficheiro.")

    work_df["data_avaliacao"] = pd.to_datetime(work_df.get("data_avaliacao"), errors="coerce", dayfirst=True).dt.date
    for col in allowed_columns:
        if col in {"atleta_id", "data_avaliacao"}:
            continue
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    return work_df[allowed_columns].copy()


def _prepare_report_physical_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")

    work_df = df_upload.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]

    if "id" not in work_df.columns and "atleta_id" not in work_df.columns and "Nome" not in work_df.columns:
        raise RuntimeError("A folha tem de incluir a coluna 'id'/'atleta_id' ou 'Nome'.")

    if "atleta_id" not in work_df.columns:
        if "id" in work_df.columns:
            work_df["atleta_id"] = work_df["id"]
        else:
            work_df["atleta_id"] = pd.NA

    work_df["atleta_id"] = work_df["atleta_id"].fillna("").astype(str).str.strip()

    if work_df["atleta_id"].eq("").any() and "Nome" in work_df.columns:
        athlete_name_map = {
            _clean_text_value(row.get("nome")).lower(): _clean_text_value(row.get("atleta_id"))
            for _, row in athletes_df.iterrows()
            if _clean_text_value(row.get("nome"))
        }
        missing_mask = work_df["atleta_id"].eq("")
        work_df.loc[missing_mask, "atleta_id"] = (
            work_df.loc[missing_mask, "Nome"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map(athlete_name_map)
            .fillna("")
        )

    rename_map = {source: target for source, target in REPORT_PHYSICAL_COLUMN_MAP.items() if source in work_df.columns}
    work_df = work_df.rename(columns=rename_map)
    work_df["data_avaliacao"] = _clean_date(data_avaliacao)

    for col in PHYSICAL_UPLOAD_COLUMNS:
        if col not in work_df.columns:
            work_df[col] = pd.NA

    work_df["atleta_id"] = work_df["atleta_id"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nenhuma linha valida foi encontrada. Confirma o ID ou o Nome dos atletas.")

    for col in PHYSICAL_UPLOAD_COLUMNS:
        if col in {"atleta_id", "data_avaliacao"}:
            continue
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    _validate_athlete_ids(work_df[["atleta_id"]].copy(), athletes_df)
    return work_df[["atleta_id", "data_avaliacao"] + PHYSICAL_COLUMNS].copy()


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


def _myjumplab_defaults(uploaded_file) -> tuple[dict, str]:
    uploaded_file.seek(0)
    df = pd.read_csv(uploaded_file, sep=";", decimal=",", encoding="utf-8-sig")
    if df.empty:
        raise RuntimeError("O ficheiro MyJumpLab nao contem registos.")
    row = df.iloc[0].to_dict()
    jump_values = [_clean_number(row.get(f"Jump {idx} (cm)")) for idx in range(1, 11)]
    valid_jumps = [value for value in jump_values if value is not None]
    max_jump = max(valid_jumps) if valid_jumps else None
    min_jump = min(valid_jumps) if valid_jumps else None
    fadiga = None
    if max_jump and min_jump is not None and max_jump > 0:
        fadiga = ((max_jump - min_jump) / max_jump) * 100
    imported_name = _clean_text_value(row.get("Name"))
    defaults = {
        "data_avaliacao": _clean_date(row.get("Date")),
        "peso_kg": _clean_number(row.get("Body weight(kg)")),
        "jump_1_cm": _clean_number(row.get("Jump 1 (cm)")),
        "jump_2_cm": _clean_number(row.get("Jump 2 (cm)")),
        "jump_3_cm": _clean_number(row.get("Jump 3 (cm)")),
        "jump_4_cm": _clean_number(row.get("Jump 4 (cm)")),
        "jump_5_cm": _clean_number(row.get("Jump 5 (cm)")),
        "jump_6_cm": _clean_number(row.get("Jump 6 (cm)")),
        "jump_7_cm": _clean_number(row.get("Jump 7 (cm)")),
        "jump_8_cm": _clean_number(row.get("Jump 8 (cm)")),
        "jump_9_cm": _clean_number(row.get("Jump 9 (cm)")),
        "jump_10_cm": _clean_number(row.get("Jump 10 (cm)")),
        "contact_1_ms": _clean_number(row.get("Contact time 1 (ms)")),
        "contact_2_ms": _clean_number(row.get("Contact time 2 (ms)")),
        "contact_3_ms": _clean_number(row.get("Contact time 3 (ms)")),
        "contact_4_ms": _clean_number(row.get("Contact time 4 (ms)")),
        "contact_5_ms": _clean_number(row.get("Contact time 5 (ms)")),
        "contact_6_ms": _clean_number(row.get("Contact time 6 (ms)")),
        "contact_7_ms": _clean_number(row.get("Contact time 7 (ms)")),
        "contact_8_ms": _clean_number(row.get("Contact time 8 (ms)")),
        "contact_9_ms": _clean_number(row.get("Contact time 9 (ms)")),
        "contact_10_ms": _clean_number(row.get("Contact time 10 (ms)")),
        "j10_rsi_10_5": _clean_number(row.get("RSI 10-5")),
        "j10_cmj_cm": _clean_number(row.get("CMJ (cm)")),
        "j10_media_saltos_cm": sum(valid_jumps) / len(valid_jumps) if valid_jumps else None,
        "j10_maximo_cm": max_jump,
        "j10_minimo_cm": min_jump,
        "indice_fadiga_10j_pct": fadiga,
    }
    return defaults, imported_name


def _athletes_with_latest(athletes_df: pd.DataFrame, physical_df: pd.DataFrame, technical_df: pd.DataFrame) -> pd.DataFrame:
    work_df = athletes_df.copy()
    if work_df.empty:
        return work_df
    work_df["ativo"] = work_df["ativo"].fillna(False).astype(bool)
    work_df["idade"] = work_df["data_nascimento"].map(_calculate_age)
    work_df["escalao"] = work_df["idade"].map(_derive_escalao_from_age)

    latest_physical = latest_records_by_athlete(physical_df)
    latest_technical = latest_records_by_athlete(technical_df)
    if not latest_physical.empty:
        latest_physical = latest_physical.add_prefix("fis_").rename(columns={"fis_atleta_id": "atleta_id"})
        work_df = work_df.merge(latest_physical, on="atleta_id", how="left")
    if not latest_technical.empty:
        latest_technical = latest_technical.add_prefix("tec_").rename(columns={"tec_atleta_id": "atleta_id"})
        work_df = work_df.merge(latest_technical, on="atleta_id", how="left")
    return work_df


def _render_athlete_registry(athletes_df: pd.DataFrame, physical_df: pd.DataFrame, technical_df: pd.DataFrame) -> None:
    st.subheader("Atletas registados")
    registry_df = _athletes_with_latest(athletes_df, physical_df, technical_df)
    if registry_df.empty:
        st.info("Ainda nao existem atletas registados.")
        return

    filter_col1, filter_col2, filter_col3 = st.columns(3)
    search_term = filter_col1.text_input("Pesquisar atleta", key="futsal_registry_search")
    state_filter = filter_col2.selectbox("Estado", options=["Todos", "Ativos", "Inativos"], key="futsal_registry_state")
    escalao_options = ["Todos"] + sorted([value for value in registry_df["escalao"].dropna().astype(str).unique().tolist() if value])
    escalao_filter = filter_col3.selectbox("Escalao", options=escalao_options, key="futsal_registry_escalao")

    view_df = registry_df.copy()
    if search_term:
        query = search_term.strip().lower()
        view_df = view_df[
            view_df["nome"].astype(str).str.lower().str.contains(query, na=False)
            | view_df["atleta_id"].astype(str).str.lower().str.contains(query, na=False)
        ].copy()
    if state_filter == "Ativos":
        view_df = view_df[view_df["ativo"]].copy()
    elif state_filter == "Inativos":
        view_df = view_df[~view_df["ativo"]].copy()
    if escalao_filter != "Todos":
        view_df = view_df[view_df["escalao"].astype(str) == escalao_filter].copy()

    if view_df.empty:
        st.info("Sem atletas para este filtro.")
        return

    for _, row in view_df.sort_values(["ativo", "nome"], ascending=[False, True], na_position="last").iterrows():
        label = _clean_text_value(row.get("nome")) or _clean_text_value(row.get("atleta_id"))
        with st.expander(label, expanded=False):
            head_col1, head_col2 = st.columns([0.8, 2.2], gap="large")
            with head_col1:
                photo_path = _clean_text_value(row.get("foto_path"))
                if photo_path:
                    st.image(photo_path, width=140)
                else:
                    st.caption("Sem foto registada.")
            with head_col2:
                st.markdown(f"**ID:** {_clean_text_value(row.get('atleta_id'))}")
                st.markdown(f"**Nome:** {_clean_text_value(row.get('nome')) or '-'}")
                st.markdown(f"**Data nascimento:** {_format_date(row.get('data_nascimento'))}")
                st.markdown(f"**Idade:** {_calculate_age(row.get('data_nascimento')) or '-'}")
                st.markdown(f"**Genero:** {_clean_text_value(row.get('genero')) or '-'}")
                st.markdown(f"**Escalao:** {_clean_text_value(row.get('escalao')) or '-'}")
                st.markdown(f"**Selecao:** {_clean_text_value(row.get('selecao')) or '-'}")
                st.markdown(f"**Posicao:** {_clean_text_value(row.get('posicao')) or '-'}")

            st.markdown("**Ultima ficha antropometrica/fisica**")
            st.caption(
                f"Data: {_format_date(row.get('fis_data_avaliacao'))} | Peso: {_format_metric(row.get('fis_peso_kg'), ' kg')} | "
                f"Sprint 10m: {_format_metric(row.get('fis_sprint_10m_s'), ' s', 2)} | CMJ: {_format_metric(row.get('fis_cmj_altura_cm'), ' cm')}"
            )
            st.markdown("**Ultima ficha tecnica/psicologica**")
            st.caption(
                f"Data: {_format_date(row.get('tec_data_avaliacao'))} | Passe: {_format_metric(row.get('tec_passe_score'), '/10')} | "
                f"Decisao: {_format_metric(row.get('tec_decisao_score'), '/10')} | Resiliencia: {_format_metric(row.get('tec_resiliencia_score'), '/10')}"
            )

            action_col1, action_col2 = st.columns(2)
            if action_col1.button("Eliminar atleta e historico", key=f"delete_athlete_{row['atleta_id']}"):
                delete_photo(_clean_text_value(row.get("foto_path")))
                delete_athlete(_clean_text_value(row.get("atleta_id")))
                st.success("Atleta e historico eliminados com sucesso.")
                st.rerun()
            if action_col2.button("Inativar/Ativar", key=f"toggle_athlete_{row['atleta_id']}"):
                updated = athletes_df.copy()
                updated.loc[updated["atleta_id"].astype(str) == str(row["atleta_id"]), "ativo"] = not bool(row.get("ativo"))
                for _, athlete_row in updated.iterrows():
                    upsert_athlete(athlete_row.to_dict())
                st.success("Estado do atleta atualizado.")
                st.rerun()


def _render_create_athlete_form(athletes_df: pd.DataFrame) -> None:
    st.subheader("Criar atleta")
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
            data_nascimento = line2_col1.date_input("Data de nascimento", value=None, min_value=ATHLETE_BIRTHDATE_MIN, max_value=ATHLETE_BIRTHDATE_MAX, format="DD/MM/YYYY")
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
            st.success(
                f"Lote criado. Novos atletas: {result['created']} | Ja existentes: {result['skipped']}"
            )
            st.caption("Atletas identificados: " + ", ".join(athlete_names))
            st.rerun()


def _render_physical_tab(athletes_df: pd.DataFrame, physical_df: pd.DataFrame) -> None:
    st.subheader("Ficha Antropometrica e Testes Fisicos")
    athlete_options = athletes_df["atleta_id"].astype(str).tolist() if not athletes_df.empty else []
    if not athlete_options:
        st.info("Cria primeiro atletas na ficha mestre.")
        return

    st.markdown("**Insercao individual**")
    myjump_defaults = {}
    imported_name = ""
    uploaded_myjump = st.file_uploader("Importar CSV individual MyJumpLab", type=["csv"], key="physical_myjump_upload")
    if uploaded_myjump is not None:
        try:
            myjump_defaults, imported_name = _myjumplab_defaults(uploaded_myjump)
            msg = f"MyJumpLab lido com sucesso. Atleta no ficheiro: {imported_name or '-'}"
            st.caption(msg)
        except Exception as exc:
            st.warning(f"Nao foi possivel ler o CSV MyJumpLab: {exc}")
            myjump_defaults = {}

    with st.form("physical_single_record_form", clear_on_submit=True):
        col1, col2, col3 = st.columns(3)
        atleta_id = col1.selectbox("Atleta ID", options=athlete_options)
        data_avaliacao = col2.date_input("Data avaliacao", value=myjump_defaults.get("data_avaliacao"))
        col3.markdown("**Conferencia MyJumpLab**")
        col3.caption(imported_name or "-")

        st.markdown("**Antropometria**")
        a1, a2, a3, a4 = st.columns(4)
        peso_kg = a1.number_input("Peso corporal (kg)", min_value=0.0, value=float(myjump_defaults.get("peso_kg", 0.0) or 0.0), step=0.1)
        altura_cm = a2.number_input("Altura (cm)", min_value=0.0, value=0.0, step=0.1)
        envergadura_cm = a3.number_input("Envergadura (cm)", min_value=0.0, value=0.0, step=0.1)
        comprimento_perna_cm = a4.number_input("Comprimento Perna (cm)", min_value=0.0, value=0.0, step=0.1)

        st.markdown("**Velocidade e Agilidade**")
        v1, v2, v3, v4, v5 = st.columns(5)
        sprint_10m_s = v1.number_input("Sprint 10m (s)", min_value=0.0, value=0.0, step=0.01)
        sprint_20m_s = v2.number_input("Sprint 20m (s)", min_value=0.0, value=0.0, step=0.01)
        teste_505_esq_s = v3.number_input("Teste 5-0-5 (esq)", min_value=0.0, value=0.0, step=0.01)
        teste_505_dir_s = v4.number_input("Teste 5-0-5 (dir)", min_value=0.0, value=0.0, step=0.01)
        v5.empty()

        st.markdown("**Saltos Simples**")
        j1, j2, j3, j4, j5 = st.columns(5)
        sj_altura_cm = j1.number_input("SJ altura (cm)", min_value=0.0, value=0.0, step=0.1)
        cmj_altura_cm = j2.number_input("CMJ altura (cm)", min_value=0.0, value=0.0, step=0.1)
        dj_caixa_m = j3.number_input("DJ caixa (m)", min_value=0.0, value=0.0, step=0.01)
        dj_altura_cm = j4.number_input("DJ altura (cm)", min_value=0.0, value=0.0, step=0.1)
        dj_rsi = j5.number_input("DJ RSI", min_value=0.0, value=0.0, step=0.01)

        d1, d2 = st.columns(2)
        dj_rsi_mod_mps = d1.number_input("DJ RSI mod (m/s)", min_value=0.0, value=0.0, step=0.01)
        dj_contacto_ms = d2.number_input("DJ contacto (ms)", min_value=0.0, value=0.0, step=0.1)

        st.markdown("**10 Jumps - Saltos**")
        jump_cols_1 = st.columns(5)
        jump_1_cm = jump_cols_1[0].number_input("Jump 1 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_1_cm", 0.0) or 0.0), step=0.1)
        jump_2_cm = jump_cols_1[1].number_input("Jump 2 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_2_cm", 0.0) or 0.0), step=0.1)
        jump_3_cm = jump_cols_1[2].number_input("Jump 3 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_3_cm", 0.0) or 0.0), step=0.1)
        jump_4_cm = jump_cols_1[3].number_input("Jump 4 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_4_cm", 0.0) or 0.0), step=0.1)
        jump_5_cm = jump_cols_1[4].number_input("Jump 5 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_5_cm", 0.0) or 0.0), step=0.1)

        jump_cols_2 = st.columns(5)
        jump_6_cm = jump_cols_2[0].number_input("Jump 6 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_6_cm", 0.0) or 0.0), step=0.1)
        jump_7_cm = jump_cols_2[1].number_input("Jump 7 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_7_cm", 0.0) or 0.0), step=0.1)
        jump_8_cm = jump_cols_2[2].number_input("Jump 8 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_8_cm", 0.0) or 0.0), step=0.1)
        jump_9_cm = jump_cols_2[3].number_input("Jump 9 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_9_cm", 0.0) or 0.0), step=0.1)
        jump_10_cm = jump_cols_2[4].number_input("Jump 10 (cm)", min_value=0.0, value=float(myjump_defaults.get("jump_10_cm", 0.0) or 0.0), step=0.1)

        st.markdown("**10 Jumps - Contactos**")
        contact_cols_1 = st.columns(5)
        contact_1_ms = contact_cols_1[0].number_input("Contact 1 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_1_ms", 0.0) or 0.0), step=0.1)
        contact_2_ms = contact_cols_1[1].number_input("Contact 2 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_2_ms", 0.0) or 0.0), step=0.1)
        contact_3_ms = contact_cols_1[2].number_input("Contact 3 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_3_ms", 0.0) or 0.0), step=0.1)
        contact_4_ms = contact_cols_1[3].number_input("Contact 4 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_4_ms", 0.0) or 0.0), step=0.1)
        contact_5_ms = contact_cols_1[4].number_input("Contact 5 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_5_ms", 0.0) or 0.0), step=0.1)

        contact_cols_2 = st.columns(5)
        contact_6_ms = contact_cols_2[0].number_input("Contact 6 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_6_ms", 0.0) or 0.0), step=0.1)
        contact_7_ms = contact_cols_2[1].number_input("Contact 7 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_7_ms", 0.0) or 0.0), step=0.1)
        contact_8_ms = contact_cols_2[2].number_input("Contact 8 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_8_ms", 0.0) or 0.0), step=0.1)
        contact_9_ms = contact_cols_2[3].number_input("Contact 9 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_9_ms", 0.0) or 0.0), step=0.1)
        contact_10_ms = contact_cols_2[4].number_input("Contact 10 (ms)", min_value=0.0, value=float(myjump_defaults.get("contact_10_ms", 0.0) or 0.0), step=0.1)

        st.markdown("**10 Jumps - Resumo**")
        t1, t2, t3, t4, t5 = st.columns(5)
        j10_rsi_10_5 = t1.number_input("10J RSI 10-5", min_value=0.0, value=float(myjump_defaults.get("j10_rsi_10_5", 0.0) or 0.0), step=0.01)
        j10_cmj_cm = t2.number_input("10J CMJ (cm)", min_value=0.0, value=float(myjump_defaults.get("j10_cmj_cm", 0.0) or 0.0), step=0.1)
        j10_media_saltos_cm = t3.number_input("10J media saltos (cm)", min_value=0.0, value=float(myjump_defaults.get("j10_media_saltos_cm", 0.0) or 0.0), step=0.1)
        j10_maximo_cm = t4.number_input("10J maximo (cm)", min_value=0.0, value=float(myjump_defaults.get("j10_maximo_cm", 0.0) or 0.0), step=0.1)
        j10_minimo_cm = t5.number_input("10J minimo (cm)", min_value=0.0, value=float(myjump_defaults.get("j10_minimo_cm", 0.0) or 0.0), step=0.1)

        x1 = st.columns(1)[0]
        indice_fadiga_10j_pct = x1.number_input("Indice fadiga 10J (%)", value=float(myjump_defaults.get("indice_fadiga_10j_pct", 0.0) or 0.0), step=0.1)
        submitted = st.form_submit_button("Guardar ficha fisica", type="primary")

    if submitted:
        new_record = pd.DataFrame([{
            "atleta_id": atleta_id,
            "data_avaliacao": _clean_date(data_avaliacao),
            "peso_kg": peso_kg if peso_kg > 0 else None,
            "altura_cm": altura_cm if altura_cm > 0 else None,
            "envergadura_cm": envergadura_cm if envergadura_cm > 0 else None,
            "comprimento_perna_cm": comprimento_perna_cm if comprimento_perna_cm > 0 else None,
            "sprint_10m_s": sprint_10m_s if sprint_10m_s > 0 else None,
            "sprint_20m_s": sprint_20m_s if sprint_20m_s > 0 else None,
            "teste_505_esq_s": teste_505_esq_s if teste_505_esq_s > 0 else None,
            "teste_505_dir_s": teste_505_dir_s if teste_505_dir_s > 0 else None,
            "sj_altura_cm": sj_altura_cm if sj_altura_cm > 0 else None,
            "cmj_altura_cm": cmj_altura_cm if cmj_altura_cm > 0 else None,
            "dj_caixa_m": dj_caixa_m if dj_caixa_m > 0 else None,
            "dj_altura_cm": dj_altura_cm if dj_altura_cm > 0 else None,
            "dj_rsi": dj_rsi if dj_rsi > 0 else None,
            "dj_rsi_mod_mps": dj_rsi_mod_mps if dj_rsi_mod_mps > 0 else None,
            "dj_contacto_ms": dj_contacto_ms if dj_contacto_ms > 0 else None,
            "jump_1_cm": jump_1_cm if jump_1_cm > 0 else None,
            "jump_2_cm": jump_2_cm if jump_2_cm > 0 else None,
            "jump_3_cm": jump_3_cm if jump_3_cm > 0 else None,
            "jump_4_cm": jump_4_cm if jump_4_cm > 0 else None,
            "jump_5_cm": jump_5_cm if jump_5_cm > 0 else None,
            "jump_6_cm": jump_6_cm if jump_6_cm > 0 else None,
            "jump_7_cm": jump_7_cm if jump_7_cm > 0 else None,
            "jump_8_cm": jump_8_cm if jump_8_cm > 0 else None,
            "jump_9_cm": jump_9_cm if jump_9_cm > 0 else None,
            "jump_10_cm": jump_10_cm if jump_10_cm > 0 else None,
            "contact_1_ms": contact_1_ms if contact_1_ms > 0 else None,
            "contact_2_ms": contact_2_ms if contact_2_ms > 0 else None,
            "contact_3_ms": contact_3_ms if contact_3_ms > 0 else None,
            "contact_4_ms": contact_4_ms if contact_4_ms > 0 else None,
            "contact_5_ms": contact_5_ms if contact_5_ms > 0 else None,
            "contact_6_ms": contact_6_ms if contact_6_ms > 0 else None,
            "contact_7_ms": contact_7_ms if contact_7_ms > 0 else None,
            "contact_8_ms": contact_8_ms if contact_8_ms > 0 else None,
            "contact_9_ms": contact_9_ms if contact_9_ms > 0 else None,
            "contact_10_ms": contact_10_ms if contact_10_ms > 0 else None,
            "j10_rsi_10_5": j10_rsi_10_5 if j10_rsi_10_5 > 0 else None,
            "j10_cmj_cm": j10_cmj_cm if j10_cmj_cm > 0 else None,
            "j10_media_saltos_cm": j10_media_saltos_cm if j10_media_saltos_cm > 0 else None,
            "j10_maximo_cm": j10_maximo_cm if j10_maximo_cm > 0 else None,
            "j10_minimo_cm": j10_minimo_cm if j10_minimo_cm > 0 else None,
            "indice_fadiga_10j_pct": indice_fadiga_10j_pct if indice_fadiga_10j_pct != 0 else None,
        }])
        append_records("physical", new_record, source_type="manual")
        st.success("Ficha fisica guardada com sucesso.")
        st.rerun()

    st.divider()
    st.markdown("**Upload em lote**")
    st.download_button("Descarregar template fisico CSV", data=_build_template_csv(PHYSICAL_UPLOAD_COLUMNS), file_name="template_fisico_futsal.csv", mime="text/csv")
    uploaded_physical = st.file_uploader("Carregar ficheiro fisico normalizado", type=["csv"], key="physical_bulk_upload")
    if uploaded_physical is not None and st.button("Importar lote fisico", key="import_physical_batch"):
        try:
            df_upload = _read_table_upload(uploaded_physical)
            df_prepared = _prepare_uploaded_records(df_upload, PHYSICAL_UPLOAD_COLUMNS)
            _validate_athlete_ids(df_prepared, athletes_df)
            result = append_records("physical", df_prepared, source_type="upload", source_file=uploaded_physical.name)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Lote fisico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}")
            st.rerun()

    st.caption("Opcao direta para a folha da equipa em Excel/CSV, com colunas como no relatorio final.")
    report_col1, report_col2 = st.columns([1.4, 1])
    uploaded_physical_report = report_col1.file_uploader(
        "Carregar folha da equipa",
        type=["xlsx", "xls", "csv"],
        key="physical_report_upload",
    )
    report_date = report_col2.date_input(
        "Data da avaliacao do lote",
        value=date.today(),
        format="DD/MM/YYYY",
        key="physical_report_date",
    )
    if uploaded_physical_report is not None and st.button("Importar folha da equipa", key="import_physical_report"):
        try:
            df_upload = _read_table_upload(uploaded_physical_report)
            df_prepared = _prepare_report_physical_records(df_upload, athletes_df, report_date)
            result = append_records("physical", df_prepared, source_type="team_report", source_file=uploaded_physical_report.name)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Folha da equipa importada com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}")
            st.rerun()

    st.divider()
    st.markdown("**Historico fisico**")
    if physical_df.empty:
        st.info("Sem historico fisico.")
    else:
        history_df = physical_df.copy()
        history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce").dt.strftime("%d/%m/%Y")
        history_df["inserted_at"] = pd.to_datetime(history_df["inserted_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(
            history_df[
                [
                    "record_id",
                    "batch_id",
                    "inserted_at",
                    "source_type",
                    "source_file",
                    "atleta_id",
                    "data_avaliacao",
                    "peso_kg",
                    "altura_cm",
                    "sprint_10m_s",
                    "sj_altura_cm",
                    "cmj_altura_cm",
                    "j10_rsi_10_5",
                    "j10_media_saltos_cm",
                    "indice_fadiga_10j_pct",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )
        record_to_delete = st.selectbox("Eliminar registo fisico", options=history_df["record_id"].astype(str).tolist(), key="delete_physical_record")
        if st.button("Eliminar registo fisico selecionado", key="delete_physical_btn"):
            delete_record("physical", record_to_delete)
            st.success("Registo fisico eliminado com sucesso.")
            st.rerun()


def _render_technical_tab(athletes_df: pd.DataFrame, technical_df: pd.DataFrame) -> None:
    st.subheader("Ficha Tecnica e Psicologica")
    athlete_options = athletes_df["atleta_id"].astype(str).tolist() if not athletes_df.empty else []
    if not athlete_options:
        st.info("Cria primeiro atletas na ficha mestre.")
        return

    with st.form("technical_single_record_form", clear_on_submit=True):
        col1, col2 = st.columns(2)
        atleta_id = col1.selectbox("Atleta ID", options=athlete_options, key="technical_atleta_id")
        data_avaliacao = col2.date_input("Data avaliacao", value=None, format="DD/MM/YYYY", key="technical_data_avaliacao")
        c1, c2, c3, c4, c5 = st.columns(5)
        passe_score = c1.number_input("Passe", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        remate_score = c2.number_input("Remate", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        drible_score = c3.number_input("Drible", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        controlo_bola_score = c4.number_input("Controlo bola", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        decisao_score = c5.number_input("Decisao", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        p1, p2, p3, p4 = st.columns(4)
        concentracao_score = p1.number_input("Concentracao", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        lideranca_score = p2.number_input("Lideranca", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        resiliencia_score = p3.number_input("Resiliencia", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        competitividade_score = p4.number_input("Competitividade", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        observacoes = st.text_input("Observacoes")
        submitted = st.form_submit_button("Guardar ficha tecnica/psicologica", type="primary")

    if submitted:
        record_df = pd.DataFrame([{
            "atleta_id": atleta_id,
            "data_avaliacao": _clean_date(data_avaliacao),
            "passe_score": passe_score if passe_score > 0 else None,
            "remate_score": remate_score if remate_score > 0 else None,
            "drible_score": drible_score if drible_score > 0 else None,
            "controlo_bola_score": controlo_bola_score if controlo_bola_score > 0 else None,
            "decisao_score": decisao_score if decisao_score > 0 else None,
            "concentracao_score": concentracao_score if concentracao_score > 0 else None,
            "lideranca_score": lideranca_score if lideranca_score > 0 else None,
            "resiliencia_score": resiliencia_score if resiliencia_score > 0 else None,
            "competitividade_score": competitividade_score if competitividade_score > 0 else None,
            "observacoes": _clean_text_value(observacoes),
        }])
        append_records("technical", record_df, source_type="manual")
        st.success("Ficha tecnica/psicologica guardada com sucesso.")
        st.rerun()

    st.divider()
    st.markdown("**Upload normalizado em lote**")
    st.download_button("Descarregar template tecnico/psicologico CSV", data=_build_template_csv(TECHNICAL_UPLOAD_COLUMNS), file_name="template_tecnico_psicologico_futsal.csv", mime="text/csv")
    uploaded_technical = st.file_uploader("Carregar ficheiro tecnico/psicologico normalizado", type=["csv"], key="technical_bulk_upload")
    if uploaded_technical is not None and st.button("Importar lote tecnico/psicologico", key="import_technical_batch"):
        try:
            df_upload = _read_table_upload(uploaded_technical)
            df_prepared = _prepare_uploaded_records(df_upload, TECHNICAL_UPLOAD_COLUMNS)
            _validate_athlete_ids(df_prepared, athletes_df)
            result = append_records("technical", df_prepared, source_type="upload", source_file=uploaded_technical.name)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Lote tecnico/psicologico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}")
            st.rerun()

    st.divider()
    st.markdown("**Historico tecnico/psicologico**")
    if technical_df.empty:
        st.info("Sem historico tecnico/psicologico.")
    else:
        history_df = technical_df.copy()
        history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce").dt.strftime("%d/%m/%Y")
        history_df["inserted_at"] = pd.to_datetime(history_df["inserted_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        st.dataframe(history_df[["record_id", "batch_id", "inserted_at", "source_type", "source_file", "atleta_id", "data_avaliacao", "passe_score", "decisao_score", "resiliencia_score"]], use_container_width=True, hide_index=True)
        record_to_delete = st.selectbox("Eliminar registo tecnico/psicologico", options=history_df["record_id"].astype(str).tolist(), key="delete_technical_record")
        if st.button("Eliminar registo tecnico/psicologico selecionado", key="delete_technical_btn"):
            delete_record("technical", record_to_delete)
            st.success("Registo tecnico/psicologico eliminado com sucesso.")
            st.rerun()


st.title("Base de Dados Futsal")
st.caption("Parquet local com ficha mestre de atletas, ficha antropometrica/fisica e ficha tecnica/psicologica.")

athletes_df = read_athletes()
physical_df = read_physical_records()
technical_df = read_technical_records()

_render_summary_metrics(athletes_df, physical_df, technical_df)

tab_registry, tab_create, tab_physical, tab_technical = st.tabs(
    ["Atletas", "Criar Atleta", "Ficha Antropometria | Fisica", "Ficha Tecnica | Psicologica"]
)

with tab_registry:
    _render_athlete_registry(athletes_df, physical_df, technical_df)

with tab_create:
    _render_create_athlete_form(athletes_df)

with tab_physical:
    _render_physical_tab(athletes_df, physical_df)

with tab_technical:
    _render_technical_tab(athletes_df, technical_df)
