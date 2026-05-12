from __future__ import annotations

from datetime import date
from io import BytesIO, StringIO

import pandas as pd
import streamlit as st

from fpf_modules.futsal_parquet_store import PHYSICAL_COLUMNS, append_records, read_athletes


REPORT_PHYSICAL_COLUMN_MAP = {
    "Peso(kg)": "peso_kg",
    "Peso corporal (kg)": "peso_kg",
    "Altura (cm)": "altura_cm",
    "Altura Sentada (cm)": "altura_sentada_cm",
    "Altura sentada (cm)": "altura_sentada_cm",
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

MODEL_COLUMNS = [
    "ID",
    "Nome",
    "Posição",
    "Peso(kg)",
    "Altura (cm)",
    "Altura Sentada (cm)",
    "Envergadura (cm)",
    "Comprimento Perna (cm)",
    "Sprint 10m(s)",
    "Sprint 20m (s)",
    "Teste 5-0-5 (esq)",
    "Teste 5-0-5 (dir)",
    "SJ altura (cm)",
    "CMJ altura (cm)",
    "DJ caixa (m)",
    "DJ altura (cm)",
    "DJ RSI",
    "DJ RSI mod (m/s)",
    "DJ contacto (ms)",
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
    "Contact 1 (ms)",
    "Contact 2 (ms)",
    "Contact 3 (ms)",
    "Contact 4 (ms)",
    "Contact 5 (ms)",
    "Contact 6 (ms)",
    "Contact 7 (ms)",
    "Contact 8 (ms)",
    "Contact 9 (ms)",
    "Contact 10 (ms)",
    "10J RSI 10-5",
    "10J CMJ (cm)",
    "10J média saltos (cm)",
    "10J máximo (cm)",
    "10J mínimo (cm)",
    "Índice fadiga 10J (%)",
]


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


def _clean_number(value):
    if value in ("", None) or pd.isna(value):
        return None
    try:
        return float(str(value).strip().replace(",", "."))
    except Exception:
        return None


def _normalize_length_series(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    # In the team Excel, stature-related fields may come in meters (e.g. 1.59).
    meter_mask = numeric.notna() & (numeric > 0) & (numeric < 3)
    numeric.loc[meter_mask] = numeric.loc[meter_mask] * 100
    return numeric


def _calculate_decimal_age(birth_date, reference_date) -> float | None:
    birth = _clean_date(birth_date)
    ref = _clean_date(reference_date)
    if not birth or not ref:
        return None
    return (ref - birth).days / 365.25


def _calculate_maturity_offset(genero, birth_date, reference_date, peso_kg, altura_cm, altura_sentada_cm) -> float | None:
    age = _calculate_decimal_age(birth_date, reference_date)
    weight = _clean_number(peso_kg)
    stature = _clean_number(altura_cm)
    sitting_height = _clean_number(altura_sentada_cm)
    sex = _clean_text_value(genero).lower()
    if age is None or weight is None or stature is None or sitting_height is None:
        return None

    leg_length = stature - sitting_height
    if leg_length <= 0:
        return None

    weight_height_ratio = (weight / stature) * 100 if stature else None
    if weight_height_ratio is None:
        return None

    if sex == "masculino":
        return (
            -9.236
            + (0.0002708 * (leg_length * sitting_height))
            - (0.001663 * (age * leg_length))
            + (0.007216 * (age * sitting_height))
            + (0.02292 * weight_height_ratio)
        )

    return (
        -9.376
        + (0.0001882 * (leg_length * sitting_height))
        + (0.0022 * (age * leg_length))
        + (0.005841 * (age * sitting_height))
        - (0.002658 * (age * weight))
        + (0.07693 * weight_height_ratio)
    )


def _classify_maturity_offset(maturity_offset: float | None) -> str:
    value = _clean_number(maturity_offset)
    if value is None:
        return ""
    if value < -1:
        return "Pre-PHV"
    if value <= 1:
        return "Circa-PHV"
    return "Post-PHV"


def _read_table_upload(uploaded_file) -> pd.DataFrame:
    uploaded_file.seek(0)
    file_name = str(getattr(uploaded_file, "name", "") or "").lower()
    if file_name.endswith((".xlsx", ".xls")):
        try:
            return pd.read_excel(uploaded_file)
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Para ler ficheiros Excel (.xlsx/.xls) nesta página, a app precisa da biblioteca 'openpyxl' instalada."
            ) from exc
    raw_bytes = uploaded_file.getvalue()
    for sep in [",", ";", "\t"]:
        try:
            return pd.read_csv(StringIO(raw_bytes.decode("utf-8-sig")), sep=sep)
        except Exception:
            continue
    raise RuntimeError("Nao foi possivel ler o ficheiro.")


def _validate_athlete_ids(df_upload: pd.DataFrame, athletes_df: pd.DataFrame) -> None:
    athlete_ids = set(athletes_df["atleta_id"].astype(str).str.strip().tolist()) if not athletes_df.empty else set()
    upload_ids = set(df_upload["atleta_id"].fillna("").astype(str).str.strip().tolist())
    missing = sorted([athlete_id for athlete_id in upload_ids if athlete_id and athlete_id not in athlete_ids])
    if missing:
        raise RuntimeError(f"Os seguintes atleta_id nao existem na ficha mestre: {', '.join(missing[:15])}")


def _prepare_bulk_physical_records(df_upload: pd.DataFrame, athletes_df: pd.DataFrame, data_avaliacao) -> pd.DataFrame:
    if df_upload is None or df_upload.empty:
        raise RuntimeError("O ficheiro nao contem linhas.")

    work_df = df_upload.copy()
    work_df.columns = [str(col).strip() for col in work_df.columns]

    if "ID" not in work_df.columns:
        raise RuntimeError("O modelo tem de incluir a coluna 'ID'.")

    work_df["atleta_id"] = work_df["ID"].fillna("").astype(str).str.strip()
    work_df = work_df[work_df["atleta_id"].ne("")].copy()
    if work_df.empty:
        raise RuntimeError("Nenhuma linha valida foi encontrada no modelo.")

    rename_map = {source: target for source, target in REPORT_PHYSICAL_COLUMN_MAP.items() if source in work_df.columns}
    work_df = work_df.rename(columns=rename_map)
    work_df["data_avaliacao"] = pd.to_datetime(data_avaliacao).date()

    for col in PHYSICAL_COLUMNS:
        if col not in work_df.columns:
            work_df[col] = pd.NA

    for col in PHYSICAL_COLUMNS:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")

    for col in ["altura_cm", "altura_sentada_cm", "envergadura_cm", "comprimento_perna_cm"]:
        if col in work_df.columns:
            work_df[col] = _normalize_length_series(work_df[col])

    _validate_athlete_ids(work_df[["atleta_id"]].copy(), athletes_df)
    athlete_meta = athletes_df[["atleta_id", "data_nascimento", "genero"]].copy()
    work_df = work_df.merge(athlete_meta, on="atleta_id", how="left")
    work_df["salto_maturacional"] = work_df.apply(
        lambda row: _calculate_maturity_offset(
            row.get("genero"),
            row.get("data_nascimento"),
            row.get("data_avaliacao"),
            row.get("peso_kg"),
            row.get("altura_cm"),
            row.get("altura_sentada_cm"),
        ),
        axis=1,
    )
    work_df["estado_maturacional"] = work_df["salto_maturacional"].map(_classify_maturity_offset)
    return work_df[["atleta_id", "data_avaliacao"] + PHYSICAL_COLUMNS].copy()


def _build_model_excel() -> bytes:
    output = BytesIO()
    try:
        with pd.ExcelWriter(output) as writer:
            pd.DataFrame(columns=MODEL_COLUMNS).to_excel(writer, sheet_name="Dados_Atletas", index=False)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Para gerar o modelo Excel (.xlsx), a app precisa da biblioteca 'openpyxl' instalada."
        ) from exc
    return output.getvalue()


def _build_model_csv() -> bytes:
    return pd.DataFrame(columns=MODEL_COLUMNS).to_csv(index=False).encode("utf-8-sig")


st.title("Inserção de Dados")
st.caption("Importação em lote para uma seleção inteira, usando o ID existente na base de dados.")

success_message = st.session_state.pop("futsal_bulk_insert_success", "")
if success_message:
    st.success(success_message)

athletes_df = read_athletes()
if athletes_df.empty:
    st.info("Ainda nao existem atletas registadas.")
    st.stop()

try:
    excel_model = _build_model_excel()
except RuntimeError as exc:
    st.warning(str(exc))
    st.download_button(
        "Descarregar modelo base em CSV",
        data=_build_model_csv(),
        file_name="modelo_insercao_dados_futsal.csv",
        mime="text/csv",
    )
else:
    st.download_button(
        "Descarregar modelo Excel base",
        data=excel_model,
        file_name="modelo_insercao_dados_futsal.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

data_avaliacao = st.date_input("Data da avaliacao do lote", value=date.today(), format="DD/MM/YYYY")
uploaded_file = st.file_uploader("Carregar modelo Excel da seleção", type=["xlsx", "xls", "csv"], key="futsal_bulk_selection_upload")

if uploaded_file is not None:
    try:
        raw_df = _read_table_upload(uploaded_file)
        prepared_df = _prepare_bulk_physical_records(raw_df, athletes_df, data_avaliacao)
    except Exception as exc:
        st.error(str(exc))
    else:
        preview_df = prepared_df.merge(
            athletes_df[["atleta_id", "nome", "posicao"]],
            on="atleta_id",
            how="left",
        )
        st.markdown("**Pré-visualização**")
        preview_columns = [
            "atleta_id",
            "nome",
            "posicao",
            "data_avaliacao",
        ]
        preview_columns.extend(PHYSICAL_COLUMNS)
        st.dataframe(
            preview_df[preview_columns],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(f"Linhas prontas a importar: {len(prepared_df)}")
        if st.button("Importar dados da seleção", type="primary", key="import_bulk_selection_button"):
            result = append_records("physical", prepared_df, source_type="selection_excel", source_file=uploaded_file.name)
            st.session_state["futsal_bulk_insert_success"] = (
                f"Operação concluída. Lote importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}"
            )
            st.rerun()
