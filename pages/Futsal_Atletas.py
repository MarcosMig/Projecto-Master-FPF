from __future__ import annotations

import re
import unicodedata
from datetime import date
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpf_modules.futsal_analytics import build_metric_history
from fpf_modules.futsal_parquet_store import (
    ATHLETE_COLUMNS,
    PHYSICAL_COLUMNS,
    TECHNICAL_COLUMNS,
    append_records,
    append_athlete_pathway_entry,
    delete_athlete,
    delete_athlete_pathway_entry,
    delete_photo,
    delete_record,
    generate_athlete_id,
    latest_records_by_athlete,
    read_athlete_history,
    read_athlete_pathway,
    read_athletes,
    read_physical_records,
    read_technical_records,
    save_photo,
    update_athlete_pathway_entry,
    update_record,
    upsert_athlete,
)


ATHLETE_BIRTHDATE_MIN = date(1980, 1, 1)
ATHLETE_BIRTHDATE_MAX = date.today()
GENDER_OPTIONS = ["Masculino", "Feminino"]
POSITION_OPTIONS = ["", "GR", "Fixo", "Ala", "Pivot", "Universal"]
SELECTION_OPTIONS = ["", "S12", "S13", "S14", "S15", "S16", "S17"]
STATE_OPTIONS = ["", "Observado", "Referenciado", "Processo Seleção", "Seleção Distrital", "Estágio Seleção Nacional", "Internacional"]

def _season_label(start_year: int) -> str:
    return f"{start_year}/{str((start_year + 1) % 100).zfill(2)}"


def _season_options(selected: str = "") -> list[str]:
    current_year = date.today().year
    options = [""] + [_season_label(year) for year in range(current_year - 3, current_year + 5)]
    selected_value = _clean_text_value(selected)
    if selected_value and selected_value not in options:
        options.append(selected_value)
    return options


def _selection_display(selecao_base, epoca) -> str:
    selecao_value = _clean_text_value(selecao_base)
    epoca_value = _clean_text_value(epoca)
    if selecao_value and epoca_value:
        return f"{selecao_value} | {epoca_value}"
    return selecao_value or epoca_value


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
    "altura_sentada_cm",
    "salto_maturacional",
    "estado_maturacional",
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
    "um_x_um_ofensivo_score",
    "um_x_um_defensivo_score",
    "lateralidade_score",
    "imprevisibilidade_score",
    "leitura_jogo_score",
    "dominio_espaco_score",
    "posicionamento_prontidao_score",
    "defesa_membros_superiores_score",
    "defesa_membros_inferiores_score",
    "defesa_6m_ocupa_espaco_score",
    "espirito_equipa_score",
    "controlo_emocional_score",
    "tenacidade_resiliencia_score",
    "atencao_concentracao_score",
    "observacoes",
]

REPORT_PHYSICAL_COLUMN_MAP = {
    "Peso corporal (kg)": "peso_kg",
    "Altura (cm)": "altura_cm",
    "Envergadura (cm)": "envergadura_cm",
    "Comprimento Perna (cm)": "comprimento_perna_cm",
    "Altura sentada (cm)": "altura_sentada_cm",
    "Salto maturacional": "salto_maturacional",
    "Estado maturacional": "estado_maturacional",
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


def _normalize_length_value(value):
    number = _clean_number(value)
    if number is None:
        return None
    return number * 100 if 0 < number < 3 else number


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


def _render_stat_strip(items: list[tuple[str, str | int]], class_name: str, columns: int | None = None) -> None:
    column_count = columns or len(items)
    cards_html = "".join(
        f"""
        <div class="{class_name}-card">
            <div class="{class_name}-label">{label}</div>
            <div class="{class_name}-value">{value}</div>
        </div>
        """
        for label, value in items
    )
    st.markdown(
        f"""
        <style>
        .{class_name}-strip {{
            display: grid;
            grid-template-columns: repeat({column_count}, minmax(160px, 160px));
            justify-content: start;
            gap: 8px;
            margin: 0.35rem 0 1rem 0;
        }}
        .{class_name}-card {{
            background: #f7f8fb;
            border: 1px solid #e6e8ef;
            border-radius: 10px;
            padding: 8px 10px;
            min-height: 56px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .{class_name}-label {{
            font-size: 10px;
            line-height: 1.2;
            color: #667085;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }}
        .{class_name}-value {{
            font-size: 18px;
            line-height: 1.1;
            font-weight: 700;
            color: #101828;
        }}
        @media (max-width: 1200px) {{
            .{class_name}-strip {{
                grid-template-columns: repeat(4, minmax(160px, 160px));
            }}
        }}
        @media (max-width: 720px) {{
            .{class_name}-strip {{
                grid-template-columns: repeat(2, minmax(160px, 160px));
            }}
        }}
        </style>
        <div class="{class_name}-strip">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _interpolate_hex(color_a: tuple[int, int, int], color_b: tuple[int, int, int], factor: float) -> str:
    factor = max(0.0, min(1.0, factor))
    channels = [
        int(round(color_a[idx] + (color_b[idx] - color_a[idx]) * factor))
        for idx in range(3)
    ]
    return "#{:02x}{:02x}{:02x}".format(*channels)


def _hsl_to_hex(h: float, s: float, l: float) -> str:
    h = h % 360.0
    s = max(0.0, min(1.0, s))
    l = max(0.0, min(1.0, l))
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs((h / 60.0) % 2 - 1))
    m = l - c / 2
    if h < 60:
        r1, g1, b1 = c, x, 0
    elif h < 120:
        r1, g1, b1 = x, c, 0
    elif h < 180:
        r1, g1, b1 = 0, c, x
    elif h < 240:
        r1, g1, b1 = 0, x, c
    elif h < 300:
        r1, g1, b1 = x, 0, c
    else:
        r1, g1, b1 = c, 0, x
    rgb = [int(round((channel + m) * 255)) for channel in (r1, g1, b1)]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def _score_gradient(score_value) -> tuple[str, str]:
    number = _clean_number(score_value)
    if number is None:
        return "#667085", "#98a2b3"
    clipped = max(0.0, min(100.0, number))
    hue = 120.0 * (clipped / 100.0)
    base = _hsl_to_hex(hue, 0.76, 0.40)
    accent = _hsl_to_hex(hue, 0.88, 0.52)
    return base, accent


@st.cache_data(show_spinner=False)
def _get_metric_history_cached(cache_key: str) -> pd.DataFrame:
    return build_metric_history()


def _build_metric_history_cache_key() -> str:
    physical = globals().get("physical_df")
    technical = globals().get("technical_df")

    def _frame_stamp(df: pd.DataFrame | None, timestamp_col: str) -> str:
        if df is None or df.empty or timestamp_col not in df.columns:
            return "0:"
        stamp = pd.to_datetime(df[timestamp_col], errors="coerce")
        max_stamp = "" if stamp.dropna().empty else stamp.max().isoformat()
        return f"{len(df)}:{max_stamp}"

    return f"phys={_frame_stamp(physical, 'inserted_at')}|tech={_frame_stamp(technical, 'inserted_at')}"


def _to_radar_scale_1_7(value) -> float | None:
    number = _clean_number(value)
    if number is None:
        return None
    clipped = min(max(number, 1.0), 7.0)
    return ((clipped - 1.0) / 6.0) * 100.0


def _comparison_scope_label(scope_fields: list[str]) -> str:
    labels = {
        "genero": "Genero",
        "selecao": "Selecao",
        "escalao_avaliacao": "Escalao",
        "posicao": "Posicao",
    }
    if not scope_fields:
        return "Historico total"
    return " + ".join(labels.get(field, field) for field in scope_fields)


def _metric_comparison(metric_key: str, value, athlete_row: pd.Series, scope_sets: list[list[str]]) -> dict | None:
    number = _clean_number(value)
    if number is None:
        return None
    history_df = _get_metric_history_cached(_build_metric_history_cache_key())
    metric_df = history_df[history_df["metric_key"] == metric_key].copy()
    if metric_df.empty:
        return None

    field_values = {
        "genero": _clean_text_value(athlete_row.get("genero")),
        "selecao": _clean_text_value(athlete_row.get("selecao")),
        "escalao_avaliacao": _clean_text_value(athlete_row.get("escalao")),
        "posicao": _clean_text_value(athlete_row.get("posicao")),
    }

    chosen_df = metric_df
    chosen_scope: list[str] = []
    for scope_fields in scope_sets:
        scoped_df = metric_df.copy()
        valid_scope = True
        for field in scope_fields:
            field_value = field_values.get(field, "")
            if not field_value:
                valid_scope = False
                break
            scoped_df = scoped_df[scoped_df[field].astype(str) == field_value].copy()
        if valid_scope and len(scoped_df) >= 4 and scoped_df["atleta_id"].nunique() >= 3:
            chosen_df = scoped_df
            chosen_scope = scope_fields
            break

    values = pd.to_numeric(chosen_df["metric_value"], errors="coerce").dropna()
    if values.empty:
        return None
    direction = _clean_text_value(chosen_df["direction"].iloc[0])
    p25 = float(values.quantile(0.25))
    p50 = float(values.quantile(0.50))
    p75 = float(values.quantile(0.75))
    if direction == "lower":
        perf_values = -values
        perf_value = -number
    else:
        perf_values = values
        perf_value = number
    percentile = float((perf_values <= perf_value).mean() * 100)

    if direction == "lower":
        if number <= p25:
            band = "Acima do perfil do grupo"
        elif number <= p50:
            band = "Entre P25 e P50"
        elif number <= p75:
            band = "Entre P50 e P75"
        else:
            band = "Abaixo do perfil do grupo"
        median_text = "Acima da mediana do grupo" if number <= p50 else "Abaixo da mediana do grupo"
    else:
        if number >= p75:
            band = "Acima do perfil do grupo"
        elif number >= p50:
            band = "Entre P50 e P75"
        elif number >= p25:
            band = "Entre P25 e P50"
        else:
            band = "Abaixo do perfil do grupo"
        median_text = "Acima da mediana do grupo" if number >= p50 else "Abaixo da mediana do grupo"

    return {
        "band": band,
        "median_text": median_text,
        "p25": p25,
        "p50": p50,
        "p75": p75,
        "direction": direction,
        "percentile": percentile,
        "scope_label": _comparison_scope_label(chosen_scope),
        "n_avaliacoes": int(len(chosen_df)),
        "n_atletas": int(chosen_df["atleta_id"].nunique()),
    }


def _render_spider_map(title: str, axes: list[tuple[str, float | None]], subtitle: str = "", chart_key: str = "") -> None:
    valid_axes = [(label, value) for label, value in axes if value is not None]
    if not valid_axes:
        st.info(f"Sem dados suficientes para o mapa de {title.lower()}.")
        return
    categories = [label for label, _ in valid_axes]
    values = [value for _, value in valid_axes]
    categories.append(categories[0])
    values.append(values[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            mode="lines+markers",
            line=dict(color="#0f766e", width=3),
            marker=dict(size=7, color="#0f766e"),
            fillcolor="rgba(15, 118, 110, 0.22)",
            hovertemplate="%{theta}: %{r:.1f}<extra></extra>",
            name=title,
        )
    )
    fig.update_layout(
        title={"text": title, "x": 0.02, "xanchor": "left", "font": {"size": 16}},
        margin=dict(l=30, r=30, t=55, b=20),
        height=420,
        showlegend=False,
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickvals=[20, 40, 60, 80, 100],
                ticktext=["20", "40", "60", "80", "100"],
                gridcolor="#d0d5dd",
                linecolor="#d0d5dd",
            ),
            angularaxis=dict(
                gridcolor="#eaecf0",
                linecolor="#d0d5dd",
                tickfont=dict(size=11),
            ),
        ),
        annotations=[
            dict(
                text=subtitle,
                x=0.02,
                y=1.08,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=11, color="#667085"),
                align="left",
            )
        ] if subtitle else [],
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False}, key=chart_key or title)


def _render_absolute_metric_grid(
    athlete_row: pd.Series,
    title: str,
    metrics: list[dict],
    scope_sets: list[list[str]],
    columns_count: int = 3,
) -> None:
    st.markdown(f"**{title}**")
    valid_metrics = [metric for metric in metrics if _clean_number(athlete_row.get(metric["row_key"])) is not None]
    if not valid_metrics:
        st.info(f"Sem dados disponíveis para {title.lower()}.")
        return

    rows = [valid_metrics[idx:idx + columns_count] for idx in range(0, len(valid_metrics), columns_count)]
    for metric_row in rows:
        cols = st.columns(columns_count)
        for idx, metric in enumerate(metric_row):
            col = cols[idx]
            metric_value = athlete_row.get(metric["row_key"])
            formatted_value = _format_metric(metric_value, metric.get("suffix", ""), metric.get("decimals", 1))
            comparison = _metric_comparison(metric["metric_key"], metric_value, athlete_row, scope_sets)
            col.markdown(f"**{metric['label']}:** {formatted_value}")
            if comparison:
                col.caption(
                    f"{comparison['band']} | Mediana: {_format_metric(comparison['p50'], metric.get('suffix', ''), metric.get('decimals', 1))}"
                )
                col.caption(f"Grupo: {comparison['scope_label']}")


def _render_grouped_absolute_metrics(
    athlete_row: pd.Series,
    groups: list[tuple[str, list[dict]]],
    scope_sets: list[list[str]],
    columns_count: int = 3,
) -> None:
    for title, metrics in groups:
        _render_absolute_metric_grid(
            athlete_row,
            title,
            metrics,
            scope_sets=scope_sets,
            columns_count=columns_count,
        )


def _build_context_radar_axes(
    athlete_row: pd.Series,
    metrics: list[dict],
    scope_sets: list[list[str]],
) -> list[tuple[str, float | None]]:
    axes: list[tuple[str, float | None]] = []
    for metric in metrics:
        comparison = _metric_comparison(
            metric["metric_key"],
            athlete_row.get(metric["row_key"]),
            athlete_row,
            scope_sets,
        )
        axes.append((metric["label"], None if comparison is None else comparison["percentile"]))
    return axes


def _build_grouped_context_radar_axes(
    athlete_row: pd.Series,
    grouped_metrics: list[tuple[str, list[dict]]],
    scope_sets: list[list[str]],
) -> list[tuple[str, float | None]]:
    axes: list[tuple[str, float | None]] = []
    for group_label, metrics in grouped_metrics:
        percentiles: list[float] = []
        for metric in metrics:
            comparison = _metric_comparison(
                metric["metric_key"],
                athlete_row.get(metric["row_key"]),
                athlete_row,
                scope_sets,
            )
            percentile = None if comparison is None else comparison["percentile"]
            if percentile is not None:
                percentiles.append(float(percentile))
        axes.append((group_label, None if not percentiles else sum(percentiles) / len(percentiles)))
    return axes


def _average_percentile_for_metrics(
    athlete_row: pd.Series,
    metrics: list[dict],
    scope_sets: list[list[str]],
) -> float | None:
    values: list[float] = []
    for metric in metrics:
        comparison = _metric_comparison(
            metric["metric_key"],
            athlete_row.get(metric["row_key"]),
            athlete_row,
            scope_sets,
        )
        if comparison is not None and comparison.get("percentile") is not None:
            values.append(float(comparison["percentile"]))
    if not values:
        return None
    return sum(values) / len(values)


def _compute_profile_scores(athlete_row: pd.Series) -> dict[str, float | None]:
    scope_general = [["genero", "escalao_avaliacao"], ["genero"], []]
    scope_field = [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []]
    scope_psych = [["genero", "selecao", "escalao_avaliacao"], ["genero", "escalao_avaliacao"], ["genero"], []]

    anthropometry_metrics = [
        {"row_key": "fis_peso_kg", "metric_key": "peso_kg"},
        {"row_key": "fis_altura_cm", "metric_key": "altura_cm"},
        {"row_key": "fis_envergadura_cm", "metric_key": "envergadura_cm"},
        {"row_key": "fis_comprimento_perna_cm", "metric_key": "comprimento_perna_cm"},
        {"row_key": "fis_altura_sentada_cm", "metric_key": "altura_sentada_cm"},
        {"row_key": "fis_salto_maturacional", "metric_key": "salto_maturacional"},
    ]
    physical_metrics = [
        {"row_key": "fis_sprint_10m_s", "metric_key": "sprint_10m_s"},
        {"row_key": "fis_sprint_20m_s", "metric_key": "sprint_20m_s"},
        {"row_key": "fis_teste_505_esq_s", "metric_key": "teste_505_esq_s"},
        {"row_key": "fis_teste_505_dir_s", "metric_key": "teste_505_dir_s"},
        {"row_key": "fis_sj_altura_cm", "metric_key": "sj_altura_cm"},
        {"row_key": "fis_cmj_altura_cm", "metric_key": "cmj_altura_cm"},
        {"row_key": "fis_dj_altura_cm", "metric_key": "dj_altura_cm"},
        {"row_key": "fis_dj_rsi", "metric_key": "dj_rsi"},
        {"row_key": "fis_j10_rsi_10_5", "metric_key": "j10_rsi_10_5"},
        {"row_key": "fis_j10_media_saltos_cm", "metric_key": "j10_media_saltos_cm"},
        {"row_key": "fis_indice_fadiga_10j_pct", "metric_key": "indice_fadiga_10j_pct"},
    ]
    is_goalkeeper = _clean_text_value(athlete_row.get("posicao")) == "GR"
    technical_metrics = (
        [
            {"row_key": "tec_reposicao_pe_score", "metric_key": "reposicao_pe_score"},
            {"row_key": "tec_reposicao_mao_score", "metric_key": "reposicao_mao_score"},
            {"row_key": "tec_posicionamento_prontidao_score", "metric_key": "posicionamento_prontidao_score"},
            {"row_key": "tec_defesa_membros_superiores_score", "metric_key": "defesa_membros_superiores_score"},
            {"row_key": "tec_defesa_membros_inferiores_score", "metric_key": "defesa_membros_inferiores_score"},
        ]
        if is_goalkeeper
        else [
            {"row_key": "tec_um_x_um_ofensivo_score", "metric_key": "um_x_um_ofensivo_score"},
            {"row_key": "tec_um_x_um_defensivo_score", "metric_key": "um_x_um_defensivo_score"},
            {"row_key": "tec_lateralidade_score", "metric_key": "lateralidade_score"},
        ]
    )
    tactical_metrics = (
        [
            {"row_key": "tec_tomada_decisao_score", "metric_key": "tomada_decisao_score"},
            {"row_key": "tec_comunicacao_score", "metric_key": "comunicacao_score"},
            {"row_key": "tec_posicionamento_prontidao_score", "metric_key": "posicionamento_prontidao_score"},
            {"row_key": "tec_defesa_6m_ocupa_espaco_score", "metric_key": "defesa_6m_ocupa_espaco_score"},
            {"row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score"},
        ]
        if is_goalkeeper
        else [
            {"row_key": "tec_imprevisibilidade_score", "metric_key": "imprevisibilidade_score"},
            {"row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score"},
            {"row_key": "tec_dominio_espaco_score", "metric_key": "dominio_espaco_score"},
        ]
    )
    psychological_metrics = [
        {"row_key": "tec_espirito_equipa_score", "metric_key": "espirito_equipa_score"},
        {"row_key": "tec_controlo_emocional_score", "metric_key": "controlo_emocional_score"},
        {"row_key": "tec_tenacidade_resiliencia_score", "metric_key": "tenacidade_resiliencia_score"},
        {"row_key": "tec_atencao_concentracao_score", "metric_key": "atencao_concentracao_score"},
    ]

    return {
        "ANT": _average_percentile_for_metrics(athlete_row, anthropometry_metrics, scope_general),
        "FIS": _average_percentile_for_metrics(athlete_row, physical_metrics, scope_general),
        "TEC": _average_percentile_for_metrics(athlete_row, technical_metrics, scope_field),
        "TAT": _average_percentile_for_metrics(athlete_row, tactical_metrics, scope_field),
        "PSI": _average_percentile_for_metrics(athlete_row, psychological_metrics, scope_psych),
    }


def _render_profile_score_strip(athlete_row: pd.Series) -> None:
    scores = _compute_profile_scores(athlete_row)

    def score_text(key: str) -> str:
        value = scores.get(key)
        return "--" if value is None else str(int(round(value)))

    ant_base, ant_accent = _score_gradient(scores.get("ANT"))
    fis_base, fis_accent = _score_gradient(scores.get("FIS"))
    tec_base, tec_accent = _score_gradient(scores.get("TEC"))
    tat_base, tat_accent = _score_gradient(scores.get("TAT"))
    psi_base, psi_accent = _score_gradient(scores.get("PSI"))

    html = f"""
    <style>
    .player-score-strip {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 10px;
        margin: 10px 0 4px 0;
    }}
    .player-score-pill {{
        text-align: center;
        color: white;
        border-radius: 18px;
        padding: 12px 8px 10px;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .player-score-pill.ant {{ background: linear-gradient(135deg, {ant_base}, {ant_accent}); }}
    .player-score-pill.fis {{ background: linear-gradient(135deg, {fis_base}, {fis_accent}); }}
    .player-score-pill.tec {{ background: linear-gradient(135deg, {tec_base}, {tec_accent}); }}
    .player-score-pill.tat {{ background: linear-gradient(135deg, {tat_base}, {tat_accent}); }}
    .player-score-pill.psi {{ background: linear-gradient(135deg, {psi_base}, {psi_accent}); }}
    .player-score-value {{
        font-size: 26px;
        line-height: 1;
        font-weight: 800;
    }}
    .player-score-label {{
        font-size: 11px;
        letter-spacing: 0.08em;
        margin-top: 5px;
        text-transform: uppercase;
        color: rgba(255,255,255,0.84);
        font-weight: 700;
    }}
    @media (max-width: 720px) {{
        .player-score-strip {{
            grid-template-columns: repeat(3, minmax(0, 1fr));
        }}
    }}
    </style>
    <div class="player-score-strip">
        <div class="player-score-pill ant"><div class="player-score-value">{score_text("ANT")}</div><div class="player-score-label">ANT</div></div>
        <div class="player-score-pill fis"><div class="player-score-value">{score_text("FIS")}</div><div class="player-score-label">FIS</div></div>
        <div class="player-score-pill tec"><div class="player-score-value">{score_text("TEC")}</div><div class="player-score-label">TEC</div></div>
        <div class="player-score-pill tat"><div class="player-score-value">{score_text("TAT")}</div><div class="player-score-label">TAT</div></div>
        <div class="player-score-pill psi"><div class="player-score-value">{score_text("PSI")}</div><div class="player-score-label">PSI</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


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


def _calculate_decimal_age(birth_date, reference_date) -> float | None:
    birth = _clean_date(birth_date)
    ref = _clean_date(reference_date)
    if not birth or not ref:
        return None
    return (ref - birth).days / 365.25


def _calculate_maturity_offset(genero, birth_date, reference_date, peso_kg, altura_cm, altura_sentada_cm) -> float | None:
    age = _calculate_decimal_age(birth_date, reference_date)
    weight = _clean_number(peso_kg)
    stature = _normalize_length_value(altura_cm)
    sitting_height = _normalize_length_value(altura_sentada_cm)
    sex = _clean_text_value(genero).lower()
    if age is None or weight is None or stature is None or sitting_height is None:
        return None
    if stature <= 0 or sitting_height <= 0 or weight <= 0 or sitting_height >= stature:
        return None

    leg_length = stature - sitting_height
    weight_height_ratio = (weight / stature) * 100

    if sex == "feminino":
        return (
            -9.376
            + (0.0001882 * (leg_length * sitting_height))
            + (0.0022 * (age * leg_length))
            + (0.005841 * (age * sitting_height))
            - (0.002658 * (age * weight))
            + (0.07693 * weight_height_ratio)
        )

    return (
        -9.236
        + (0.0002708 * (leg_length * sitting_height))
        - (0.001663 * (age * leg_length))
        + (0.007216 * (age * sitting_height))
        + (0.02292 * weight_height_ratio)
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


def _selected_athlete_row(athletes_df: pd.DataFrame, atleta_id: str) -> pd.Series | None:
    if athletes_df is None or athletes_df.empty:
        return None
    row = athletes_df[athletes_df["atleta_id"].astype(str) == str(atleta_id)].head(1)
    return None if row.empty else row.iloc[0]


def _render_record_athlete_header(
    athletes_df: pd.DataFrame,
    athlete_options: list[str],
    selector_key: str,
) -> tuple[str, pd.Series | None]:
    selector_col, name_col = st.columns([0.8, 2.2], gap="small")
    selected_id = selector_col.selectbox("ID", options=athlete_options, key=selector_key)
    selected_athlete = _selected_athlete_row(athletes_df, selected_id)
    athlete_name = _clean_text_value(selected_athlete.get("nome")) if selected_athlete is not None else ""
    name_col.text_input(
        "Nome da atleta",
        value=athlete_name,
        disabled=True,
        key=f"{selector_key}_athlete_name",
    )

    photo_col, info_col = st.columns([0.7, 2.3], gap="large")
    with photo_col:
        photo_path = _clean_text_value(selected_athlete.get("foto_path")) if selected_athlete is not None else ""
        if photo_path:
            st.image(photo_path, width=120)
        else:
            st.caption("Sem foto registada.")

    with info_col:
        info_left, info_right = st.columns(2, gap="large")
        idade = _calculate_age(selected_athlete.get("data_nascimento")) if selected_athlete is not None else None
        escalao = _derive_escalao_from_age(idade)
        info_left.markdown(f"**Nome:** {athlete_name or '-'}")
        info_left.markdown(
            f"**Data nascimento:** {_format_date(selected_athlete.get('data_nascimento')) if selected_athlete is not None else '-'}"
        )
        info_left.markdown(f"**Idade:** {idade if idade is not None else '-'}")
        info_right.markdown(
            f"**Genero:** {_clean_text_value(selected_athlete.get('genero')) if selected_athlete is not None else '-'}"
        )
        info_right.markdown(f"**Escalao:** {escalao or '-'}")
        info_right.markdown(
            f"**Selecao:** {_clean_text_value(selected_athlete.get('selecao')) if selected_athlete is not None else '-'}"
        )

    return selected_id, selected_athlete


def _athlete_history_rows(
    athlete_row: pd.Series,
    athlete_history_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
) -> pd.DataFrame:
    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    history_frames: list[pd.DataFrame] = []

    athlete_events = athlete_history_df[athlete_history_df["atleta_id"].astype(str) == atleta_id].copy() if not athlete_history_df.empty else pd.DataFrame()
    if athlete_events.empty:
        synthetic_rows = [
            {
                "momento": athlete_row.get("created_at"),
                "tipo": "Criacao atleta",
                "data_avaliacao": None,
                "detalhe": "Criacao da atleta",
                "selecao": _selection_display(athlete_row.get("selecao"), athlete_row.get("epoca_atual")),
                "selecao_base": _clean_text_value(athlete_row.get("selecao")),
                "epoca": _clean_text_value(athlete_row.get("epoca_atual")),
                "estado": "",
                "ano": None,
                "escalao": "",
                "data_referencia": None,
                "observacoes": "",
            }
        ]
        updated_at = athlete_row.get("updated_at")
        created_at = athlete_row.get("created_at")
        if _clean_text_value(updated_at) and str(updated_at) != str(created_at):
            synthetic_rows.append(
                {
                    "momento": updated_at,
                    "tipo": "Atualizacao atleta",
                    "data_avaliacao": None,
                    "detalhe": "Atualizacao da ficha da atleta",
                    "selecao": _selection_display(athlete_row.get("selecao"), athlete_row.get("epoca_atual")),
                    "selecao_base": _clean_text_value(athlete_row.get("selecao")),
                    "epoca": _clean_text_value(athlete_row.get("epoca_atual")),
                    "estado": "",
                    "ano": None,
                    "escalao": "",
                    "data_referencia": None,
                    "observacoes": "",
                }
            )
        athlete_events = pd.DataFrame(synthetic_rows)
    else:
        athlete_events = athlete_events.rename(columns={"inserted_at": "momento"})
        athlete_events["tipo"] = athlete_events["event_type"].map(
            {
                "create_athlete": "Criacao atleta",
                "update_athlete": "Atualizacao atleta",
            }
        ).fillna("Evento atleta")
        athlete_events["detalhe"] = athlete_events["descricao"].fillna("")
        athlete_events["selecao_base"] = athlete_events["selecao"].map(_clean_text_value)
        athlete_events["epoca"] = ""
        athlete_events["selecao"] = athlete_events["selecao_base"]
        athlete_events["estado"] = ""
        athlete_events["ano"] = None
        athlete_events["escalao"] = ""
        athlete_events["data_referencia"] = None
        athlete_events["observacoes"] = ""
        athlete_events = athlete_events[["momento", "tipo", "data_nascimento", "detalhe", "selecao", "selecao_base", "epoca", "estado", "ano", "escalao", "data_referencia", "observacoes"]].rename(
            columns={"data_nascimento": "data_avaliacao"}
        )
        athlete_events["record_kind"] = "athlete_event"
        athlete_events["record_id"] = athlete_events.index.astype(str)
        athlete_events["source_type"] = "athlete_event"
    history_frames.append(athlete_events)

    pathway_rows = pathway_df[pathway_df["atleta_id"].astype(str) == atleta_id].copy() if not pathway_df.empty else pd.DataFrame()
    if not pathway_rows.empty:
        pathway_rows["momento"] = pathway_rows["inserted_at"]
        pathway_rows["tipo"] = "Atualizacao percurso"
        pathway_rows["data_avaliacao"] = pathway_rows["data_referencia"]
        pathway_rows["detalhe"] = pathway_rows["observacoes"].fillna("")
        pathway_rows["selecao_base"] = pathway_rows["selecao"].map(_clean_text_value)
        pathway_rows["epoca"] = pathway_rows["epoca"].map(_clean_text_value)
        pathway_rows["selecao"] = pathway_rows.apply(lambda row: _selection_display(row.get("selecao_base"), row.get("epoca")), axis=1)
        pathway_rows["estado"] = pathway_rows["estado"].map(_clean_text_value)
        pathway_rows["record_kind"] = "pathway"
        pathway_rows["record_id"] = pathway_rows["entry_id"]
        history_frames.append(pathway_rows[["momento", "tipo", "data_avaliacao", "detalhe", "selecao", "selecao_base", "epoca", "estado", "ano", "escalao", "data_referencia", "observacoes", "record_kind", "record_id"]])

    physical_rows = physical_df[physical_df["atleta_id"].astype(str) == atleta_id].copy() if not physical_df.empty else pd.DataFrame()
    if not physical_rows.empty:
        physical_rows["momento"] = physical_rows["inserted_at"]
        physical_rows["tipo"] = physical_rows["source_type"].map(
            {
                "manual_anthropometry": "Ficha Antropometrica",
                "manual_physical_tests": "Testes Fisicos",
                "upload": "Testes Fisicos",
                "team_report": "Testes Fisicos",
            }
        ).fillna("Testes Fisicos")
        physical_rows["detalhe"] = physical_rows.apply(
            lambda row: ", ".join(
                [
                    f"Sprint 10m {_format_metric(row.get('sprint_10m_s'), ' s', 2)}" if _clean_number(row.get("sprint_10m_s")) is not None else "",
                    f"Sprint 20m {_format_metric(row.get('sprint_20m_s'), ' s', 2)}" if _clean_number(row.get("sprint_20m_s")) is not None else "",
                    f"505 esq {_format_metric(row.get('teste_505_esq_s'), ' s', 2)}" if _clean_number(row.get("teste_505_esq_s")) is not None else "",
                    f"505 dir {_format_metric(row.get('teste_505_dir_s'), ' s', 2)}" if _clean_number(row.get("teste_505_dir_s")) is not None else "",
                    f"SJ {_format_metric(row.get('sj_altura_cm'), ' cm')}" if _clean_number(row.get("sj_altura_cm")) is not None else "",
                    f"CMJ {_format_metric(row.get('cmj_altura_cm'), ' cm')}" if _clean_number(row.get("cmj_altura_cm")) is not None else "",
                    f"DJ altura {_format_metric(row.get('dj_altura_cm'), ' cm')}" if _clean_number(row.get("dj_altura_cm")) is not None else "",
                    f"DJ RSI {_format_metric(row.get('dj_rsi'), '', 2)}" if _clean_number(row.get("dj_rsi")) is not None else "",
                    f"10J RSI 10-5 {_format_metric(row.get('j10_rsi_10_5'), '', 2)}" if _clean_number(row.get("j10_rsi_10_5")) is not None else "",
                    f"10J media {_format_metric(row.get('j10_media_saltos_cm'), ' cm')}" if _clean_number(row.get("j10_media_saltos_cm")) is not None else "",
                    f"Indice fadiga {_format_metric(row.get('indice_fadiga_10j_pct'), '%', 2)}" if _clean_number(row.get("indice_fadiga_10j_pct")) is not None else "",
                ]
            ).strip(", "),
            axis=1,
        )
        physical_rows["selecao"] = ""
        physical_rows["selecao_base"] = ""
        physical_rows["epoca"] = ""
        physical_rows["estado"] = ""
        physical_rows["ano"] = None
        physical_rows["escalao"] = ""
        physical_rows["data_referencia"] = None
        physical_rows["observacoes"] = ""
        physical_rows["record_kind"] = "physical"
        history_frames.append(physical_rows[["momento", "tipo", "data_avaliacao", "detalhe", "selecao", "selecao_base", "epoca", "estado", "ano", "escalao", "data_referencia", "observacoes", "record_kind", "record_id", "source_type"]])

    technical_rows = technical_df[technical_df["atleta_id"].astype(str) == atleta_id].copy() if not technical_df.empty else pd.DataFrame()
    if not technical_rows.empty:
        is_goalkeeper = _clean_text_value(athlete_row.get("posicao")) == "GR"
        technical_rows["momento"] = technical_rows["inserted_at"]
        technical_rows["tipo"] = technical_rows["source_type"].map(
            {
                "manual_technical": "Ficha Tecnica | Tatica | Psicologica",
                "upload": "Ficha Tecnica | Tatica | Psicologica",
            }
        ).fillna("Ficha Tecnica | Tatica | Psicologica")
        technical_rows["detalhe"] = technical_rows.apply(
            lambda row: ", ".join(
                (
                    [
                        f"Reposição pé {_format_metric(row.get('reposicao_pe_score'), '/7')}" if _clean_number(row.get("reposicao_pe_score")) is not None else "",
                        f"Reposição mão {_format_metric(row.get('reposicao_mao_score'), '/7')}" if _clean_number(row.get("reposicao_mao_score")) is not None else "",
                        f"Tomada decisão {_format_metric(row.get('tomada_decisao_score'), '/7')}" if _clean_number(row.get("tomada_decisao_score")) is not None else "",
                        f"Comunicação {_format_metric(row.get('comunicacao_score'), '/7')}" if _clean_number(row.get("comunicacao_score")) is not None else "",
                        f"Posicionamento {_format_metric(row.get('posicionamento_prontidao_score'), '/7')}" if _clean_number(row.get("posicionamento_prontidao_score")) is not None else "",
                        f"Defesa MS {_format_metric(row.get('defesa_membros_superiores_score'), '/7')}" if _clean_number(row.get("defesa_membros_superiores_score")) is not None else "",
                        f"Defesa MI {_format_metric(row.get('defesa_membros_inferiores_score'), '/7')}" if _clean_number(row.get("defesa_membros_inferiores_score")) is not None else "",
                        f"Defesa 6m {_format_metric(row.get('defesa_6m_ocupa_espaco_score'), '/7')}" if _clean_number(row.get("defesa_6m_ocupa_espaco_score")) is not None else "",
                        f"Leitura {_format_metric(row.get('leitura_jogo_score'), '/7')}" if _clean_number(row.get("leitura_jogo_score")) is not None else "",
                        f"Controlo emocional {_format_metric(row.get('controlo_emocional_score'), '/7')}" if _clean_number(row.get("controlo_emocional_score")) is not None else "",
                    ]
                    if is_goalkeeper
                    else [
                        f"1x1 Ofensivo {_format_metric(row.get('um_x_um_ofensivo_score'), '/7')}" if _clean_number(row.get("um_x_um_ofensivo_score")) is not None else "",
                        f"Leitura {_format_metric(row.get('leitura_jogo_score'), '/7')}" if _clean_number(row.get("leitura_jogo_score")) is not None else "",
                        f"Controlo emocional {_format_metric(row.get('controlo_emocional_score'), '/7')}" if _clean_number(row.get("controlo_emocional_score")) is not None else "",
                    ]
                )
            ).strip(", "),
            axis=1,
        )
        technical_rows["selecao"] = ""
        technical_rows["selecao_base"] = ""
        technical_rows["epoca"] = ""
        technical_rows["estado"] = ""
        technical_rows["ano"] = None
        technical_rows["escalao"] = ""
        technical_rows["data_referencia"] = None
        technical_rows["observacoes"] = ""
        technical_rows["record_kind"] = "technical"
        history_frames.append(technical_rows[["momento", "tipo", "data_avaliacao", "detalhe", "selecao", "selecao_base", "epoca", "estado", "ano", "escalao", "data_referencia", "observacoes", "record_kind", "record_id", "source_type"]])

    history_df = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    if history_df.empty:
        return history_df
    history_df["momento"] = pd.to_datetime(history_df["momento"], errors="coerce", utc=True).dt.tz_localize(None)
    history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce")
    history_df["data_temporal"] = history_df["data_avaliacao"]
    missing_temporal = history_df["data_temporal"].isna()
    history_df.loc[missing_temporal, "data_temporal"] = history_df.loc[missing_temporal, "momento"]
    history_df["data_temporal"] = pd.to_datetime(history_df["data_temporal"], errors="coerce")
    history_df = history_df.sort_values(
        ["data_temporal", "momento"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    history_df["momento"] = history_df["momento"].dt.strftime("%d/%m/%Y %H:%M")
    history_df["data_avaliacao"] = history_df["data_avaliacao"].dt.strftime("%d/%m/%Y").replace("NaT", "")
    history_df["selecao"] = history_df["selecao"].fillna("").map(_clean_text_value)
    history_df["estado"] = history_df["estado"].fillna("").map(_clean_text_value)
    history_df["detalhe"] = history_df["detalhe"].fillna("").replace("", "-")
    return history_df.drop(columns=["data_temporal"], errors="ignore")


def _render_summary_metrics(athletes_df: pd.DataFrame, pathway_df: pd.DataFrame) -> None:
    total_atletas = int(len(athletes_df))
    masculinos = int(athletes_df["genero"].fillna("").astype(str).eq("Masculino").sum()) if not athletes_df.empty else 0
    femininos = int(athletes_df["genero"].fillna("").astype(str).eq("Feminino").sum()) if not athletes_df.empty else 0

    latest_pathway = pd.DataFrame(columns=["atleta_id", "estado"])
    if pathway_df is not None and not pathway_df.empty:
        latest_pathway = pathway_df.copy()
        latest_pathway["ano"] = pd.to_numeric(latest_pathway["ano"], errors="coerce")
        latest_pathway["data_referencia"] = pd.to_datetime(latest_pathway["data_referencia"], errors="coerce", dayfirst=True)
        latest_pathway["inserted_at"] = pd.to_datetime(latest_pathway["inserted_at"], errors="coerce")
        latest_pathway = latest_pathway.sort_values(
            ["atleta_id", "ano", "data_referencia", "inserted_at"],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop_duplicates(subset=["atleta_id"], keep="first")

    state_series = latest_pathway["estado"].fillna("").astype(str) if not latest_pathway.empty else pd.Series(dtype="object")
    internacionais = int(state_series.eq("Internacional").sum())
    estagio_selecao_nacional = int(state_series.eq("Estágio Seleção Nacional").sum())
    selecao_distrital = int(state_series.eq("Seleção Distrital").sum())
    processo_selecao = int(state_series.eq("Processo Seleção").sum())
    referenciados = int(state_series.eq("Referenciado").sum())
    observados = int(state_series.eq("Observado").sum())

    _render_stat_strip(
        [
            ("Atletas", total_atletas),
            ("Masculinos", masculinos),
            ("Femininos", femininos),
        ],
        "ath-summary-top",
        columns=5,
    )
    _render_stat_strip(
        [
            ("Internacionais", internacionais),
            ("Estagio Selecao Nacional", estagio_selecao_nacional),
            ("Selecao Distrital", selecao_distrital),
            ("Processo Selecao", processo_selecao),
            ("Referenciados", referenciados),
            ("Observados", observados),
        ],
        "ath-summary-bottom",
        columns=6,
    )


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


def _athletes_with_latest(
    athletes_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
) -> pd.DataFrame:
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
    if pathway_df is not None and not pathway_df.empty:
        latest_pathway = pathway_df.copy()
        latest_pathway["ano"] = pd.to_numeric(latest_pathway["ano"], errors="coerce")
        latest_pathway["data_referencia"] = pd.to_datetime(latest_pathway["data_referencia"], errors="coerce", dayfirst=True)
        latest_pathway["inserted_at"] = pd.to_datetime(latest_pathway["inserted_at"], errors="coerce")
        latest_pathway = latest_pathway.sort_values(
            ["atleta_id", "ano", "data_referencia", "inserted_at"],
            ascending=[True, False, False, False],
            na_position="last",
        ).drop_duplicates(subset=["atleta_id"], keep="first")
        latest_pathway = latest_pathway.add_prefix("per_").rename(columns={"per_atleta_id": "atleta_id"})
        work_df = work_df.merge(latest_pathway, on="atleta_id", how="left")
        work_df["selecao_base_atual"] = work_df["per_selecao"].where(
            work_df["per_selecao"].notna() & (work_df["per_selecao"].astype(str) != ""),
            work_df["selecao"],
        )
        work_df["epoca_atual"] = work_df["per_epoca"].map(_clean_text_value)
        work_df["selecao_atual"] = [
            _selection_display(selecao_base, epoca)
            for selecao_base, epoca in zip(work_df["selecao_base_atual"], work_df["epoca_atual"])
        ]
        work_df["escalao_atual"] = work_df["escalao"]
        work_df["estado_atual"] = work_df["per_estado"].fillna("")
    else:
        work_df["selecao_base_atual"] = work_df["selecao"]
        work_df["epoca_atual"] = ""
        work_df["selecao_atual"] = work_df["selecao"]
        work_df["escalao_atual"] = work_df["escalao"]
        work_df["estado_atual"] = ""
    return work_df


def _latest_physical_snapshot(physical_df: pd.DataFrame, atleta_id: str) -> dict:
    if physical_df is None or physical_df.empty:
        return {}
    latest_df = latest_records_by_athlete(physical_df)
    row = latest_df[latest_df["atleta_id"].astype(str) == str(atleta_id)].head(1)
    return {} if row.empty else row.iloc[0].to_dict()


def _build_physical_record(physical_df: pd.DataFrame, atleta_id: str, data_avaliacao, updates: dict) -> pd.DataFrame:
    base = {col: None for col in ["atleta_id", "data_avaliacao"] + PHYSICAL_COLUMNS}
    latest_snapshot = _latest_physical_snapshot(physical_df, atleta_id)
    for col in PHYSICAL_COLUMNS:
        if col in latest_snapshot:
            base[col] = latest_snapshot.get(col)
    base["atleta_id"] = atleta_id
    base["data_avaliacao"] = _clean_date(data_avaliacao)
    for key, value in updates.items():
        base[key] = value
    return pd.DataFrame([base])


def _latest_technical_snapshot(technical_df: pd.DataFrame, atleta_id: str) -> dict:
    if technical_df is None or technical_df.empty:
        return {}
    latest_df = latest_records_by_athlete(technical_df)
    row = latest_df[latest_df["atleta_id"].astype(str) == str(atleta_id)].head(1)
    return {} if row.empty else row.iloc[0].to_dict()


def _build_technical_record(technical_df: pd.DataFrame, atleta_id: str, data_avaliacao, updates: dict) -> pd.DataFrame:
    base = {col: None for col in ["atleta_id", "data_avaliacao"] + TECHNICAL_COLUMNS}
    latest_snapshot = _latest_technical_snapshot(technical_df, atleta_id)
    for col in TECHNICAL_COLUMNS:
        if col in latest_snapshot:
            base[col] = latest_snapshot.get(col)
    base["atleta_id"] = atleta_id
    base["data_avaliacao"] = _clean_date(data_avaliacao)
    for key, value in updates.items():
        base[key] = value
    return pd.DataFrame([base])


def _latest_pathway_snapshot(pathway_df: pd.DataFrame, atleta_id: str) -> dict:
    if pathway_df is None or pathway_df.empty:
        return {}
    work_df = pathway_df[pathway_df["atleta_id"].astype(str) == str(atleta_id)].copy()
    if work_df.empty:
        return {}
    work_df["ano"] = pd.to_numeric(work_df["ano"], errors="coerce")
    work_df["data_referencia"] = pd.to_datetime(work_df["data_referencia"], errors="coerce", dayfirst=True)
    work_df["inserted_at"] = pd.to_datetime(work_df["inserted_at"], errors="coerce")
    work_df = work_df.sort_values(["ano", "data_referencia", "inserted_at"], ascending=[False, False, False], na_position="last")
    return work_df.head(1).iloc[0].to_dict()


def _record_row_by_id(df: pd.DataFrame, record_id: str) -> pd.Series | None:
    if df is None or df.empty:
        return None
    row = df[df["record_id"].astype(str) == str(record_id)].head(1)
    return None if row.empty else row.iloc[0]


def _toggle_state(key: str) -> None:
    st.session_state[key] = not st.session_state.get(key, False)


def _open_athlete_editor(atleta_id: str) -> None:
    st.session_state["open_athlete_editor_id"] = str(atleta_id)


def _close_athlete_editor() -> None:
    st.session_state["open_athlete_editor_id"] = ""


def _close_athlete_delete_confirm(atleta_id: str) -> None:
    st.session_state[f"confirm_delete_athlete_{atleta_id}"] = False


def _selected_table_row(df: pd.DataFrame, selection):
    selected_rows = selection.selection.rows if selection and selection.selection else []
    if not selected_rows:
        return None
    row_index = selected_rows[0]
    if row_index is None or row_index < 0 or row_index >= len(df):
        return None
    return df.iloc[row_index]


def _toggle_insert_panel(panel_key: str) -> None:
    st.session_state[panel_key] = not st.session_state.get(panel_key, False)


def _render_inline_anthropometry_insert(athlete_row: pd.Series, physical_df: pd.DataFrame) -> None:
    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    panel_key = f"open_insert_anth_{atleta_id}"
    if st.button("Inserir", key=f"insert_anth_button_{atleta_id}"):
        _toggle_state(panel_key)
        st.rerun()
    if not st.session_state.get(panel_key, False):
        return

    latest_snapshot = _latest_physical_snapshot(physical_df, atleta_id)
    with st.form(f"insert_anth_form_{atleta_id}"):
        data_avaliacao = st.date_input("Data avaliacao", value=date.today(), format="DD/MM/YYYY", key=f"anth_date_{atleta_id}")
        a1, a2, a3, a4, a5 = st.columns(5)
        peso_kg = a1.number_input("Peso corporal (kg)", min_value=0.0, value=float(latest_snapshot.get("peso_kg", 0.0) or 0.0), step=0.1, key=f"anth_weight_{atleta_id}")
        altura_cm = a2.number_input("Altura (cm)", min_value=0.0, value=float(latest_snapshot.get("altura_cm", 0.0) or 0.0), step=0.1, key=f"anth_height_{atleta_id}")
        envergadura_cm = a3.number_input("Envergadura (cm)", min_value=0.0, value=float(latest_snapshot.get("envergadura_cm", 0.0) or 0.0), step=0.1, key=f"anth_span_{atleta_id}")
        comprimento_perna_cm = a4.number_input("Comprimento Perna (cm)", min_value=0.0, value=float(latest_snapshot.get("comprimento_perna_cm", 0.0) or 0.0), step=0.1, key=f"anth_leg_{atleta_id}")
        altura_sentada_cm = a5.number_input("Altura sentada (cm)", min_value=0.0, value=float(latest_snapshot.get("altura_sentada_cm", 0.0) or 0.0), step=0.1, key=f"anth_sit_{atleta_id}")
        maturity_offset = _calculate_maturity_offset(athlete_row.get("genero"), athlete_row.get("data_nascimento"), data_avaliacao, peso_kg, altura_cm, altura_sentada_cm)
        maturity_status = _classify_maturity_offset(maturity_offset)
        b1, b2 = st.columns(2)
        b1.text_input("Salto maturacional", value="" if maturity_offset is None else f"{maturity_offset:.2f}", disabled=True, key=f"anth_offset_{atleta_id}")
        b2.text_input("Estado maturacional", value=maturity_status, disabled=True, key=f"anth_status_{atleta_id}")
        submitted = st.form_submit_button("Guardar registo antropometrico", type="primary")
    if submitted:
        new_record = _build_physical_record(physical_df, atleta_id, data_avaliacao, {
            "peso_kg": peso_kg if peso_kg > 0 else None,
            "altura_cm": altura_cm if altura_cm > 0 else None,
            "envergadura_cm": envergadura_cm if envergadura_cm > 0 else None,
            "comprimento_perna_cm": comprimento_perna_cm if comprimento_perna_cm > 0 else None,
            "altura_sentada_cm": altura_sentada_cm if altura_sentada_cm > 0 else None,
            "salto_maturacional": maturity_offset,
            "estado_maturacional": maturity_status,
        })
        append_records("physical", new_record, source_type="manual_anthropometry")
        st.session_state[panel_key] = False
        st.success("Registo antropometrico guardado com sucesso.")
        st.rerun()


def _render_inline_physical_insert(athlete_row: pd.Series, physical_df: pd.DataFrame) -> None:
    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    panel_key = f"open_insert_phys_{atleta_id}"
    if st.button("Inserir", key=f"insert_phys_button_{atleta_id}"):
        _toggle_state(panel_key)
        st.rerun()
    if not st.session_state.get(panel_key, False):
        return

    latest_snapshot = _latest_physical_snapshot(physical_df, atleta_id)
    with st.form(f"insert_phys_form_{atleta_id}"):
        data_avaliacao = st.date_input("Data avaliacao", value=date.today(), format="DD/MM/YYYY", key=f"phys_date_{atleta_id}")
        st.markdown("**Velocidade e Agilidade**")
        v1, v2, v3, v4, v5 = st.columns(5)
        sprint_10m_s = v1.number_input("Sprint 10m (s)", min_value=0.0, value=float(latest_snapshot.get("sprint_10m_s", 0.0) or 0.0), step=0.01, key=f"phys_s10_{atleta_id}")
        sprint_20m_s = v2.number_input("Sprint 20m (s)", min_value=0.0, value=float(latest_snapshot.get("sprint_20m_s", 0.0) or 0.0), step=0.01, key=f"phys_s20_{atleta_id}")
        teste_505_esq_s = v3.number_input("505 esq (s)", min_value=0.0, value=float(latest_snapshot.get("teste_505_esq_s", 0.0) or 0.0), step=0.01, key=f"phys_505e_{atleta_id}")
        teste_505_dir_s = v4.number_input("505 dir (s)", min_value=0.0, value=float(latest_snapshot.get("teste_505_dir_s", 0.0) or 0.0), step=0.01, key=f"phys_505d_{atleta_id}")
        v5.empty()

        st.markdown("**Saltos Simples**")
        s1, s2, s3, s4, s5 = st.columns(5)
        sj_altura_cm = s1.number_input("SJ altura (cm)", min_value=0.0, value=float(latest_snapshot.get("sj_altura_cm", 0.0) or 0.0), step=0.1, key=f"phys_sj_{atleta_id}")
        cmj_altura_cm = s2.number_input("CMJ altura (cm)", min_value=0.0, value=float(latest_snapshot.get("cmj_altura_cm", 0.0) or 0.0), step=0.1, key=f"phys_cmj_{atleta_id}")
        dj_caixa_m = s3.number_input("DJ caixa (m)", min_value=0.0, value=float(latest_snapshot.get("dj_caixa_m", 0.0) or 0.0), step=0.01, key=f"phys_djbox_{atleta_id}")
        dj_altura_cm = s4.number_input("DJ altura (cm)", min_value=0.0, value=float(latest_snapshot.get("dj_altura_cm", 0.0) or 0.0), step=0.1, key=f"phys_djh_{atleta_id}")
        dj_rsi = s5.number_input("DJ RSI", min_value=0.0, value=float(latest_snapshot.get("dj_rsi", 0.0) or 0.0), step=0.01, key=f"phys_djrsi_{atleta_id}")
        s6, s7, s8, s9, s10 = st.columns(5)
        dj_rsi_mod_mps = s6.number_input("DJ RSI mod (m/s)", min_value=0.0, value=float(latest_snapshot.get("dj_rsi_mod_mps", 0.0) or 0.0), step=0.01, key=f"phys_djrsi_mod_{atleta_id}")
        dj_contacto_ms = s7.number_input("DJ contacto (ms)", min_value=0.0, value=float(latest_snapshot.get("dj_contacto_ms", 0.0) or 0.0), step=0.1, key=f"phys_djcontact_{atleta_id}")
        s8.empty()
        s9.empty()
        s10.empty()

        st.markdown("**10 Jumps - Resumo**")
        j1, j2, j3, j4, j5 = st.columns(5)
        j10_rsi_10_5 = j1.number_input("10J RSI 10-5", min_value=0.0, value=float(latest_snapshot.get("j10_rsi_10_5", 0.0) or 0.0), step=0.01, key=f"phys_rsi_{atleta_id}")
        j10_cmj_cm = j2.number_input("10J CMJ (cm)", min_value=0.0, value=float(latest_snapshot.get("j10_cmj_cm", 0.0) or 0.0), step=0.1, key=f"phys_j10cmj_{atleta_id}")
        j10_media_saltos_cm = j3.number_input("10J media (cm)", min_value=0.0, value=float(latest_snapshot.get("j10_media_saltos_cm", 0.0) or 0.0), step=0.1, key=f"phys_j10avg_{atleta_id}")
        j10_maximo_cm = j4.number_input("10J maximo (cm)", min_value=0.0, value=float(latest_snapshot.get("j10_maximo_cm", 0.0) or 0.0), step=0.1, key=f"phys_j10max_{atleta_id}")
        j10_minimo_cm = j5.number_input("10J minimo (cm)", min_value=0.0, value=float(latest_snapshot.get("j10_minimo_cm", 0.0) or 0.0), step=0.1, key=f"phys_j10min_{atleta_id}")
        j6, j7, j8, j9, j10 = st.columns(5)
        indice_fadiga_10j_pct = j6.number_input("Indice fadiga 10J (%)", value=float(latest_snapshot.get("indice_fadiga_10j_pct", 0.0) or 0.0), step=0.1, key=f"phys_fad_{atleta_id}")
        j7.empty()
        j8.empty()
        j9.empty()
        j10.empty()

        jump_values = {}
        contact_values = {}
        st.markdown("**10 Jumps - Detalhe por Salto**")
        jump_cols_1 = st.columns(5)
        for idx in range(1, 6):
            with jump_cols_1[idx - 1]:
                jump_values[f"jump_{idx}_cm"] = st.number_input(
                    f"Jump {idx} (cm)",
                    min_value=0.0,
                    value=float(latest_snapshot.get(f"jump_{idx}_cm", 0.0) or 0.0),
                    step=0.1,
                    key=f"phys_jump_{idx}_{atleta_id}",
                )
                contact_values[f"contact_{idx}_ms"] = st.number_input(
                    f"Contact {idx} (ms)",
                    min_value=0.0,
                    value=float(latest_snapshot.get(f"contact_{idx}_ms", 0.0) or 0.0),
                    step=0.1,
                    key=f"phys_contact_{idx}_{atleta_id}",
                )
        jump_cols_2 = st.columns(5)
        for idx in range(6, 11):
            with jump_cols_2[idx - 6]:
                jump_values[f"jump_{idx}_cm"] = st.number_input(
                    f"Jump {idx} (cm)",
                    min_value=0.0,
                    value=float(latest_snapshot.get(f"jump_{idx}_cm", 0.0) or 0.0),
                    step=0.1,
                    key=f"phys_jump_{idx}_{atleta_id}",
                )
                contact_values[f"contact_{idx}_ms"] = st.number_input(
                    f"Contact {idx} (ms)",
                    min_value=0.0,
                    value=float(latest_snapshot.get(f"contact_{idx}_ms", 0.0) or 0.0),
                    step=0.1,
                    key=f"phys_contact_{idx}_{atleta_id}",
                )
        submitted = st.form_submit_button("Guardar registo fisico", type="primary")
    if submitted:
        updates = {
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
            "j10_rsi_10_5": j10_rsi_10_5 if j10_rsi_10_5 > 0 else None,
            "j10_cmj_cm": j10_cmj_cm if j10_cmj_cm > 0 else None,
            "j10_media_saltos_cm": j10_media_saltos_cm if j10_media_saltos_cm > 0 else None,
            "j10_maximo_cm": j10_maximo_cm if j10_maximo_cm > 0 else None,
            "j10_minimo_cm": j10_minimo_cm if j10_minimo_cm > 0 else None,
            "indice_fadiga_10j_pct": indice_fadiga_10j_pct if indice_fadiga_10j_pct != 0 else None,
        }
        updates.update({key: (value if value > 0 else None) for key, value in jump_values.items()})
        updates.update({key: (value if value > 0 else None) for key, value in contact_values.items()})
        new_record = _build_physical_record(physical_df, atleta_id, data_avaliacao, updates)
        append_records("physical", new_record, source_type="manual_physical_tests")
        st.session_state[panel_key] = False
        st.success("Registo fisico guardado com sucesso.")
        st.rerun()


def _render_inline_technical_insert(athlete_row: pd.Series, technical_df: pd.DataFrame, include_psychological: bool = False) -> None:
    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    mode_key = "psych" if include_psychological else "tech"
    panel_key = f"open_insert_{mode_key}_{atleta_id}"
    button_key = f"insert_{mode_key}_button_{atleta_id}"
    if st.button("Inserir", key=button_key):
        _toggle_state(panel_key)
        st.rerun()
    if not st.session_state.get(panel_key, False):
        return

    latest_snapshot = _latest_technical_snapshot(technical_df, atleta_id)
    is_goalkeeper = _clean_text_value(athlete_row.get("posicao")) == "GR"
    with st.form(f"insert_{mode_key}_form_{atleta_id}"):
        data_avaliacao = st.date_input("Data avaliacao", value=date.today(), format="DD/MM/YYYY", key=f"{mode_key}_date_{atleta_id}")
        updates: dict = {}
        if is_goalkeeper:
            c1, c2, c3, c4 = st.columns(4)
            updates["reposicao_pe_score"] = c1.number_input("Reposição com o pé", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("reposicao_pe_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_rep_pe_{atleta_id}")
            updates["reposicao_mao_score"] = c2.number_input("Reposição com a mão", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("reposicao_mao_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_rep_mao_{atleta_id}")
            updates["tomada_decisao_score"] = c3.number_input("Tomada de decisão", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("tomada_decisao_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_td_{atleta_id}")
            updates["comunicacao_score"] = c4.number_input("Comunicação", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("comunicacao_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_com_{atleta_id}")
            c5, c6, c7, c8, c9 = st.columns(5)
            updates["posicionamento_prontidao_score"] = c5.number_input("Posicionamento", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("posicionamento_prontidao_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_pos_{atleta_id}")
            updates["defesa_membros_inferiores_score"] = c6.number_input("Defesa MI", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("defesa_membros_inferiores_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_dmi_{atleta_id}")
            updates["defesa_membros_superiores_score"] = c7.number_input("Defesa MS", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("defesa_membros_superiores_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_dms_{atleta_id}")
            updates["defesa_6m_ocupa_espaco_score"] = c8.number_input("Defesa 6m", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("defesa_6m_ocupa_espaco_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_d6_{atleta_id}")
            updates["leitura_jogo_score"] = c9.number_input("Leitura de jogo", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("leitura_jogo_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_lj_{atleta_id}")
        else:
            c1, c2, c3 = st.columns(3)
            updates["um_x_um_ofensivo_score"] = c1.number_input("1x1 Ofensivo", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("um_x_um_ofensivo_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_11o_{atleta_id}")
            updates["um_x_um_defensivo_score"] = c2.number_input("1x1 Defensivo", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("um_x_um_defensivo_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_11d_{atleta_id}")
            updates["lateralidade_score"] = c3.number_input("Lateralidade", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("lateralidade_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_lat_{atleta_id}")
            c4, c5, c6 = st.columns(3)
            updates["imprevisibilidade_score"] = c4.number_input("Imprevisibilidade", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("imprevisibilidade_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_imp_{atleta_id}")
            updates["leitura_jogo_score"] = c5.number_input("Leitura de Jogo", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("leitura_jogo_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_lei_{atleta_id}")
            updates["dominio_espaco_score"] = c6.number_input("Dominio do Espaco", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("dominio_espaco_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_dom_{atleta_id}")
        if include_psychological:
            st.markdown("**Psicologico**")
            p1, p2, p3, p4 = st.columns(4)
            updates["espirito_equipa_score"] = p1.number_input("Espírito de equipa", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("espirito_equipa_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_eq_{atleta_id}")
            updates["controlo_emocional_score"] = p2.number_input("Controlo emocional", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("controlo_emocional_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_ce_{atleta_id}")
            updates["tenacidade_resiliencia_score"] = p3.number_input("Tenacidade / Resiliencia", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("tenacidade_resiliencia_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_tr_{atleta_id}")
            updates["atencao_concentracao_score"] = p4.number_input("Atencao / concentracao", min_value=0.0, max_value=7.0, value=float(latest_snapshot.get("atencao_concentracao_score", 0.0) or 0.0), step=0.1, key=f"{mode_key}_ac_{atleta_id}")
        submitted = st.form_submit_button("Guardar registo", type="primary")
    if submitted:
        cleaned_updates = {key: (value if value > 0 else None) for key, value in updates.items()}
        record_df = _build_technical_record(technical_df, atleta_id, data_avaliacao, cleaned_updates)
        append_records("technical", record_df, source_type="manual_technical")
        st.session_state[panel_key] = False
        st.success("Registo guardado com sucesso.")
        st.rerun()


def _render_inline_pathway_insert(athlete_row: pd.Series, pathway_df: pd.DataFrame) -> None:
    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    panel_key = f"open_insert_pathway_{atleta_id}"
    if st.button("Inserir", key=f"insert_pathway_button_{atleta_id}"):
        _toggle_state(panel_key)
        st.rerun()

    if not st.session_state.get(panel_key, False):
        return

    latest_snapshot = _latest_pathway_snapshot(pathway_df, atleta_id)
    default_year = _clean_integer(latest_snapshot.get("ano")) or date.today().year
    default_selection = _clean_text_value(latest_snapshot.get("selecao")) or _clean_text_value(athlete_row.get("selecao_base_atual"))
    default_season = _clean_text_value(latest_snapshot.get("epoca")) or _clean_text_value(athlete_row.get("epoca_atual"))
    default_state = _clean_text_value(latest_snapshot.get("estado"))
    default_reference = _clean_date(latest_snapshot.get("data_referencia")) or date.today()
    default_notes = _clean_text_value(latest_snapshot.get("observacoes"))
    season_options = _season_options(default_season)

    with st.form(f"insert_pathway_form_{atleta_id}"):
        p1, p2, p3, p4 = st.columns(4)
        ano = p1.number_input("Ano", min_value=2000, max_value=2100, value=int(default_year), step=1, key=f"path_year_{atleta_id}")
        selecao = p2.selectbox(
            "Selecao base",
            options=SELECTION_OPTIONS,
            index=SELECTION_OPTIONS.index(default_selection) if default_selection in SELECTION_OPTIONS else 0,
            key=f"path_sel_{atleta_id}",
        )
        epoca = p3.selectbox(
            "Epoca",
            options=season_options,
            index=season_options.index(default_season) if default_season in season_options else 0,
            key=f"path_epoca_{atleta_id}",
        )
        estado = p4.selectbox(
            "Estado",
            options=STATE_OPTIONS,
            index=STATE_OPTIONS.index(default_state) if default_state in STATE_OPTIONS else 0,
            key=f"path_state_{atleta_id}",
        )
        data_referencia = st.date_input("Data de referencia", value=default_reference, format="DD/MM/YYYY", key=f"path_date_{atleta_id}")
        observacoes = st.text_area("Observacoes", value=default_notes, key=f"path_notes_{atleta_id}", height=90)
        submitted = st.form_submit_button("Guardar percurso", type="primary")

    if submitted:
        if not _clean_text_value(estado):
            st.error("O campo Estado e obrigatorio.")
            return
        if bool(_clean_text_value(selecao)) != bool(_clean_text_value(epoca)):
            st.error("Selecao base e Epoca devem ser preenchidas em conjunto.")
            return
        append_athlete_pathway_entry(
            {
                "atleta_id": atleta_id,
                "ano": int(ano),
                "selecao": _clean_text_value(selecao),
                "epoca": _clean_text_value(epoca),
                "escalao": _clean_text_value(athlete_row.get("escalao")),
                "estado": _clean_text_value(estado),
                "data_referencia": _clean_date(data_referencia),
                "observacoes": _clean_text_value(observacoes),
            }
        )
        st.session_state[panel_key] = False
        st.success("Percurso guardado com sucesso.")
        st.rerun()


def _render_pathway_history_actions(athlete_row: pd.Series, pathway_history: pd.DataFrame, selection) -> None:
    selected_row = _selected_table_row(pathway_history, selection)
    if selected_row is None:
        st.caption("Seleciona uma linha do percurso para editar ou eliminar.")
        return

    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    entry_id = _clean_text_value(selected_row.get("entry_id"))

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Editar registo", key=f"edit_pathway_row_{atleta_id}_{entry_id}"):
        _toggle_state(f"open_pathway_edit_{atleta_id}_{entry_id}")
        st.rerun()
    if action_col2.button("Eliminar registo", key=f"delete_pathway_row_{atleta_id}_{entry_id}"):
        delete_athlete_pathway_entry(entry_id)
        st.success("Registo de percurso eliminado com sucesso.")
        st.rerun()

    if not st.session_state.get(f"open_pathway_edit_{atleta_id}_{entry_id}", False):
        return

    with st.form(f"edit_pathway_form_{entry_id}"):
        current_selection = _clean_text_value(selected_row.get("selecao_base"))
        current_season = _clean_text_value(selected_row.get("epoca"))
        current_state = _clean_text_value(selected_row.get("estado"))
        season_options = _season_options(current_season)
        p1, p2, p3, p4 = st.columns(4)
        edit_year = p1.number_input("Ano", min_value=2000, max_value=2100, value=int(_clean_integer(selected_row.get("ano")) or date.today().year), step=1, key=f"edit_path_year_{entry_id}")
        edit_selection = p2.selectbox(
            "Selecao base",
            options=SELECTION_OPTIONS,
            index=SELECTION_OPTIONS.index(current_selection) if current_selection in SELECTION_OPTIONS else 0,
            key=f"edit_path_sel_{entry_id}",
        )
        edit_season = p3.selectbox(
            "Epoca",
            options=season_options,
            index=season_options.index(current_season) if current_season in season_options else 0,
            key=f"edit_path_epoca_{entry_id}",
        )
        edit_state = p4.selectbox(
            "Estado",
            options=STATE_OPTIONS,
            index=STATE_OPTIONS.index(current_state) if current_state in STATE_OPTIONS else 0,
            key=f"edit_path_state_{entry_id}",
        )
        edit_reference = st.date_input("Data de referencia", value=_clean_date(selected_row.get("data_referencia")) or date.today(), format="DD/MM/YYYY", key=f"edit_path_date_{entry_id}")
        edit_notes = st.text_area("Observacoes", value=_clean_text_value(selected_row.get("observacoes")), key=f"edit_path_notes_{entry_id}", height=90)
        save_edit = st.form_submit_button("Guardar alteracoes", type="primary")

    if save_edit:
        if not _clean_text_value(edit_state):
            st.error("O campo Estado e obrigatorio.")
            return
        if bool(_clean_text_value(edit_selection)) != bool(_clean_text_value(edit_season)):
            st.error("Selecao base e Epoca devem ser preenchidas em conjunto.")
            return
        update_athlete_pathway_entry(
            entry_id,
            {
                "ano": int(edit_year),
                "selecao": _clean_text_value(edit_selection),
                "epoca": _clean_text_value(edit_season),
                "escalao": _clean_text_value(athlete_row.get("escalao")),
                "estado": _clean_text_value(edit_state),
                "data_referencia": _clean_date(edit_reference),
                "observacoes": _clean_text_value(edit_notes),
            },
        )
        st.session_state[f"open_pathway_edit_{atleta_id}_{entry_id}"] = False
        st.success("Registo de percurso atualizado com sucesso.")
        st.rerun()


def _render_admin_history_actions(athlete_row: pd.Series, admin_history: pd.DataFrame, selection) -> None:
    selected_row = _selected_table_row(admin_history, selection)
    if selected_row is None:
        st.caption("Seleciona uma linha do historico da ficha para editar ou eliminar.")
        return

    if _clean_text_value(selected_row.get("record_kind")) == "pathway":
        _render_pathway_history_actions(athlete_row, admin_history, selection)
    else:
        st.caption("Os eventos de criacao e atualizacao da ficha ficam apenas para consulta.")


def _render_unified_history_actions(
    athlete_row: pd.Series,
    athletes_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    athlete_history: pd.DataFrame,
    selection,
) -> None:
    selected_row = _selected_table_row(athlete_history, selection)
    if selected_row is None:
        st.caption("Seleciona uma linha do historico para editar ou eliminar.")
        return

    record_kind = _clean_text_value(selected_row.get("record_kind"))
    if record_kind in {"physical", "technical"}:
        _render_inline_history_actions(athlete_row, athletes_df, physical_df, technical_df, athlete_history, selection)
    else:
        _render_admin_history_actions(athlete_row, athlete_history, selection)


def _render_inline_history_actions(athlete_row: pd.Series, athletes_df: pd.DataFrame, physical_df: pd.DataFrame, technical_df: pd.DataFrame, athlete_history: pd.DataFrame, selection) -> None:
    selected_history_row = _selected_table_row(athlete_history, selection)
    if selected_history_row is None:
        st.caption("Seleciona uma linha do historico para editar ou eliminar.")
        return

    atleta_id = _clean_text_value(athlete_row.get("atleta_id"))
    record_kind = _clean_text_value(selected_history_row.get("record_kind"))
    record_id = _clean_text_value(selected_history_row.get("record_id"))
    source_type = _clean_text_value(selected_history_row.get("source_type"))

    action_col1, action_col2 = st.columns(2)
    if action_col1.button("Editar registo", key=f"edit_history_row_{atleta_id}_{record_id}"):
        _toggle_state(f"open_history_edit_{atleta_id}_{record_id}")
        st.rerun()
    if action_col2.button("Eliminar registo", key=f"delete_history_row_{atleta_id}_{record_id}"):
        if record_kind in {"physical", "technical"}:
            delete_record(record_kind, record_id)
            st.success("Registo eliminado com sucesso.")
            st.rerun()
        st.info("Os eventos da ficha da atleta nao podem ser eliminados aqui.")

    if not st.session_state.get(f"open_history_edit_{atleta_id}_{record_id}", False):
        return
    if record_kind == "physical":
        edit_row = _record_row_by_id(physical_df, record_id)
        if edit_row is None:
            return
        if source_type == "manual_anthropometry":
            with st.form(f"edit_history_anth_{record_id}"):
                e1, e2 = st.columns(2)
                edit_data_avaliacao = e1.date_input("Data avaliacao", value=_clean_date(edit_row.get("data_avaliacao")), format="DD/MM/YYYY", key=f"hist_anth_date_{record_id}")
                edit_peso = e2.number_input("Peso corporal (kg)", min_value=0.0, value=float(edit_row.get("peso_kg", 0.0) or 0.0), step=0.1, key=f"hist_anth_weight_{record_id}")
                e3, e4, e5, e6 = st.columns(4)
                edit_altura = e3.number_input("Altura (cm)", min_value=0.0, value=float(edit_row.get("altura_cm", 0.0) or 0.0), step=0.1, key=f"hist_anth_height_{record_id}")
                edit_envergadura = e4.number_input("Envergadura (cm)", min_value=0.0, value=float(edit_row.get("envergadura_cm", 0.0) or 0.0), step=0.1, key=f"hist_anth_span_{record_id}")
                edit_perna = e5.number_input("Comprimento Perna (cm)", min_value=0.0, value=float(edit_row.get("comprimento_perna_cm", 0.0) or 0.0), step=0.1, key=f"hist_anth_leg_{record_id}")
                edit_sentada = e6.number_input("Altura sentada (cm)", min_value=0.0, value=float(edit_row.get("altura_sentada_cm", 0.0) or 0.0), step=0.1, key=f"hist_anth_sit_{record_id}")
                edit_offset = _calculate_maturity_offset(athlete_row.get("genero"), athlete_row.get("data_nascimento"), edit_data_avaliacao, edit_peso, edit_altura, edit_sentada)
                edit_status = _classify_maturity_offset(edit_offset)
                save_edit = st.form_submit_button("Guardar alteracoes", type="primary")
            if save_edit:
                update_record("physical", record_id, {
                    "data_avaliacao": _clean_date(edit_data_avaliacao),
                    "peso_kg": edit_peso if edit_peso > 0 else None,
                    "altura_cm": edit_altura if edit_altura > 0 else None,
                    "envergadura_cm": edit_envergadura if edit_envergadura > 0 else None,
                    "comprimento_perna_cm": edit_perna if edit_perna > 0 else None,
                    "altura_sentada_cm": edit_sentada if edit_sentada > 0 else None,
                    "salto_maturacional": edit_offset,
                    "estado_maturacional": edit_status,
                })
                st.session_state[f"open_history_edit_{atleta_id}_{record_id}"] = False
                st.success("Registo atualizado com sucesso.")
                st.rerun()
        else:
            with st.form(f"edit_history_phys_{record_id}"):
                e1, e2, e3, e4 = st.columns(4)
                edit_data_avaliacao = e1.date_input("Data avaliacao", value=_clean_date(edit_row.get("data_avaliacao")), format="DD/MM/YYYY", key=f"hist_phys_date_{record_id}")
                edit_sprint10 = e2.number_input("Sprint 10m (s)", min_value=0.0, value=float(edit_row.get("sprint_10m_s", 0.0) or 0.0), step=0.01, key=f"hist_phys_s10_{record_id}")
                edit_sprint20 = e3.number_input("Sprint 20m (s)", min_value=0.0, value=float(edit_row.get("sprint_20m_s", 0.0) or 0.0), step=0.01, key=f"hist_phys_s20_{record_id}")
                edit_cmj = e4.number_input("CMJ altura (cm)", min_value=0.0, value=float(edit_row.get("cmj_altura_cm", 0.0) or 0.0), step=0.1, key=f"hist_phys_cmj_{record_id}")
                e5, e6, e7 = st.columns(3)
                edit_sj = e5.number_input("SJ altura (cm)", min_value=0.0, value=float(edit_row.get("sj_altura_cm", 0.0) or 0.0), step=0.1, key=f"hist_phys_sj_{record_id}")
                edit_rsi = e6.number_input("10J RSI 10-5", min_value=0.0, value=float(edit_row.get("j10_rsi_10_5", 0.0) or 0.0), step=0.01, key=f"hist_phys_rsi_{record_id}")
                edit_fadiga = e7.number_input("Indice fadiga 10J (%)", value=float(edit_row.get("indice_fadiga_10j_pct", 0.0) or 0.0), step=0.1, key=f"hist_phys_fad_{record_id}")
                save_edit = st.form_submit_button("Guardar alteracoes", type="primary")
            if save_edit:
                update_record("physical", record_id, {
                    "data_avaliacao": _clean_date(edit_data_avaliacao),
                    "sprint_10m_s": edit_sprint10 if edit_sprint10 > 0 else None,
                    "sprint_20m_s": edit_sprint20 if edit_sprint20 > 0 else None,
                    "cmj_altura_cm": edit_cmj if edit_cmj > 0 else None,
                    "sj_altura_cm": edit_sj if edit_sj > 0 else None,
                    "j10_rsi_10_5": edit_rsi if edit_rsi > 0 else None,
                    "indice_fadiga_10j_pct": edit_fadiga if edit_fadiga != 0 else None,
                })
                st.session_state[f"open_history_edit_{atleta_id}_{record_id}"] = False
                st.success("Registo atualizado com sucesso.")
                st.rerun()
    elif record_kind == "technical":
        edit_row = _record_row_by_id(technical_df, record_id)
        if edit_row is None:
            return
        is_goalkeeper = _clean_text_value(athlete_row.get("posicao")) == "GR"
        with st.form(f"edit_history_tech_{record_id}"):
            edit_data_avaliacao = st.date_input("Data avaliacao", value=_clean_date(edit_row.get("data_avaliacao")), format="DD/MM/YYYY", key=f"hist_tech_date_{record_id}")
            updates = {}
            if is_goalkeeper:
                c1, c2, c3, c4 = st.columns(4)
                updates["reposicao_pe_score"] = c1.number_input("Reposição com o pé", min_value=0.0, max_value=7.0, value=float(edit_row.get("reposicao_pe_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_rep_pe_{record_id}")
                updates["reposicao_mao_score"] = c2.number_input("Reposição com a mão", min_value=0.0, max_value=7.0, value=float(edit_row.get("reposicao_mao_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_rep_mao_{record_id}")
                updates["tomada_decisao_score"] = c3.number_input("Tomada de decisão", min_value=0.0, max_value=7.0, value=float(edit_row.get("tomada_decisao_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_td_{record_id}")
                updates["comunicacao_score"] = c4.number_input("Comunicação", min_value=0.0, max_value=7.0, value=float(edit_row.get("comunicacao_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_com_{record_id}")
                c5, c6, c7, c8, c9 = st.columns(5)
                updates["posicionamento_prontidao_score"] = c5.number_input("Posicionamento", min_value=0.0, max_value=7.0, value=float(edit_row.get("posicionamento_prontidao_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_pos_{record_id}")
                updates["defesa_membros_inferiores_score"] = c6.number_input("Defesa MI", min_value=0.0, max_value=7.0, value=float(edit_row.get("defesa_membros_inferiores_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_dmi_{record_id}")
                updates["defesa_membros_superiores_score"] = c7.number_input("Defesa MS", min_value=0.0, max_value=7.0, value=float(edit_row.get("defesa_membros_superiores_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_dms_{record_id}")
                updates["defesa_6m_ocupa_espaco_score"] = c8.number_input("Defesa 6m", min_value=0.0, max_value=7.0, value=float(edit_row.get("defesa_6m_ocupa_espaco_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_d6_{record_id}")
                updates["leitura_jogo_score"] = c9.number_input("Leitura de Jogo", min_value=0.0, max_value=7.0, value=float(edit_row.get("leitura_jogo_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_lj_{record_id}")
            else:
                c1, c2, c3 = st.columns(3)
                updates["um_x_um_ofensivo_score"] = c1.number_input("1x1 Ofensivo", min_value=0.0, max_value=7.0, value=float(edit_row.get("um_x_um_ofensivo_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_11o_{record_id}")
                updates["um_x_um_defensivo_score"] = c2.number_input("1x1 Defensivo", min_value=0.0, max_value=7.0, value=float(edit_row.get("um_x_um_defensivo_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_11d_{record_id}")
                updates["lateralidade_score"] = c3.number_input("Lateralidade", min_value=0.0, max_value=7.0, value=float(edit_row.get("lateralidade_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_lat_{record_id}")
                c4, c5, c6 = st.columns(3)
                updates["imprevisibilidade_score"] = c4.number_input("Imprevisibilidade", min_value=0.0, max_value=7.0, value=float(edit_row.get("imprevisibilidade_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_imp_{record_id}")
                updates["leitura_jogo_score"] = c5.number_input("Leitura de Jogo", min_value=0.0, max_value=7.0, value=float(edit_row.get("leitura_jogo_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_lei_{record_id}")
                updates["dominio_espaco_score"] = c6.number_input("Dominio do Espaco", min_value=0.0, max_value=7.0, value=float(edit_row.get("dominio_espaco_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_dom_{record_id}")
            st.markdown("**Psicologico**")
            p1, p2, p3, p4 = st.columns(4)
            updates["espirito_equipa_score"] = p1.number_input("Espírito de equipa", min_value=0.0, max_value=7.0, value=float(edit_row.get("espirito_equipa_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_eq_{record_id}")
            updates["controlo_emocional_score"] = p2.number_input("Controlo emocional", min_value=0.0, max_value=7.0, value=float(edit_row.get("controlo_emocional_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_ce_{record_id}")
            updates["tenacidade_resiliencia_score"] = p3.number_input("Tenacidade / Resiliencia", min_value=0.0, max_value=7.0, value=float(edit_row.get("tenacidade_resiliencia_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_tr_{record_id}")
            updates["atencao_concentracao_score"] = p4.number_input("Atencao / concentracao", min_value=0.0, max_value=7.0, value=float(edit_row.get("atencao_concentracao_score", 0.0) or 0.0), step=0.1, key=f"hist_tech_ac_{record_id}")
            save_edit = st.form_submit_button("Guardar alteracoes", type="primary")
        if save_edit:
            cleaned = {key: (value if value > 0 else None) for key, value in updates.items()}
            cleaned["data_avaliacao"] = _clean_date(edit_data_avaliacao)
            update_record("technical", record_id, cleaned)
            st.session_state[f"open_history_edit_{atleta_id}_{record_id}"] = False
            st.success("Registo atualizado com sucesso.")
            st.rerun()


def _render_athlete_registry(
    athletes_df: pd.DataFrame,
    athlete_history_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    technical_df: pd.DataFrame,
    pathway_df: pd.DataFrame,
) -> None:
    registry_df = _athletes_with_latest(athletes_df, physical_df, technical_df, pathway_df)
    if registry_df.empty:
        st.info("Ainda nao existem atletas registados.")
        return

    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    search_term = filter_col1.text_input("Pesquisar atleta", key="futsal_registry_search")
    gender_options = ["Todos"] + sorted([value for value in registry_df["genero"].dropna().astype(str).unique().tolist() if value])
    selected_gender = filter_col2.selectbox("Genero", options=gender_options, key="futsal_registry_genero")
    season_options = ["Todos"] + sorted([value for value in registry_df["epoca_atual"].dropna().astype(str).unique().tolist() if value])
    selected_season = filter_col3.selectbox("Epoca", options=season_options, key="futsal_registry_epoca")
    selection_options = ["Todos"] + sorted([value for value in registry_df["selecao_base_atual"].dropna().astype(str).unique().tolist() if value])
    selected_selection = filter_col4.selectbox("Selecao", options=selection_options, key="futsal_registry_selecao")

    view_df = registry_df.copy()
    if search_term:
        query = search_term.strip().lower()
        view_df = view_df[
            view_df["nome"].astype(str).str.lower().str.contains(query, na=False)
            | view_df["atleta_id"].astype(str).str.lower().str.contains(query, na=False)
        ].copy()
    if selected_gender != "Todos":
        view_df = view_df[view_df["genero"].astype(str) == selected_gender].copy()
    if selected_season != "Todos":
        view_df = view_df[view_df["epoca_atual"].astype(str) == selected_season].copy()
    if selected_selection != "Todos":
        view_df = view_df[view_df["selecao_base_atual"].astype(str) == selected_selection].copy()

    if view_df.empty:
        st.info("Sem atletas para este filtro.")
        return

    ordered_view_df = view_df.sort_values(["ativo", "nome"], ascending=[False, True], na_position="last").reset_index(drop=True)
    athlete_ids = ordered_view_df["atleta_id"].astype(str).tolist()

    def _athlete_option_label(atleta_id: str) -> str:
        row = ordered_view_df[ordered_view_df["atleta_id"].astype(str) == str(atleta_id)].head(1)
        if row.empty:
            return str(atleta_id)
        athlete = row.iloc[0]
        name = _clean_text_value(athlete.get("nome")) or str(atleta_id)
        pos = _clean_text_value(athlete.get("posicao"))
        sel = _clean_text_value(athlete.get("selecao_atual"))
        return " | ".join([part for part in [name, pos, sel] if part])

    current_selected_id = st.session_state.get("selected_futsal_athlete_id", athlete_ids[0] if athlete_ids else "")
    if current_selected_id not in athlete_ids:
        current_selected_id = athlete_ids[0] if athlete_ids else ""

    list_df = ordered_view_df.copy()
    list_df["Atleta"] = list_df["nome"].map(_clean_text_value)
    list_df["ID"] = list_df["atleta_id"].map(_clean_text_value)
    list_df["Escalao"] = list_df["escalao_atual"].map(_clean_text_value)
    list_df["Selecao"] = list_df["selecao_atual"].map(_clean_text_value)
    list_df["Estado"] = list_df["estado_atual"].map(_clean_text_value)
    list_df["Posicao"] = list_df["posicao"].map(_clean_text_value)
    list_df["Ativa"] = list_df["ativo"].map(lambda value: "Sim" if bool(value) else "Nao")

    st.markdown("**Lista de atletas**")
    list_selection = st.dataframe(
        list_df[["Atleta", "ID", "Escalao", "Selecao", "Estado", "Posicao", "Ativa"]],
        use_container_width=True,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="futsal_athletes_list_table",
    )

    selected_rows = list_selection.selection.rows if list_selection and list_selection.selection else []
    if selected_rows and 0 <= selected_rows[0] < len(ordered_view_df):
        selected_athlete_id = ordered_view_df.iloc[selected_rows[0]]["atleta_id"]
        st.session_state["selected_futsal_athlete_id"] = str(selected_athlete_id)
    else:
        selected_athlete_id = current_selected_id

    st.caption(f"Atletas filtradas: {len(ordered_view_df)}")

    selected_view_df = ordered_view_df[ordered_view_df["atleta_id"].astype(str) == str(selected_athlete_id)].copy()

    for _, row in selected_view_df.iterrows():
        label = _clean_text_value(row.get("nome")) or _clean_text_value(row.get("atleta_id"))
        with st.expander(label, expanded=True):
            is_editing_athlete = st.session_state.get("open_athlete_editor_id", "") == _clean_text_value(row.get("atleta_id"))

            if is_editing_athlete:
                with st.form(f"edit_athlete_form_{row['atleta_id']}"):
                    head_col1, head_col2 = st.columns([0.8, 2.2], gap="large")
                    with head_col1:
                        photo_path = _clean_text_value(row.get("foto_path"))
                        if photo_path:
                            st.image(photo_path, width=140)
                        else:
                            st.caption("Sem foto registada.")
                        st.caption("Atualizar foto")
                        nova_foto_upload = st.file_uploader(
                            "Carregar nova foto",
                            type=["png", "jpg", "jpeg", "webp"],
                            key=f"edit_photo_upload_{row['atleta_id']}",
                            label_visibility="collapsed",
                        )
                        nova_foto = st.camera_input(
                            "Tirar foto",
                            key=f"edit_camera_{row['atleta_id']}",
                            label_visibility="collapsed",
                        )
                    with head_col2:
                        info_col1, info_col2 = st.columns(2, gap="large")
                        with info_col1:
                            st.text_input("ID", value=_clean_text_value(row.get("atleta_id")), disabled=True, key=f"edit_id_{row['atleta_id']}")
                            nome = st.text_input("Nome", value=_clean_text_value(row.get("nome")), key=f"edit_nome_{row['atleta_id']}")
                            data_nascimento = st.date_input(
                                "Data nascimento",
                                value=_clean_date(row.get("data_nascimento")),
                                min_value=ATHLETE_BIRTHDATE_MIN,
                                max_value=ATHLETE_BIRTHDATE_MAX,
                                format="DD/MM/YYYY",
                                key=f"edit_birth_{row['atleta_id']}",
                            )
                            idade_preview = _calculate_age(data_nascimento)
                            st.text_input(
                                "Idade",
                                value="" if idade_preview is None else str(idade_preview),
                                disabled=True,
                                key=f"edit_age_{row['atleta_id']}",
                            )
                        with info_col2:
                            genero = st.selectbox(
                                "Genero",
                                options=GENDER_OPTIONS,
                                index=GENDER_OPTIONS.index(_clean_text_value(row.get("genero"))) if _clean_text_value(row.get("genero")) in GENDER_OPTIONS else 0,
                                key=f"edit_genero_{row['atleta_id']}",
                            )
                            escalao_preview = _derive_escalao_from_age(idade_preview)
                            st.text_input("Escalao", value=escalao_preview, disabled=True, key=f"edit_scale_{row['atleta_id']}")
                            selecao = st.selectbox(
                                "Selecao base",
                                options=SELECTION_OPTIONS,
                                index=SELECTION_OPTIONS.index(_clean_text_value(row.get("selecao_base_atual"))) if _clean_text_value(row.get("selecao_base_atual")) in SELECTION_OPTIONS else 0,
                                key=f"edit_selecao_{row['atleta_id']}",
                            )
                            season_options = _season_options(_clean_text_value(row.get("epoca_atual")))
                            epoca = st.selectbox(
                                "Epoca",
                                options=season_options,
                                index=season_options.index(_clean_text_value(row.get("epoca_atual"))) if _clean_text_value(row.get("epoca_atual")) in season_options else 0,
                                key=f"edit_epoca_{row['atleta_id']}",
                            )
                            estado = st.selectbox(
                                "Estado",
                                options=STATE_OPTIONS,
                                index=STATE_OPTIONS.index(_clean_text_value(row.get("estado_atual"))) if _clean_text_value(row.get("estado_atual")) in STATE_OPTIONS else 0,
                                key=f"edit_estado_{row['atleta_id']}",
                            )
                            posicao = st.selectbox(
                                "Posicao",
                                options=POSITION_OPTIONS,
                                index=POSITION_OPTIONS.index(_clean_text_value(row.get("posicao"))) if _clean_text_value(row.get("posicao")) in POSITION_OPTIONS else 0,
                                key=f"edit_posicao_{row['atleta_id']}",
                            )
                            ativo = st.checkbox("Ativo", value=bool(row.get("ativo")), key=f"edit_ativo_{row['atleta_id']}")
                    save_col, cancel_col = st.columns([1, 1])
                    save_edit = save_col.form_submit_button("Guardar alteracoes", type="primary")
                    cancel_edit = cancel_col.form_submit_button("Cancelar")

                if cancel_edit:
                    _close_athlete_editor()
                    st.rerun()

                if save_edit:
                    if not _clean_text_value(nome):
                        st.error("O campo Nome e obrigatorio.")
                    elif bool(_clean_text_value(selecao)) != bool(_clean_text_value(epoca)):
                        st.error("Selecao base e Epoca devem ser preenchidas em conjunto.")
                    else:
                        photo_source = nova_foto_upload if nova_foto_upload is not None else nova_foto
                        updated_photo_path = _clean_text_value(row.get("foto_path"))
                        if photo_source is not None:
                            if updated_photo_path:
                                delete_photo(updated_photo_path)
                            updated_photo_path = save_photo(_clean_text_value(row.get("atleta_id")), photo_source)
                        upsert_athlete(
                            {
                                "atleta_id": _clean_text_value(row.get("atleta_id")),
                                "nome": _clean_text_value(nome),
                                "data_nascimento": _clean_date(data_nascimento),
                                "genero": _clean_text_value(genero),
                                "selecao": _clean_text_value(selecao),
                                "posicao": _clean_text_value(posicao),
                                "foto_path": updated_photo_path,
                                "ativo": bool(ativo),
                                "created_at": row.get("created_at"),
                            }
                        )
                        previous_selection = _clean_text_value(row.get("selecao_base_atual"))
                        previous_season = _clean_text_value(row.get("epoca_atual"))
                        previous_state = _clean_text_value(row.get("estado_atual"))
                        if (
                            _clean_text_value(selecao) != previous_selection
                            or _clean_text_value(epoca) != previous_season
                            or _clean_text_value(estado) != previous_state
                            or not previous_state
                        ) and (_clean_text_value(selecao) or _clean_text_value(estado)):
                            append_athlete_pathway_entry(
                                {
                                    "atleta_id": _clean_text_value(row.get("atleta_id")),
                                    "ano": date.today().year,
                                    "selecao": _clean_text_value(selecao),
                                    "epoca": _clean_text_value(epoca),
                                    "escalao": _clean_text_value(escalao_preview),
                                    "estado": _clean_text_value(estado),
                                    "data_referencia": date.today(),
                                    "observacoes": "",
                                }
                            )
                        _close_athlete_editor()
                        st.success("Atleta atualizada com sucesso.")
                        st.rerun()
            else:
                head_col1, head_col2 = st.columns([0.8, 2.2], gap="large")
                with head_col1:
                    photo_path = _clean_text_value(row.get("foto_path"))
                    if photo_path:
                        st.image(photo_path, width=140)
                    else:
                        st.caption("Sem foto registada.")
                with head_col2:
                    info_col1, info_col2 = st.columns(2, gap="large")
                    with info_col1:
                        st.markdown(f"**ID:** {_clean_text_value(row.get('atleta_id'))}")
                        st.markdown(f"**Nome:** {_clean_text_value(row.get('nome')) or '-'}")
                        st.markdown(f"**Data nascimento:** {_format_date(row.get('data_nascimento'))}")
                        st.markdown(f"**Idade:** {_calculate_age(row.get('data_nascimento')) or '-'}")
                        st.markdown(f"**Genero:** {_clean_text_value(row.get('genero')) or '-'}")
                    with info_col2:
                        st.markdown(f"**Escalao:** {_clean_text_value(row.get('escalao_atual')) or '-'}")
                        st.markdown(f"**Selecao:** {_clean_text_value(row.get('selecao_atual')) or '-'}")
                        st.markdown(f"**Posicao:** {_clean_text_value(row.get('posicao')) or '-'}")
                        st.markdown(f"**Estado:** {_clean_text_value(row.get('estado_atual')) or '-'}")

            _render_profile_score_strip(row)

            st.markdown(
                """
                <style>
                button[id*="open_edit_athlete_"] {
                    border-color: #2f6fed !important;
                    background: rgba(47, 111, 237, 0.12) !important;
                    color: #dbe7ff !important;
                }
                button[id*="delete_athlete_"] {
                    border-color: #d34f4f !important;
                    background: rgba(211, 79, 79, 0.14) !important;
                    color: #ffdede !important;
                }
                button[id*="toggle_athlete_"] {
                    border-color: #d2a43b !important;
                    background: rgba(210, 164, 59, 0.14) !important;
                    color: #fff1c7 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            atleta_id = _clean_text_value(row.get("atleta_id"))
            delete_confirm_key = f"confirm_delete_athlete_{atleta_id}"
            spacer_col, action_col1, action_col2, action_col3 = st.columns([3.4, 1.2, 1.2, 1.2], gap="small")
            if action_col1.button("Editar atleta", key=f"open_edit_athlete_{row['atleta_id']}", use_container_width=True):
                _open_athlete_editor(_clean_text_value(row.get("atleta_id")))
                st.rerun()
            if action_col2.button("Eliminar atleta e historico", key=f"delete_athlete_{row['atleta_id']}", use_container_width=True):
                st.session_state[delete_confirm_key] = True
                st.rerun()
            if action_col3.button("Inativar/Ativar", key=f"toggle_athlete_{row['atleta_id']}", use_container_width=True):
                current_row = athletes_df[athletes_df["atleta_id"].astype(str) == str(row["atleta_id"])].head(1)
                if not current_row.empty:
                    athlete_payload = current_row.iloc[0].to_dict()
                    athlete_payload["ativo"] = not bool(row.get("ativo"))
                    upsert_athlete(athlete_payload)
                st.success("Estado do atleta atualizado.")
                st.rerun()

            if st.session_state.get(delete_confirm_key, False):
                st.warning("Confirma a eliminação definitiva da atleta e de todo o histórico associado.")
                confirm_spacer, confirm_col1, confirm_col2 = st.columns([3.8, 1.2, 1.2], gap="small")
                if confirm_col1.button("Confirmar eliminacao", key=f"confirm_delete_btn_{row['atleta_id']}", use_container_width=True):
                    delete_photo(_clean_text_value(row.get("foto_path")))
                    delete_athlete(atleta_id)
                    _close_athlete_delete_confirm(atleta_id)
                    st.success("Atleta e historico eliminados com sucesso.")
                    st.rerun()
                if confirm_col2.button("Cancelar", key=f"cancel_delete_btn_{row['atleta_id']}", use_container_width=True):
                    _close_athlete_delete_confirm(atleta_id)
                    st.rerun()

            tab_anth, tab_phys, tab_tech, tab_psych, tab_history = st.tabs(
                ["Ficha Antropometrica", "Testes Fisicos", "Ficha Tecnico | Tatica", "Ficha Psicologica", "Historico"]
            )

            with tab_anth:
                _render_inline_anthropometry_insert(row, physical_df)
                st.caption(f"Data: {_format_date(row.get('fis_data_avaliacao'))}")
                anthropometry_metrics = [
                    {"label": "Peso", "row_key": "fis_peso_kg", "metric_key": "peso_kg", "suffix": " kg", "decimals": 1},
                    {"label": "Altura", "row_key": "fis_altura_cm", "metric_key": "altura_cm", "suffix": " cm", "decimals": 1},
                    {"label": "Envergadura", "row_key": "fis_envergadura_cm", "metric_key": "envergadura_cm", "suffix": " cm", "decimals": 1},
                    {"label": "Comp. perna", "row_key": "fis_comprimento_perna_cm", "metric_key": "comprimento_perna_cm", "suffix": " cm", "decimals": 1},
                    {"label": "Alt. sentada", "row_key": "fis_altura_sentada_cm", "metric_key": "altura_sentada_cm", "suffix": " cm", "decimals": 1},
                    {"label": "Maturacao", "row_key": "fis_salto_maturacional", "metric_key": "salto_maturacional", "suffix": "", "decimals": 2},
                ]
                _render_spider_map(
                    "Spider de Enquadramento | Antropometria",
                    _build_context_radar_axes(row, anthropometry_metrics, [["genero", "escalao_avaliacao"], ["genero"], []]),
                    "Mapa percentilico 0-100 do enquadramento da atleta face ao grupo comparavel.",
                    chart_key=f"radar_anth_{_clean_text_value(row.get('atleta_id'))}",
                )
                _render_absolute_metric_grid(
                    row,
                    "Resultados absolutos",
                    anthropometry_metrics,
                    scope_sets=[["genero", "escalao_avaliacao"], ["genero"], []],
                )
                st.markdown(f"**Estado maturacional:** {_clean_text_value(row.get('fis_estado_maturacional')) or '-'}")

            with tab_phys:
                _render_inline_physical_insert(row, physical_df)
                st.caption(f"Data: {_format_date(row.get('fis_data_avaliacao'))}")
                physical_metrics = [
                    {"label": "Sprint 10m", "row_key": "fis_sprint_10m_s", "metric_key": "sprint_10m_s", "suffix": " s", "decimals": 2},
                    {"label": "Sprint 20m", "row_key": "fis_sprint_20m_s", "metric_key": "sprint_20m_s", "suffix": " s", "decimals": 2},
                    {"label": "505 Esq", "row_key": "fis_teste_505_esq_s", "metric_key": "teste_505_esq_s", "suffix": " s", "decimals": 2},
                    {"label": "505 Dir", "row_key": "fis_teste_505_dir_s", "metric_key": "teste_505_dir_s", "suffix": " s", "decimals": 2},
                    {"label": "SJ", "row_key": "fis_sj_altura_cm", "metric_key": "sj_altura_cm", "suffix": " cm", "decimals": 1},
                    {"label": "CMJ", "row_key": "fis_cmj_altura_cm", "metric_key": "cmj_altura_cm", "suffix": " cm", "decimals": 1},
                    {"label": "DJ Altura", "row_key": "fis_dj_altura_cm", "metric_key": "dj_altura_cm", "suffix": " cm", "decimals": 1},
                    {"label": "DJ RSI", "row_key": "fis_dj_rsi", "metric_key": "dj_rsi", "suffix": "", "decimals": 2},
                    {"label": "10J RSI", "row_key": "fis_j10_rsi_10_5", "metric_key": "j10_rsi_10_5", "suffix": "", "decimals": 2},
                    {"label": "10J Media", "row_key": "fis_j10_media_saltos_cm", "metric_key": "j10_media_saltos_cm", "suffix": " cm", "decimals": 1},
                    {"label": "Fatiga", "row_key": "fis_indice_fadiga_10j_pct", "metric_key": "indice_fadiga_10j_pct", "suffix": "%", "decimals": 2},
                ]
                physical_absolute_groups = [
                    (
                        "Velocidade",
                        [
                            {"label": "Sprint 10m", "row_key": "fis_sprint_10m_s", "metric_key": "sprint_10m_s", "suffix": " s", "decimals": 2},
                            {"label": "Sprint 20m", "row_key": "fis_sprint_20m_s", "metric_key": "sprint_20m_s", "suffix": " s", "decimals": 2},
                        ],
                    ),
                    (
                        "Agilidade",
                        [
                            {"label": "505 Esq", "row_key": "fis_teste_505_esq_s", "metric_key": "teste_505_esq_s", "suffix": " s", "decimals": 2},
                            {"label": "505 Dir", "row_key": "fis_teste_505_dir_s", "metric_key": "teste_505_dir_s", "suffix": " s", "decimals": 2},
                        ],
                    ),
                    (
                        "Potencia",
                        [
                            {"label": "SJ", "row_key": "fis_sj_altura_cm", "metric_key": "sj_altura_cm", "suffix": " cm", "decimals": 1},
                            {"label": "CMJ", "row_key": "fis_cmj_altura_cm", "metric_key": "cmj_altura_cm", "suffix": " cm", "decimals": 1},
                            {"label": "DJ Altura", "row_key": "fis_dj_altura_cm", "metric_key": "dj_altura_cm", "suffix": " cm", "decimals": 1},
                        ],
                    ),
                    (
                        "Reatividade",
                        [
                            {"label": "DJ RSI", "row_key": "fis_dj_rsi", "metric_key": "dj_rsi", "suffix": "", "decimals": 2},
                            {"label": "10J RSI 10-5", "row_key": "fis_j10_rsi_10_5", "metric_key": "j10_rsi_10_5", "suffix": "", "decimals": 2},
                        ],
                    ),
                    (
                        "Resistencia",
                        [
                            {"label": "10J Media", "row_key": "fis_j10_media_saltos_cm", "metric_key": "j10_media_saltos_cm", "suffix": " cm", "decimals": 1},
                            {"label": "Indice fadiga", "row_key": "fis_indice_fadiga_10j_pct", "metric_key": "indice_fadiga_10j_pct", "suffix": "%", "decimals": 2},
                        ],
                    ),
                ]
                physical_grouped_metrics = [
                    (
                        "Velocidade",
                        [
                            {"row_key": "fis_sprint_10m_s", "metric_key": "sprint_10m_s"},
                            {"row_key": "fis_sprint_20m_s", "metric_key": "sprint_20m_s"},
                        ],
                    ),
                    (
                        "Agilidade",
                        [
                            {"row_key": "fis_teste_505_esq_s", "metric_key": "teste_505_esq_s"},
                            {"row_key": "fis_teste_505_dir_s", "metric_key": "teste_505_dir_s"},
                        ],
                    ),
                    (
                        "Potencia",
                        [
                            {"row_key": "fis_sj_altura_cm", "metric_key": "sj_altura_cm"},
                            {"row_key": "fis_cmj_altura_cm", "metric_key": "cmj_altura_cm"},
                            {"row_key": "fis_dj_altura_cm", "metric_key": "dj_altura_cm"},
                        ],
                    ),
                    (
                        "Reatividade",
                        [
                            {"row_key": "fis_dj_rsi", "metric_key": "dj_rsi"},
                            {"row_key": "fis_j10_rsi_10_5", "metric_key": "j10_rsi_10_5"},
                        ],
                    ),
                    (
                        "Resistencia",
                        [
                            {"row_key": "fis_j10_media_saltos_cm", "metric_key": "j10_media_saltos_cm"},
                            {"row_key": "fis_indice_fadiga_10j_pct", "metric_key": "indice_fadiga_10j_pct"},
                        ],
                    ),
                ]
                _render_spider_map(
                    "Spider de Enquadramento | Fisico",
                    _build_grouped_context_radar_axes(row, physical_grouped_metrics, [["genero", "escalao_avaliacao"], ["genero"], []]),
                    "Mapa percentilico 0-100 por subcategoria fisica: velocidade, agilidade, potencia, reatividade e resistencia.",
                    chart_key=f"radar_phys_{_clean_text_value(row.get('atleta_id'))}",
                )
                _render_grouped_absolute_metrics(
                    row,
                    physical_absolute_groups,
                    scope_sets=[["genero", "escalao_avaliacao"], ["genero"], []],
                    columns_count=3,
                )

            with tab_tech:
                _render_inline_technical_insert(row, technical_df, include_psychological=False)
                st.caption(f"Data: {_format_date(row.get('tec_data_avaliacao'))}")
                if _clean_text_value(row.get("posicao")) == "GR":
                    technical_metrics = [
                        {"label": "Reposição pé", "row_key": "tec_reposicao_pe_score", "metric_key": "reposicao_pe_score", "suffix": "/7", "decimals": 1},
                        {"label": "Reposição mão", "row_key": "tec_reposicao_mao_score", "metric_key": "reposicao_mao_score", "suffix": "/7", "decimals": 1},
                        {"label": "Tomada decisão", "row_key": "tec_tomada_decisao_score", "metric_key": "tomada_decisao_score", "suffix": "/7", "decimals": 1},
                        {"label": "Comunicação", "row_key": "tec_comunicacao_score", "metric_key": "comunicacao_score", "suffix": "/7", "decimals": 1},
                        {"label": "Posicionamento", "row_key": "tec_posicionamento_prontidao_score", "metric_key": "posicionamento_prontidao_score", "suffix": "/7", "decimals": 1},
                        {"label": "Defesa MS", "row_key": "tec_defesa_membros_superiores_score", "metric_key": "defesa_membros_superiores_score", "suffix": "/7", "decimals": 1},
                        {"label": "Defesa MI", "row_key": "tec_defesa_membros_inferiores_score", "metric_key": "defesa_membros_inferiores_score", "suffix": "/7", "decimals": 1},
                        {"label": "Defesa 6m", "row_key": "tec_defesa_6m_ocupa_espaco_score", "metric_key": "defesa_6m_ocupa_espaco_score", "suffix": "/7", "decimals": 1},
                        {"label": "Leitura", "row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score", "suffix": "/7", "decimals": 1},
                    ]
                    technical_grouped_metrics = [
                        (
                            "Reposição",
                            [
                                {"row_key": "tec_reposicao_pe_score", "metric_key": "reposicao_pe_score"},
                                {"row_key": "tec_reposicao_mao_score", "metric_key": "reposicao_mao_score"},
                            ],
                        ),
                        (
                            "Decisão | Comunicação",
                            [
                                {"row_key": "tec_tomada_decisao_score", "metric_key": "tomada_decisao_score"},
                                {"row_key": "tec_comunicacao_score", "metric_key": "comunicacao_score"},
                            ],
                        ),
                        (
                            "Posicionamento",
                            [{"row_key": "tec_posicionamento_prontidao_score", "metric_key": "posicionamento_prontidao_score"}],
                        ),
                        (
                            "Defesa",
                            [
                                {"row_key": "tec_defesa_membros_superiores_score", "metric_key": "defesa_membros_superiores_score"},
                                {"row_key": "tec_defesa_membros_inferiores_score", "metric_key": "defesa_membros_inferiores_score"},
                                {"row_key": "tec_defesa_6m_ocupa_espaco_score", "metric_key": "defesa_6m_ocupa_espaco_score"},
                            ],
                        ),
                        (
                            "Leitura",
                            [{"row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score"}],
                        ),
                    ]
                    _render_spider_map(
                        "Spider de Enquadramento | Tecnico | Tatica GR",
                        _build_grouped_context_radar_axes(row, technical_grouped_metrics, [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []]),
                        "Mapa percentilico 0-100 por bloco funcional de GR: reposição, decisão | comunicação, posicionamento, defesa e leitura.",
                        chart_key=f"radar_tech_gr_{_clean_text_value(row.get('atleta_id'))}",
                    )
                    _render_absolute_metric_grid(
                        row,
                        "Enquadramento comparativo",
                        technical_metrics,
                        scope_sets=[["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
                    )
                else:
                    technical_metrics = [
                        {"label": "1x1 Ofensivo", "row_key": "tec_um_x_um_ofensivo_score", "metric_key": "um_x_um_ofensivo_score", "suffix": "/7", "decimals": 1},
                        {"label": "1x1 Defensivo", "row_key": "tec_um_x_um_defensivo_score", "metric_key": "um_x_um_defensivo_score", "suffix": "/7", "decimals": 1},
                        {"label": "Lateralidade", "row_key": "tec_lateralidade_score", "metric_key": "lateralidade_score", "suffix": "/7", "decimals": 1},
                        {"label": "Imprevisibilidade", "row_key": "tec_imprevisibilidade_score", "metric_key": "imprevisibilidade_score", "suffix": "/7", "decimals": 1},
                        {"label": "Leitura", "row_key": "tec_leitura_jogo_score", "metric_key": "leitura_jogo_score", "suffix": "/7", "decimals": 1},
                        {"label": "Dominio Espaco", "row_key": "tec_dominio_espaco_score", "metric_key": "dominio_espaco_score", "suffix": "/7", "decimals": 1},
                    ]
                    _render_spider_map(
                        "Spider de Enquadramento | Tecnico | Tatica",
                        _build_context_radar_axes(row, technical_metrics, [["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []]),
                        "Mapa percentilico 0-100 do enquadramento da atleta face ao grupo comparavel.",
                        chart_key=f"radar_tech_{_clean_text_value(row.get('atleta_id'))}",
                    )
                    _render_absolute_metric_grid(
                        row,
                        "Enquadramento comparativo",
                        technical_metrics,
                        scope_sets=[["genero", "selecao", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao", "posicao"], ["genero", "escalao_avaliacao"], []],
                    )

            with tab_psych:
                _render_inline_technical_insert(row, technical_df, include_psychological=True)
                st.caption(f"Data: {_format_date(row.get('tec_data_avaliacao'))}")
                psych_metrics = [
                    {"label": "Espirito Equipa", "row_key": "tec_espirito_equipa_score", "metric_key": "espirito_equipa_score", "suffix": "/7", "decimals": 1},
                    {"label": "Controlo Emocional", "row_key": "tec_controlo_emocional_score", "metric_key": "controlo_emocional_score", "suffix": "/7", "decimals": 1},
                    {"label": "Tenacidade", "row_key": "tec_tenacidade_resiliencia_score", "metric_key": "tenacidade_resiliencia_score", "suffix": "/7", "decimals": 1},
                    {"label": "Atencao", "row_key": "tec_atencao_concentracao_score", "metric_key": "atencao_concentracao_score", "suffix": "/7", "decimals": 1},
                ]
                _render_spider_map(
                    "Spider de Enquadramento | Psicologico",
                    _build_context_radar_axes(row, psych_metrics, [["genero", "selecao", "escalao_avaliacao"], ["genero", "escalao_avaliacao"], ["genero"], []]),
                    "Mapa percentilico 0-100 do enquadramento da atleta face ao grupo comparavel.",
                    chart_key=f"radar_psych_{_clean_text_value(row.get('atleta_id'))}",
                )
                _render_absolute_metric_grid(
                    row,
                    "Enquadramento comparativo",
                    psych_metrics,
                    scope_sets=[["genero", "selecao", "escalao_avaliacao"], ["genero", "escalao_avaliacao"], ["genero"], []],
                    columns_count=2,
                )

            with tab_history:
                athlete_history = _athlete_history_rows(row, athlete_history_df, physical_df, technical_df, pathway_df)
                if athlete_history.empty:
                    st.info("Sem historico disponivel.")
                else:
                    _render_inline_pathway_insert(row, pathway_df)
                    display_history = athlete_history.copy()
                    display_history["Data"] = display_history["data_avaliacao"].where(
                        display_history["data_avaliacao"].astype(str) != "",
                        display_history["momento"].astype(str).str.slice(0, 10),
                    )
                    display_history = display_history.rename(
                        columns={
                            "tipo": "Tipo",
                            "selecao": "Selecao",
                            "estado": "Estado",
                        }
                    )
                    selection = st.dataframe(
                        display_history[["Data", "Selecao", "Estado", "Tipo"]],
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="single-row",
                        key=f"history_table_{_clean_text_value(row.get('atleta_id'))}",
                    )
                    _render_unified_history_actions(row, athletes_df, physical_df, technical_df, athlete_history, selection)



def _render_create_athlete_form(athletes_df: pd.DataFrame) -> None:
    st.subheader("Criar atleta")
    with st.form("futsal_create_athlete_form", clear_on_submit=True):
        athlete_id = generate_athlete_id(athletes_df)
        st.caption(f"ID atribuido: {athlete_id}")
        top_left, top_right = st.columns([0.8, 2.2], gap="large")
        with top_left:
            st.markdown("**Foto do atleta**")
            st.caption("Adicionar por upload ou capturar tipo passe.")
            foto_upload = st.file_uploader(
                "Carregar foto",
                type=["png", "jpg", "jpeg", "webp"],
                key="futsal_master_photo_upload",
                label_visibility="collapsed",
            )
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

            line3_col1, line3_col2, line3_col3, line3_col4 = st.columns(4)
            escalao_preview = _derive_escalao_from_age(idade_preview)
            line3_col1.text_input("Escalao", value=escalao_preview, disabled=True)
            selecao = line3_col2.selectbox("Selecao base", options=SELECTION_OPTIONS)
            create_season_options = _season_options()
            epoca = line3_col3.selectbox("Epoca", options=create_season_options)
            posicao = line3_col4.selectbox("Posicao", options=POSITION_OPTIONS)
            estado = st.selectbox("Estado", options=STATE_OPTIONS, index=0)
        ativo = st.checkbox("Ativo", value=True)
        submitted = st.form_submit_button("Criar atleta", type="primary")

    if submitted:
        if not _clean_text_value(nome):
            st.error("O campo Nome e obrigatorio.")
            return
        if bool(_clean_text_value(selecao)) != bool(_clean_text_value(epoca)):
            st.error("Selecao base e Epoca devem ser preenchidas em conjunto.")
            return
        photo_source = foto_upload if foto_upload is not None else foto
        photo_path = ""
        if photo_source is not None:
            photo_path = save_photo(athlete_id, photo_source)
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
        if _clean_text_value(selecao) or _clean_text_value(estado):
            append_athlete_pathway_entry(
                {
                    "atleta_id": athlete_id,
                    "ano": date.today().year,
                    "selecao": _clean_text_value(selecao),
                    "epoca": _clean_text_value(epoca),
                    "escalao": _clean_text_value(escalao_preview),
                    "estado": _clean_text_value(estado),
                    "data_referencia": date.today(),
                    "observacoes": "",
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


def _render_anthropometry_tab(athletes_df: pd.DataFrame, physical_df: pd.DataFrame) -> None:
    st.subheader("Ficha Antropometrica")
    athlete_options = athletes_df["atleta_id"].astype(str).tolist() if not athletes_df.empty else []
    if not athlete_options:
        st.info("Cria primeiro atletas na ficha mestre.")
        return

    selected_id, selected_athlete = _render_record_athlete_header(
        athletes_df,
        athlete_options,
        "anthropometry_atleta_id",
    )
    latest_snapshot = _latest_physical_snapshot(physical_df, selected_id)

    with st.form("anthropometry_single_record_form", clear_on_submit=True):
        data_avaliacao = st.date_input("Data avaliacao", value=date.today(), format="DD/MM/YYYY")
        a1, a2, a3, a4, a5 = st.columns(5)
        peso_kg = a1.number_input("Peso corporal (kg)", min_value=0.0, value=float(latest_snapshot.get("peso_kg", 0.0) or 0.0), step=0.1)
        altura_cm = a2.number_input("Altura (cm)", min_value=0.0, value=float(latest_snapshot.get("altura_cm", 0.0) or 0.0), step=0.1)
        envergadura_cm = a3.number_input("Envergadura (cm)", min_value=0.0, value=float(latest_snapshot.get("envergadura_cm", 0.0) or 0.0), step=0.1)
        comprimento_perna_cm = a4.number_input("Comprimento Perna (cm)", min_value=0.0, value=float(latest_snapshot.get("comprimento_perna_cm", 0.0) or 0.0), step=0.1)
        altura_sentada_cm = a5.number_input("Altura sentada (cm)", min_value=0.0, value=float(latest_snapshot.get("altura_sentada_cm", 0.0) or 0.0), step=0.1)
        maturity_offset = _calculate_maturity_offset(
            selected_athlete.get("genero") if selected_athlete is not None else "",
            selected_athlete.get("data_nascimento") if selected_athlete is not None else None,
            data_avaliacao,
            peso_kg,
            altura_cm,
            altura_sentada_cm,
        )
        maturity_status = _classify_maturity_offset(maturity_offset)
        b1, b2, b3 = st.columns([1, 1, 2])
        b1.text_input(
            "Salto maturacional",
            value="" if maturity_offset is None else f"{maturity_offset:.2f}",
            disabled=True,
        )
        b2.text_input("Estado maturacional", value=maturity_status, disabled=True)
        b3.caption("Calculo automatico do maturity offset. Requer data nascimento, genero, peso, altura e altura sentada.")
        submitted = st.form_submit_button("Guardar ficha antropometrica", type="primary")

    if submitted:
        new_record = _build_physical_record(physical_df, selected_id, data_avaliacao, {
            "peso_kg": peso_kg if peso_kg > 0 else None,
            "altura_cm": altura_cm if altura_cm > 0 else None,
            "envergadura_cm": envergadura_cm if envergadura_cm > 0 else None,
            "comprimento_perna_cm": comprimento_perna_cm if comprimento_perna_cm > 0 else None,
            "altura_sentada_cm": altura_sentada_cm if altura_sentada_cm > 0 else None,
            "salto_maturacional": maturity_offset,
            "estado_maturacional": maturity_status,
        })
        append_records("physical", new_record, source_type="manual_anthropometry")
        st.success("Ficha antropometrica guardada com sucesso.")
        st.rerun()

    st.divider()
    st.markdown("**Historico antropometrico**")
    anthropometry_history_df = physical_df[physical_df["source_type"].astype(str).isin(["manual_anthropometry"])].copy() if not physical_df.empty else physical_df
    if anthropometry_history_df.empty:
        st.info("Sem historico antropometrico.")
    else:
        history_df = anthropometry_history_df.copy()
        history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce").dt.strftime("%d/%m/%Y")
        history_df["inserted_at"] = pd.to_datetime(history_df["inserted_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        selection_df = history_df[["record_id", "data_avaliacao", "peso_kg", "altura_cm", "envergadura_cm", "comprimento_perna_cm", "altura_sentada_cm", "salto_maturacional", "estado_maturacional"]].reset_index(drop=True)
        display_df = selection_df.drop(columns=["record_id"])
        selection = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_row = _selected_table_row(selection_df, selection)
        selected_record_id = None if selected_row is None else str(selected_row["record_id"])
        if not selected_record_id:
            st.caption("Seleciona uma linha do historico para editar ou eliminar.")
            return

        edit_state_key = "open_edit_anthropometry_record"
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Editar registo antropometrico", key="open_edit_anthropometry_btn"):
            _toggle_state(edit_state_key)
            st.rerun()
        if action_col2.button("Eliminar registo antropometrico", key="delete_anthropometry_btn"):
            delete_record("physical", selected_record_id)
            st.success("Registo antropometrico eliminado com sucesso.")
            st.rerun()
        edit_row = _record_row_by_id(anthropometry_history_df, selected_record_id)
        if st.session_state.get(edit_state_key, False) and edit_row is not None:
            with st.form("edit_anthropometry_record_form"):
                e1, e2 = st.columns(2)
                edit_data_avaliacao = e1.date_input("Data avaliacao", value=_clean_date(edit_row.get("data_avaliacao")), format="DD/MM/YYYY", key="edit_anth_date")
                edit_peso = e2.number_input("Peso corporal (kg)", min_value=0.0, value=float(edit_row.get("peso_kg", 0.0) or 0.0), step=0.1, key="edit_anth_weight")
                e3, e4, e5, e6 = st.columns(4)
                edit_altura = e3.number_input("Altura (cm)", min_value=0.0, value=float(edit_row.get("altura_cm", 0.0) or 0.0), step=0.1, key="edit_anth_height")
                edit_envergadura = e4.number_input("Envergadura (cm)", min_value=0.0, value=float(edit_row.get("envergadura_cm", 0.0) or 0.0), step=0.1, key="edit_anth_span")
                edit_perna = e5.number_input("Comprimento Perna (cm)", min_value=0.0, value=float(edit_row.get("comprimento_perna_cm", 0.0) or 0.0), step=0.1, key="edit_anth_leg")
                edit_sentada = e6.number_input("Altura sentada (cm)", min_value=0.0, value=float(edit_row.get("altura_sentada_cm", 0.0) or 0.0), step=0.1, key="edit_anth_sit")
                athlete_row = _selected_athlete_row(athletes_df, _clean_text_value(edit_row.get("atleta_id")))
                edit_offset = _calculate_maturity_offset(
                    athlete_row.get("genero") if athlete_row is not None else "",
                    athlete_row.get("data_nascimento") if athlete_row is not None else None,
                    edit_data_avaliacao,
                    edit_peso,
                    edit_altura,
                    edit_sentada,
                )
                edit_status = _classify_maturity_offset(edit_offset)
                e7, e8 = st.columns(2)
                e7.text_input("Salto maturacional", value="" if edit_offset is None else f"{edit_offset:.2f}", disabled=True, key="edit_anth_offset")
                e8.text_input("Estado maturacional", value=edit_status, disabled=True, key="edit_anth_status")
                save_edit = st.form_submit_button("Guardar alteracoes do registo", type="primary")
            if save_edit:
                update_record(
                    "physical",
                    selected_record_id,
                    {
                        "data_avaliacao": _clean_date(edit_data_avaliacao),
                        "peso_kg": edit_peso if edit_peso > 0 else None,
                        "altura_cm": edit_altura if edit_altura > 0 else None,
                        "envergadura_cm": edit_envergadura if edit_envergadura > 0 else None,
                        "comprimento_perna_cm": edit_perna if edit_perna > 0 else None,
                        "altura_sentada_cm": edit_sentada if edit_sentada > 0 else None,
                        "salto_maturacional": edit_offset,
                        "estado_maturacional": edit_status,
                    },
                )
                st.session_state[edit_state_key] = False
                st.success("Registo antropometrico atualizado com sucesso.")
                st.rerun()


def _render_physical_tests_tab(athletes_df: pd.DataFrame, physical_df: pd.DataFrame) -> None:
    st.subheader("Ficha Testes Fisicos")
    athlete_options = athletes_df["atleta_id"].astype(str).tolist() if not athletes_df.empty else []
    if not athlete_options:
        st.info("Cria primeiro atletas na ficha mestre.")
        return

    selected_id, _ = _render_record_athlete_header(
        athletes_df,
        athlete_options,
        "physical_atleta_id",
    )

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
        col1, col2 = st.columns([1, 1.2])
        data_avaliacao = col1.date_input("Data avaliacao", value=myjump_defaults.get("data_avaliacao"))
        col2.markdown("**Conferencia MyJumpLab**")
        col2.caption(imported_name or "-")

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
        new_record = _build_physical_record(physical_df, selected_id, data_avaliacao, {
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
        })
        append_records("physical", new_record, source_type="manual_physical_tests")
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
    physical_history_df = physical_df[~physical_df["source_type"].astype(str).isin(["manual_anthropometry"])].copy() if not physical_df.empty else physical_df
    if physical_history_df.empty:
        st.info("Sem historico fisico.")
    else:
        history_df = physical_history_df.copy()
        history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce").dt.strftime("%d/%m/%Y")
        history_df["inserted_at"] = pd.to_datetime(history_df["inserted_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        selection_df = history_df[
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
            ].reset_index(drop=True)
        display_df = selection_df.drop(columns=["record_id", "batch_id", "inserted_at", "source_type", "source_file", "atleta_id"])
        selection = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        selected_row = _selected_table_row(selection_df, selection)
        selected_record_id = None if selected_row is None else str(selected_row["record_id"])
        if not selected_record_id:
            st.caption("Seleciona uma linha do historico para editar ou eliminar.")
            return

        edit_state_key = "open_edit_physical_record"
        action_col1, action_col2 = st.columns(2)
        if action_col1.button("Editar registo fisico", key="open_edit_physical_btn"):
            _toggle_state(edit_state_key)
            st.rerun()
        if action_col2.button("Eliminar registo fisico", key="delete_physical_btn"):
            delete_record("physical", selected_record_id)
            st.success("Registo fisico eliminado com sucesso.")
            st.rerun()
        edit_row = _record_row_by_id(physical_history_df, selected_record_id)
        if st.session_state.get(edit_state_key, False) and edit_row is not None:
            with st.form("edit_physical_record_form"):
                e1, e2, e3, e4 = st.columns(4)
                edit_data_avaliacao = e1.date_input("Data avaliacao", value=_clean_date(edit_row.get("data_avaliacao")), format="DD/MM/YYYY", key="edit_phys_date")
                edit_sprint10 = e2.number_input("Sprint 10m (s)", min_value=0.0, value=float(edit_row.get("sprint_10m_s", 0.0) or 0.0), step=0.01, key="edit_phys_s10")
                edit_sprint20 = e3.number_input("Sprint 20m (s)", min_value=0.0, value=float(edit_row.get("sprint_20m_s", 0.0) or 0.0), step=0.01, key="edit_phys_s20")
                edit_cmj = e4.number_input("CMJ altura (cm)", min_value=0.0, value=float(edit_row.get("cmj_altura_cm", 0.0) or 0.0), step=0.1, key="edit_phys_cmj")
                e5, e6, e7 = st.columns(3)
                edit_sj = e5.number_input("SJ altura (cm)", min_value=0.0, value=float(edit_row.get("sj_altura_cm", 0.0) or 0.0), step=0.1, key="edit_phys_sj")
                edit_rsi = e6.number_input("10J RSI 10-5", min_value=0.0, value=float(edit_row.get("j10_rsi_10_5", 0.0) or 0.0), step=0.01, key="edit_phys_rsi")
                edit_fadiga = e7.number_input("Indice fadiga 10J (%)", value=float(edit_row.get("indice_fadiga_10j_pct", 0.0) or 0.0), step=0.1, key="edit_phys_fadiga")
                save_edit = st.form_submit_button("Guardar alteracoes do registo", type="primary")
            if save_edit:
                update_record(
                    "physical",
                    selected_record_id,
                    {
                        "data_avaliacao": _clean_date(edit_data_avaliacao),
                        "sprint_10m_s": edit_sprint10 if edit_sprint10 > 0 else None,
                        "sprint_20m_s": edit_sprint20 if edit_sprint20 > 0 else None,
                        "cmj_altura_cm": edit_cmj if edit_cmj > 0 else None,
                        "sj_altura_cm": edit_sj if edit_sj > 0 else None,
                        "j10_rsi_10_5": edit_rsi if edit_rsi > 0 else None,
                        "indice_fadiga_10j_pct": edit_fadiga if edit_fadiga != 0 else None,
                    },
                )
                st.session_state[edit_state_key] = False
                st.success("Registo fisico atualizado com sucesso.")
                st.rerun()


def _render_technical_tab(athletes_df: pd.DataFrame, technical_df: pd.DataFrame) -> None:
    st.subheader("Registo Tecnico | Tatica | Psicologico")
    athlete_options = athletes_df["atleta_id"].astype(str).tolist() if not athletes_df.empty else []
    if not athlete_options:
        st.info("Cria primeiro atletas na ficha mestre.")
        return

    atleta_id, selected_athlete = _render_record_athlete_header(
        athletes_df,
        athlete_options,
        "technical_atleta_id",
    )
    is_goalkeeper = _clean_text_value(selected_athlete.get("posicao")) == "GR" if selected_athlete is not None else False

    with st.form("technical_single_record_form", clear_on_submit=True):
        data_avaliacao = st.date_input("Data avaliacao", value=None, format="DD/MM/YYYY", key="technical_data_avaliacao")
        latest_snapshot = _latest_technical_snapshot(technical_df, atleta_id)
        um_x_um_ofensivo_score = None
        um_x_um_defensivo_score = None
        lateralidade_score = None
        imprevisibilidade_score = None
        dominio_espaco_score = None
        posicionamento_prontidao_score = None
        defesa_membros_superiores_score = None
        defesa_membros_inferiores_score = None
        defesa_6m_ocupa_espaco_score = None
        if is_goalkeeper:
            st.markdown("**Guarda-Redes**")
            c1, c2, c3 = st.columns(3)
            posicionamento_prontidao_score = c1.number_input("Posicionamento (posturas de prontidao)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            defesa_membros_superiores_score = c2.number_input("Defesa Membros Superiores", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            defesa_membros_inferiores_score = c3.number_input("Defesa Membros Inferiores", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            c4, c5 = st.columns(2)
            defesa_6m_ocupa_espaco_score = c4.number_input("Defesa 6m (ocupa Espaco)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            leitura_jogo_score = c5.number_input("Leitura de jogo (antecipacao/intercecao)", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        else:
            st.markdown("**Tecnica**")
            c1, c2, c3 = st.columns(3)
            um_x_um_ofensivo_score = c1.number_input("1x1 Ofensivo", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            um_x_um_defensivo_score = c2.number_input("1x1 Defensivo", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            lateralidade_score = c3.number_input("Lateralidade", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            st.markdown("**Tatica**")
            c4, c5, c6 = st.columns(3)
            imprevisibilidade_score = c4.number_input("Imprevisibilidade", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            leitura_jogo_score = c5.number_input("Leitura de Jogo", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
            dominio_espaco_score = c6.number_input("Dominio do Espaco", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        st.markdown("**Psicologico**")
        p1, p2, p3, p4 = st.columns(4)
        espirito_equipa_score = p1.number_input("Espírito de equipa", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        controlo_emocional_score = p2.number_input("Controlo emocional", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        tenacidade_resiliencia_score = p3.number_input("Tenacidade / Resiliência", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        atencao_concentracao_score = p4.number_input("Atenção / concentração", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        observacoes = st.text_input("Observacoes", value=_clean_text_value(latest_snapshot.get("observacoes")))
        submitted = st.form_submit_button("Guardar registo tecnico/tatica/psicologico", type="primary")

    if submitted:
        record_df = _build_technical_record(technical_df, atleta_id, data_avaliacao, {
            "um_x_um_ofensivo_score": um_x_um_ofensivo_score if um_x_um_ofensivo_score > 0 else None,
            "um_x_um_defensivo_score": um_x_um_defensivo_score if um_x_um_defensivo_score > 0 else None,
            "lateralidade_score": lateralidade_score if lateralidade_score > 0 else None,
            "imprevisibilidade_score": imprevisibilidade_score if imprevisibilidade_score > 0 else None,
            "leitura_jogo_score": leitura_jogo_score if leitura_jogo_score > 0 else None,
            "dominio_espaco_score": dominio_espaco_score if dominio_espaco_score > 0 else None,
            "posicionamento_prontidao_score": posicionamento_prontidao_score if posicionamento_prontidao_score and posicionamento_prontidao_score > 0 else None,
            "defesa_membros_superiores_score": defesa_membros_superiores_score if defesa_membros_superiores_score and defesa_membros_superiores_score > 0 else None,
            "defesa_membros_inferiores_score": defesa_membros_inferiores_score if defesa_membros_inferiores_score and defesa_membros_inferiores_score > 0 else None,
            "defesa_6m_ocupa_espaco_score": defesa_6m_ocupa_espaco_score if defesa_6m_ocupa_espaco_score and defesa_6m_ocupa_espaco_score > 0 else None,
            "espirito_equipa_score": espirito_equipa_score if espirito_equipa_score > 0 else None,
            "controlo_emocional_score": controlo_emocional_score if controlo_emocional_score > 0 else None,
            "tenacidade_resiliencia_score": tenacidade_resiliencia_score if tenacidade_resiliencia_score > 0 else None,
            "atencao_concentracao_score": atencao_concentracao_score if atencao_concentracao_score > 0 else None,
            "observacoes": _clean_text_value(observacoes),
        })
        append_records("technical", record_df, source_type="manual_technical")
        st.success("Registo tecnico/tatica/psicologico guardado com sucesso.")
        st.rerun()

    st.divider()
    st.markdown("**Historico tecnico/tatica/psicologico**")
    if technical_df.empty:
        st.info("Sem historico tecnico/tatica/psicologico.")
    else:
        history_df = technical_df.copy()
        history_df["data_avaliacao"] = pd.to_datetime(history_df["data_avaliacao"], errors="coerce").dt.strftime("%d/%m/%Y")
        history_df["inserted_at"] = pd.to_datetime(history_df["inserted_at"], errors="coerce").dt.strftime("%d/%m/%Y %H:%M")
        selection_df = history_df[["record_id", "batch_id", "inserted_at", "source_type", "source_file", "atleta_id", "data_avaliacao", "um_x_um_ofensivo_score", "um_x_um_defensivo_score", "lateralidade_score", "imprevisibilidade_score", "leitura_jogo_score", "dominio_espaco_score", "posicionamento_prontidao_score", "defesa_membros_superiores_score", "defesa_membros_inferiores_score", "defesa_6m_ocupa_espaco_score", "espirito_equipa_score", "controlo_emocional_score", "tenacidade_resiliencia_score", "atencao_concentracao_score"]].reset_index(drop=True)
        display_df = selection_df.drop(columns=["record_id", "batch_id", "inserted_at", "source_type", "source_file", "atleta_id"])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.divider()
    st.markdown("**Upload normalizado em lote**")
    st.download_button("Descarregar template tecnico/tatica/psicologico CSV", data=_build_template_csv(TECHNICAL_UPLOAD_COLUMNS), file_name="template_tecnico_tatica_psicologico_futsal.csv", mime="text/csv")
    uploaded_technical = st.file_uploader("Carregar ficheiro tecnico/tatica/psicologico normalizado", type=["csv"], key="technical_bulk_upload")
    if uploaded_technical is not None and st.button("Importar lote tecnico/tatica/psicologico", key="import_technical_batch"):
        try:
            df_upload = _read_table_upload(uploaded_technical)
            df_prepared = _prepare_uploaded_records(df_upload, TECHNICAL_UPLOAD_COLUMNS)
            _validate_athlete_ids(df_prepared, athletes_df)
            result = append_records("technical", df_prepared, source_type="upload", source_file=uploaded_technical.name)
        except Exception as exc:
            st.error(str(exc))
        else:
            st.success(f"Lote tecnico/tatica/psicologico importado com sucesso. Batch: {result['batch_id']} | Linhas: {result['inserted']}")
            st.rerun()


st.title("Registos de Atletas")
st.caption("Parquet local com ficha mestre de atletas, ficha antropometrica/fisica e ficha tecnica/psicologica.")

athletes_df = read_athletes()
athlete_history_df = read_athlete_history()
pathway_df = read_athlete_pathway()
physical_df = read_physical_records()
technical_df = read_technical_records()

_render_summary_metrics(athletes_df, pathway_df)
_render_athlete_registry(athletes_df, athlete_history_df, physical_df, technical_df, pathway_df)
