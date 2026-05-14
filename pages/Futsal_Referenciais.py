from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from fpf_modules.futsal_analytics import (
    build_database_summary,
    build_metric_history,
    build_yearly_trend,
    compute_reference_stats,
    latest_metric_records,
    score_metric_records,
)


def _fmt_number(value, decimals: int = 2, suffix: str = "") -> str:
    if value in ("", None) or pd.isna(value):
        return "-"
    return f"{float(value):.{decimals}f}{suffix}"


def _filter_values(options: list[str]) -> list[str]:
    return [value for value in options if str(value).strip()]


def _render_stat_strip(items: list[tuple[str, str | int]], class_name: str) -> None:
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
            grid-template-columns: repeat({len(items)}, minmax(110px, 1fr));
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
                grid-template-columns: repeat(4, minmax(110px, 1fr));
            }}
        }}
        @media (max-width: 720px) {{
            .{class_name}-strip {{
                grid-template-columns: repeat(2, minmax(110px, 1fr));
            }}
        }}
        </style>
        <div class="{class_name}-strip">{cards_html}</div>
        """,
        unsafe_allow_html=True,
    )


def _render_summary_strip(summary: dict) -> None:
    items = [
        ("Atletas", summary["n_atletas"]),
        ("Avaliacoes fisicas", summary["n_avaliacoes_fisicas"]),
        ("Avaliacoes tecnica | psicologica", summary["n_avaliacoes_tecnicas"]),
        ("Registos metricos", summary["n_registos_metricos"]),
        ("Anos historicos", summary["n_anos"]),
        ("Selecoes", summary["n_selecoes"]),
        ("Testes ativos", summary["n_testes"]),
    ]
    _render_stat_strip(items, "ref-summary")


def _render_trend_chart(trend_df: pd.DataFrame, metric_label: str, unit: str) -> None:
    if trend_df.empty:
        return
    chart_df = trend_df.copy()
    chart_df["Ano"] = chart_df["ano_avaliacao"].astype(str)
    chart_df["Media"] = chart_df["media"]
    chart_df["Mediana"] = chart_df["mediana"]
    value_label = f"Valor ({unit})" if unit else "Valor"
    media_color = "#0f766e"
    mediana_color = "#dc6803"

    base = alt.Chart(chart_df).encode(
        x=alt.X("Ano:N", title="Ano"),
    )

    media_line = base.mark_line(strokeWidth=3, color=media_color).encode(
        y=alt.Y("Media:Q", title=value_label),
        tooltip=[
            alt.Tooltip("Ano:N", title="Ano"),
            alt.Tooltip("Media:Q", title="Media", format=".2f"),
            alt.Tooltip("n_avaliacoes:Q", title="N avaliacoes"),
            alt.Tooltip("n_atletas:Q", title="N atletas"),
        ],
    )

    mediana_line = base.mark_line(strokeWidth=2.5, color=mediana_color, strokeDash=[5, 4]).encode(
        y=alt.Y("Mediana:Q", title=value_label),
        tooltip=[
            alt.Tooltip("Ano:N", title="Ano"),
            alt.Tooltip("Mediana:Q", title="Mediana", format=".2f"),
            alt.Tooltip("n_avaliacoes:Q", title="N avaliacoes"),
            alt.Tooltip("n_atletas:Q", title="N atletas"),
        ],
    )

    media_points = base.mark_circle(size=85, color=media_color, stroke="white", strokeWidth=1.5).encode(
        y=alt.Y("Media:Q", title=value_label),
        tooltip=[
            alt.Tooltip("Ano:N", title="Ano"),
            alt.Tooltip("Media:Q", title="Media", format=".2f"),
            alt.Tooltip("n_avaliacoes:Q", title="N avaliacoes"),
            alt.Tooltip("n_atletas:Q", title="N atletas"),
        ],
    )

    mediana_points = base.mark_square(size=75, color=mediana_color, stroke="white", strokeWidth=1.5).encode(
        y=alt.Y("Mediana:Q", title=value_label),
        tooltip=[
            alt.Tooltip("Ano:N", title="Ano"),
            alt.Tooltip("Mediana:Q", title="Mediana", format=".2f"),
            alt.Tooltip("n_avaliacoes:Q", title="N avaliacoes"),
            alt.Tooltip("n_atletas:Q", title="N atletas"),
        ],
    )

    media_label_df = chart_df.tail(1).copy()
    media_label_df["serie"] = "Media"
    media_label_df["valor_label"] = media_label_df["Media"].map(lambda value: f"Media {_fmt_number(value)}")
    mediana_label_df = chart_df.tail(1).copy()
    mediana_label_df["serie"] = "Mediana"
    mediana_label_df["valor_label"] = mediana_label_df["Mediana"].map(lambda value: f"Mediana {_fmt_number(value)}")

    media_text = alt.Chart(media_label_df).mark_text(
        align="left",
        dx=10,
        dy=-10,
        fontSize=11,
        fontWeight=700,
        color=media_color,
    ).encode(
        x=alt.X("Ano:N"),
        y=alt.Y("Media:Q"),
        text="valor_label:N",
    )

    mediana_text = alt.Chart(mediana_label_df).mark_text(
        align="left",
        dx=10,
        dy=12,
        fontSize=11,
        fontWeight=700,
        color=mediana_color,
    ).encode(
        x=alt.X("Ano:N"),
        y=alt.Y("Mediana:Q"),
        text="valor_label:N",
    )

    chart = (
        alt.layer(media_line, mediana_line, media_points, mediana_points, media_text, mediana_text)
        .resolve_scale(y="shared")
        .properties(height=320, title=f"Evolucao Temporal | {metric_label}")
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#475467", titleColor="#344054", gridColor="#eaecf0")
        .configure_title(anchor="start", color="#101828", fontSize=16, fontWeight=700)
        .configure_legend(labelColor="#475467", titleColor="#344054")
    )
    st.altair_chart(chart, use_container_width=True)


st.title("Referenciais")
st.caption("Camada analitica da modalidade: base historica, percentis, z-score e evolucao automatica.")

history_df = build_metric_history()
summary = build_database_summary()

_render_summary_strip(summary)

if history_df.empty:
    st.info("Ainda nao existem dados historicos suficientes para calcular referenciais.")
    st.stop()

available_metrics = history_df[["metric_group", "metric_label", "metric_key"]].drop_duplicates().copy()
reference_scope_col, _ = st.columns([1.1, 4.9])
reference_scope = reference_scope_col.selectbox(
    "Escopo",
    options=["Todas as avaliacoes", "Ultima avaliacao por atleta"],
    index=0,
)
working_df = history_df.copy()
if reference_scope == "Ultima avaliacao por atleta":
    working_df = latest_metric_records(working_df)

filter_row1_col1, filter_row1_col2, filter_row1_col3, filter_row1_col4 = st.columns([0.8, 1.25, 1.1, 0.95])
year_options = sorted([int(value) for value in working_df["ano_avaliacao"].dropna().unique().tolist()])
selection_options = _filter_values(sorted(working_df["selecao"].dropna().astype(str).unique().tolist()))
escalao_options = _filter_values(sorted(working_df["escalao_avaliacao"].dropna().astype(str).unique().tolist()))
position_options = _filter_values(sorted(working_df["posicao"].dropna().astype(str).unique().tolist()))

selected_years = filter_row1_col1.multiselect("Ano", options=year_options)
selected_selections = filter_row1_col2.multiselect("Selecao", options=selection_options)
selected_escaloes = filter_row1_col3.multiselect("Escalao", options=escalao_options)
selected_positions = filter_row1_col4.multiselect("Posicao", options=position_options)

filter_row2_col1, filter_row2_col2 = st.columns([0.9, 1.35])
maturity_options = _filter_values(sorted(working_df["estado_maturacional"].dropna().astype(str).unique().tolist()))
gender_options = _filter_values(sorted(working_df["genero"].dropna().astype(str).unique().tolist()))

selected_genders = filter_row2_col1.multiselect("Genero", options=gender_options)
selected_maturity = filter_row2_col2.multiselect("Estado maturacional", options=maturity_options)

if selected_years:
    working_df = working_df[working_df["ano_avaliacao"].isin(selected_years)].copy()
if selected_selections:
    working_df = working_df[working_df["selecao"].isin(selected_selections)].copy()
if selected_escaloes:
    working_df = working_df[working_df["escalao_avaliacao"].isin(selected_escaloes)].copy()
if selected_positions:
    working_df = working_df[working_df["posicao"].isin(selected_positions)].copy()
if selected_genders:
    working_df = working_df[working_df["genero"].isin(selected_genders)].copy()
if selected_maturity:
    working_df = working_df[working_df["estado_maturacional"].isin(selected_maturity)].copy()

filter_row3_col1, filter_row3_col2 = st.columns([1.15, 1.75])
group_options = working_df["metric_group"].dropna().drop_duplicates().sort_values().tolist()
selected_group = filter_row3_col1.selectbox("Bloco", options=group_options, index=0)

group_metrics = (
    working_df.loc[working_df["metric_group"] == selected_group, ["metric_label", "metric_key"]]
    .drop_duplicates()
    .sort_values("metric_label")
)
metric_label = filter_row3_col2.selectbox("Metrica", options=group_metrics["metric_label"].tolist(), index=0)
metric_key = group_metrics.loc[group_metrics["metric_label"] == metric_label, "metric_key"].iloc[0]

metric_df = working_df[working_df["metric_key"] == metric_key].copy()

if metric_df.empty:
    st.warning("Nao existem linhas para a combinacao de filtros atual.")
    st.stop()

direction = metric_df["direction"].iloc[0]
unit = metric_df["metric_unit"].iloc[0]
stats = compute_reference_stats(metric_df)
scored_df = score_metric_records(metric_df)
trend_df = build_yearly_trend(metric_df)

st.markdown(f"**Referencial ativo:** {metric_label}")
if direction == "lower":
    st.caption("Leitura interpretativa: valores mais baixos sao melhores para esta metrica.")
else:
    st.caption("Leitura interpretativa: valores mais altos sao melhores para esta metrica.")

cutoff_label = "Top 10% (max)" if direction == "higher" else "Top 10% (min)"
reference_items = [
    ("N avaliacoes", stats["n_avaliacoes"]),
    ("N atletas", stats["n_atletas"]),
    ("Media historica", _fmt_number(stats["media"], suffix=f" {unit}" if unit else "")),
    ("Desvio padrao", _fmt_number(stats["desvio_padrao"], suffix=f" {unit}" if unit else "")),
    ("P50", _fmt_number(stats["p50"], suffix=f" {unit}" if unit else "")),
    (cutoff_label, _fmt_number(stats["top_10_cutoff"], suffix=f" {unit}" if unit else "")),
]
percentile_items = [
    ("P10", _fmt_number(stats["p10"], suffix=f" {unit}" if unit else "")),
    ("P25", _fmt_number(stats["p25"], suffix=f" {unit}" if unit else "")),
    ("P75", _fmt_number(stats["p75"], suffix=f" {unit}" if unit else "")),
    ("P90", _fmt_number(stats["p90"], suffix=f" {unit}" if unit else "")),
]
_render_stat_strip(reference_items, "ref-metrics")
_render_stat_strip(percentile_items, "ref-percentiles")

if not trend_df.empty:
    st.markdown("**Evolucao temporal**")
    _render_trend_chart(trend_df, metric_label, unit)
    st.dataframe(trend_df, use_container_width=True, hide_index=True)

st.markdown("**Tabela de referenciais**")
ranked_view = scored_df[
    [
        "atleta_id",
        "nome",
        "selecao",
        "posicao",
        "escalao_avaliacao",
        "data_avaliacao",
        "metric_value",
        "percentil_historico",
        "z_score",
        "estado_maturacional",
    ]
].copy()
ranked_view = ranked_view.rename(
    columns={
        "atleta_id": "ID",
        "nome": "Nome",
        "selecao": "Selecao",
        "posicao": "Posicao",
        "escalao_avaliacao": "Escalao",
        "data_avaliacao": "Data avaliacao",
        "metric_value": metric_label,
        "percentil_historico": "Percentil",
        "z_score": "Z-Score",
        "estado_maturacional": "Estado maturacional",
    }
)
st.dataframe(ranked_view, use_container_width=True, hide_index=True)

csv_export = ranked_view.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descarregar tabela de referenciais",
    data=csv_export,
    file_name=f"referenciais_{metric_key}.csv",
    mime="text/csv",
)

with st.expander("Base historica desta metrica"):
    base_view = metric_df[
        [
            "record_source",
            "atleta_id",
            "nome",
            "data_avaliacao",
            "ano_avaliacao",
            "selecao",
            "posicao",
            "escalao_avaliacao",
            "metric_value",
            "estado_maturacional",
            "salto_maturacional",
        ]
    ].copy()
    base_view = base_view.rename(
        columns={
            "record_source": "Origem",
            "atleta_id": "ID",
            "nome": "Nome",
            "data_avaliacao": "Data avaliacao",
            "ano_avaliacao": "Ano",
            "selecao": "Selecao",
            "posicao": "Posicao",
            "escalao_avaliacao": "Escalao",
            "metric_value": metric_label,
            "estado_maturacional": "Estado maturacional",
            "salto_maturacional": "Salto maturacional",
        }
    )
    st.dataframe(base_view, use_container_width=True, hide_index=True)
