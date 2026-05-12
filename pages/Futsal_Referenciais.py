from __future__ import annotations

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


st.title("Referenciais")
st.caption("Camada analitica da modalidade: base historica, percentis, z-score e evolucao automatica.")

history_df = build_metric_history()
summary = build_database_summary()

top1, top2, top3, top4 = st.columns(4)
top1.metric("Atletas", summary["n_atletas"])
top2.metric("Avaliacoes fisicas", summary["n_avaliacoes_fisicas"])
top3.metric("Avaliacoes tecnica | psicologica", summary["n_avaliacoes_tecnicas"])
top4.metric("Registos metricos", summary["n_registos_metricos"])

mid1, mid2, mid3 = st.columns(3)
mid1.metric("Anos historicos", summary["n_anos"])
mid2.metric("Selecoes", summary["n_selecoes"])
mid3.metric("Testes ativos", summary["n_testes"])

if history_df.empty:
    st.info("Ainda nao existem dados historicos suficientes para calcular referenciais.")
    st.stop()

filter_col1, filter_col2, filter_col3 = st.columns(3)
available_metrics = history_df[["metric_group", "metric_label", "metric_key"]].drop_duplicates().copy()
group_options = available_metrics["metric_group"].dropna().drop_duplicates().sort_values().tolist()
selected_group = filter_col1.selectbox("Bloco", options=group_options, index=0)

group_metrics = (
    available_metrics.loc[available_metrics["metric_group"] == selected_group, ["metric_label", "metric_key"]]
    .drop_duplicates()
    .sort_values("metric_label")
)
metric_label = filter_col2.selectbox("Metrica", options=group_metrics["metric_label"].tolist(), index=0)
metric_key = group_metrics.loc[group_metrics["metric_label"] == metric_label, "metric_key"].iloc[0]
reference_scope = filter_col3.selectbox(
    "Escopo",
    options=["Todas as avaliacoes", "Ultima avaliacao por atleta"],
    index=0,
)

metric_df = history_df[history_df["metric_key"] == metric_key].copy()
if reference_scope == "Ultima avaliacao por atleta":
    metric_df = latest_metric_records(metric_df)

filter_row2_col1, filter_row2_col2, filter_row2_col3, filter_row2_col4 = st.columns(4)
selection_options = _filter_values(sorted(metric_df["selecao"].dropna().astype(str).unique().tolist()))
escalao_options = _filter_values(sorted(metric_df["escalao_avaliacao"].dropna().astype(str).unique().tolist()))
position_options = _filter_values(sorted(metric_df["posicao"].dropna().astype(str).unique().tolist()))
year_options = sorted([int(value) for value in metric_df["ano_avaliacao"].dropna().unique().tolist()])

selected_selections = filter_row2_col1.multiselect("Selecao", options=selection_options)
selected_escaloes = filter_row2_col2.multiselect("Escalao", options=escalao_options)
selected_positions = filter_row2_col3.multiselect("Posicao", options=position_options)
selected_years = filter_row2_col4.multiselect("Ano", options=year_options)

filter_row3_col1, filter_row3_col2 = st.columns(2)
maturity_options = _filter_values(sorted(metric_df["estado_maturacional"].dropna().astype(str).unique().tolist()))
selected_maturity = filter_row3_col1.multiselect("Estado maturacional", options=maturity_options)
selected_genders = filter_row3_col2.multiselect(
    "Genero",
    options=_filter_values(sorted(metric_df["genero"].dropna().astype(str).unique().tolist())),
)

if selected_selections:
    metric_df = metric_df[metric_df["selecao"].isin(selected_selections)].copy()
if selected_escaloes:
    metric_df = metric_df[metric_df["escalao_avaliacao"].isin(selected_escaloes)].copy()
if selected_positions:
    metric_df = metric_df[metric_df["posicao"].isin(selected_positions)].copy()
if selected_years:
    metric_df = metric_df[metric_df["ano_avaliacao"].isin(selected_years)].copy()
if selected_maturity:
    metric_df = metric_df[metric_df["estado_maturacional"].isin(selected_maturity)].copy()
if selected_genders:
    metric_df = metric_df[metric_df["genero"].isin(selected_genders)].copy()

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

card1, card2, card3, card4, card5, card6 = st.columns(6)
card1.metric("N avaliacoes", stats["n_avaliacoes"])
card2.metric("N atletas", stats["n_atletas"])
card3.metric("Media historica", _fmt_number(stats["media"], suffix=f" {unit}" if unit else ""))
card4.metric("Desvio padrao", _fmt_number(stats["desvio_padrao"], suffix=f" {unit}" if unit else ""))
card5.metric("P50", _fmt_number(stats["p50"], suffix=f" {unit}" if unit else ""))
cutoff_label = "Top 10% (max)" if direction == "higher" else "Top 10% (min)"
card6.metric(cutoff_label, _fmt_number(stats["top_10_cutoff"], suffix=f" {unit}" if unit else ""))

percent_col1, percent_col2, percent_col3, percent_col4 = st.columns(4)
percent_col1.metric("P10", _fmt_number(stats["p10"], suffix=f" {unit}" if unit else ""))
percent_col2.metric("P25", _fmt_number(stats["p25"], suffix=f" {unit}" if unit else ""))
percent_col3.metric("P75", _fmt_number(stats["p75"], suffix=f" {unit}" if unit else ""))
percent_col4.metric("P90", _fmt_number(stats["p90"], suffix=f" {unit}" if unit else ""))

if not trend_df.empty:
    st.markdown("**Evolucao temporal**")
    chart_df = trend_df.set_index("ano_avaliacao")[["media", "mediana"]]
    st.line_chart(chart_df, use_container_width=True)
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
