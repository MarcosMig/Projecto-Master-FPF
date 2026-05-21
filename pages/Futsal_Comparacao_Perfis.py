from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fpf_modules.auth_manager import require_login
from fpf_modules.futsal_analytics import build_profile_scores_dataframe


require_login()

st.title("Comparacao de Perfis")
st.caption("Compare o perfil global de 2 a 4 atletas a partir dos indices ANT | FIS | TEC | TAT | PSI.")


def _load_profiles() -> pd.DataFrame:
    return build_profile_scores_dataframe()


def _clean_text(value) -> str:
    if value in ("", None) or pd.isna(value):
        return ""
    return str(value).strip()


def _format_int(value) -> str:
    if value in ("", None) or pd.isna(value):
        return "--"
    try:
        return str(int(round(float(value))))
    except (TypeError, ValueError):
        return "--"


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
    value = None if score_value in ("", None) or pd.isna(score_value) else float(score_value)
    if value is None:
        return "#667085", "#98a2b3"
    clipped = max(0.0, min(100.0, value))
    hue = 120.0 * (clipped / 100.0)
    base = _hsl_to_hex(hue, 0.76, 0.40)
    accent = _hsl_to_hex(hue, 0.88, 0.52)
    return base, accent


def _athlete_label(row: pd.Series) -> str:
    name = _clean_text(row.get("nome")) or _clean_text(row.get("atleta_id"))
    pos = _clean_text(row.get("posicao"))
    esc = _clean_text(row.get("escalao"))
    parts = [name]
    meta = " | ".join(part for part in [row.get("atleta_id"), pos, esc] if _clean_text(part))
    if meta:
        parts.append(f"({meta})")
    return " ".join(parts)


def _score_value(row: pd.Series, key: str) -> float | None:
    value = row.get(key)
    return None if value in ("", None) or pd.isna(value) else float(value)


def _profile_distance(base_row: pd.Series, candidate_row: pd.Series) -> tuple[float | None, dict[str, float]]:
    score_keys = ["ANT", "FIS", "TEC", "TAT", "PSI"]
    diffs: dict[str, float] = {}
    for key in score_keys:
        base_value = _score_value(base_row, key)
        candidate_value = _score_value(candidate_row, key)
        if base_value is not None and candidate_value is not None:
            diffs[key] = abs(base_value - candidate_value)
    if not diffs:
        return None, {}

    mean_diff = sum(diffs.values()) / len(diffs)
    penalty = 0.0
    if _clean_text(base_row.get("genero")) and _clean_text(base_row.get("genero")) != _clean_text(candidate_row.get("genero")):
        penalty += 12.0
    if _clean_text(base_row.get("posicao")) and _clean_text(base_row.get("posicao")) != _clean_text(candidate_row.get("posicao")):
        penalty += 15.0
    if _clean_text(base_row.get("escalao")) and _clean_text(base_row.get("escalao")) != _clean_text(candidate_row.get("escalao")):
        penalty += 8.0
    if _clean_text(base_row.get("selecao")) and _clean_text(base_row.get("selecao")) != _clean_text(candidate_row.get("selecao")):
        penalty += 4.0
    return mean_diff + penalty, diffs


def _find_similar_profiles(base_row: pd.Series, candidates_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    rows: list[dict] = []
    for _, candidate_row in candidates_df.iterrows():
        if _clean_text(candidate_row.get("atleta_id")) == _clean_text(base_row.get("atleta_id")):
            continue
        distance, diffs = _profile_distance(base_row, candidate_row)
        if distance is None:
            continue
        similarity = max(0.0, 100.0 - distance)
        rows.append(
            {
                "ID": candidate_row.get("atleta_id"),
                "Nome": candidate_row.get("nome"),
                "Selecao": candidate_row.get("selecao"),
                "Escalao": candidate_row.get("escalao"),
                "Posicao": candidate_row.get("posicao"),
                "Similaridade": round(similarity, 1),
                "Distancia perfil": round(distance, 1),
                "ANT": None if pd.isna(candidate_row.get("ANT")) else int(round(float(candidate_row.get("ANT")))),
                "FIS": None if pd.isna(candidate_row.get("FIS")) else int(round(float(candidate_row.get("FIS")))),
                "TEC": None if pd.isna(candidate_row.get("TEC")) else int(round(float(candidate_row.get("TEC")))),
                "TAT": None if pd.isna(candidate_row.get("TAT")) else int(round(float(candidate_row.get("TAT")))),
                "PSI": None if pd.isna(candidate_row.get("PSI")) else int(round(float(candidate_row.get("PSI")))),
                "Desvio ANT": round(diffs.get("ANT", 0.0), 1),
                "Desvio FIS": round(diffs.get("FIS", 0.0), 1),
                "Desvio TEC": round(diffs.get("TEC", 0.0), 1),
                "Desvio TAT": round(diffs.get("TAT", 0.0), 1),
                "Desvio PSI": round(diffs.get("PSI", 0.0), 1),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Similaridade", "Distancia perfil", "Nome"], ascending=[False, True, True]).head(top_n).reset_index(drop=True)


def _render_score_pills(row: pd.Series) -> None:
    ant_base, ant_accent = _score_gradient(row.get("ANT"))
    fis_base, fis_accent = _score_gradient(row.get("FIS"))
    tec_base, tec_accent = _score_gradient(row.get("TEC"))
    tat_base, tat_accent = _score_gradient(row.get("TAT"))
    psi_base, psi_accent = _score_gradient(row.get("PSI"))
    html = f"""
    <style>
    .compare-score-strip {{
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 8px;
        margin: 10px 0 0 0;
    }}
    .compare-score-pill {{
        text-align: center;
        color: white;
        border-radius: 16px;
        padding: 10px 6px 8px;
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .compare-score-pill.ant {{ background: linear-gradient(135deg, {ant_base}, {ant_accent}); }}
    .compare-score-pill.fis {{ background: linear-gradient(135deg, {fis_base}, {fis_accent}); }}
    .compare-score-pill.tec {{ background: linear-gradient(135deg, {tec_base}, {tec_accent}); }}
    .compare-score-pill.tat {{ background: linear-gradient(135deg, {tat_base}, {tat_accent}); }}
    .compare-score-pill.psi {{ background: linear-gradient(135deg, {psi_base}, {psi_accent}); }}
    .compare-score-value {{
        font-size: 24px;
        line-height: 1;
        font-weight: 700;
        margin-bottom: 4px;
    }}
    .compare-score-label {{
        font-size: 11px;
        letter-spacing: 0.08em;
        font-weight: 600;
    }}
    </style>
    <div class="compare-score-strip">
        <div class="compare-score-pill ant"><div class="compare-score-value">{_format_int(row.get("ANT"))}</div><div class="compare-score-label">ANT</div></div>
        <div class="compare-score-pill fis"><div class="compare-score-value">{_format_int(row.get("FIS"))}</div><div class="compare-score-label">FIS</div></div>
        <div class="compare-score-pill tec"><div class="compare-score-value">{_format_int(row.get("TEC"))}</div><div class="compare-score-label">TEC</div></div>
        <div class="compare-score-pill tat"><div class="compare-score-value">{_format_int(row.get("TAT"))}</div><div class="compare-score-label">TAT</div></div>
        <div class="compare-score-pill psi"><div class="compare-score-value">{_format_int(row.get("PSI"))}</div><div class="compare-score-label">PSI</div></div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def _render_athlete_card(row: pd.Series) -> None:
    photo_path = _clean_text(row.get("photo_path"))
    if photo_path and Path(photo_path).exists():
        st.image(photo_path, width=110)
    else:
        st.caption("Sem foto registada.")
    st.markdown(f"**{_clean_text(row.get('nome'))}**")
    st.caption(f"ID: {_clean_text(row.get('atleta_id'))}")
    st.caption(f"Posicao: {_clean_text(row.get('posicao'))} | Escalao: {_clean_text(row.get('escalao'))}")
    st.caption(f"Selecao: {_clean_text(row.get('selecao'))} | Genero: {_clean_text(row.get('genero'))}")
    st.caption(f"Idade: {_format_int(row.get('idade'))} | Maturacao: {_clean_text(row.get('fis_estado_maturacional')) or '--'}")
    _render_score_pills(row)


def _build_radar(selected_df: pd.DataFrame) -> go.Figure:
    axes = ["ANT", "FIS", "TEC", "TAT", "PSI"]
    fig = go.Figure()
    palette = ["#0f766e", "#b54708", "#6941c6", "#b42318"]
    for idx, (_, row) in enumerate(selected_df.iterrows()):
        values = [row.get(axis) for axis in axes]
        values = [None if pd.isna(v) else float(v) for v in values]
        closed_axes = axes + [axes[0]]
        closed_values = values + [values[0]]
        fig.add_trace(
            go.Scatterpolar(
                r=closed_values,
                theta=closed_axes,
                fill="toself",
                mode="lines+markers",
                name=_clean_text(row.get("nome")) or _clean_text(row.get("atleta_id")),
                line=dict(color=palette[idx % len(palette)], width=3),
                marker=dict(size=7, color=palette[idx % len(palette)]),
                fillcolor=f"rgba{(*tuple(int(palette[idx % len(palette)][i:i+2], 16) for i in (1, 3, 5)), 0.14)}",
                hovertemplate="%{theta}: %{r:.0f}<extra>%{fullData.name}</extra>",
            )
        )
    fig.update_layout(
        margin=dict(l=30, r=30, t=40, b=20),
        height=520,
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
                tickfont=dict(size=12),
            ),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="left", x=0.0),
    )
    return fig


def _reference_row(group_df: pd.DataFrame, label: str) -> dict | None:
    if group_df is None or group_df.empty:
        return None
    row = {
        "atleta_id": f"REF-{label}",
        "nome": label,
        "genero": "",
        "selecao": "",
        "escalao": "",
        "posicao": "",
        "idade": None,
        "fis_estado_maturacional": "",
        "ANT": group_df["ANT"].dropna().median() if "ANT" in group_df.columns else None,
        "FIS": group_df["FIS"].dropna().median() if "FIS" in group_df.columns else None,
        "TEC": group_df["TEC"].dropna().median() if "TEC" in group_df.columns else None,
        "TAT": group_df["TAT"].dropna().median() if "TAT" in group_df.columns else None,
        "PSI": group_df["PSI"].dropna().median() if "PSI" in group_df.columns else None,
        "n_referencia": int(len(group_df)),
        "estado_atual": "",
    }
    return row


def _build_reference_profiles(base_row: pd.Series, candidates_df: pd.DataFrame, selected_reference_keys: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    base_gender = _clean_text(base_row.get("genero"))
    base_position = _clean_text(base_row.get("posicao"))
    base_scale = _clean_text(base_row.get("escalao"))
    base_selection = _clean_text(base_row.get("selecao"))
    candidate_pool = candidates_df.copy()
    if base_gender:
        candidate_pool = candidate_pool[candidate_pool["genero"].astype(str) == base_gender].copy()

    reference_definitions = [
        ("Mesma posicao", candidate_pool[candidate_pool["posicao"].astype(str) == base_position].copy() if base_position else pd.DataFrame()),
        ("Mesmo escalao", candidate_pool[candidate_pool["escalao"].astype(str) == base_scale].copy() if base_scale else pd.DataFrame()),
        ("Mesma selecao", candidate_pool[candidate_pool["selecao"].astype(str) == base_selection].copy() if base_selection else pd.DataFrame()),
        ("Internacionais", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Internacional"].copy()),
        ("Estagio Selecao Nacional", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Estágio Seleção Nacional"].copy()),
        ("Selecao Distrital", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Seleção Distrital"].copy()),
        ("Processo Selecao", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Processo Seleção"].copy()),
        ("Referenciadas", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Referenciado"].copy()),
        ("Observadas", candidate_pool[candidate_pool["estado_atual"].astype(str) == "Observado"].copy()),
    ]

    for label, group_df in reference_definitions:
        if label not in selected_reference_keys:
            continue
        group_df = group_df[group_df["atleta_id"].astype(str) != _clean_text(base_row.get("atleta_id"))].copy()
        ref_row = _reference_row(group_df, label)
        if ref_row is not None and ref_row["n_referencia"] > 0:
            rows.append(ref_row)

    return pd.DataFrame(rows)


profiles_df = _load_profiles()
if profiles_df.empty:
    st.info("Ainda nao existem atletas suficientes para comparar perfis.")
    st.stop()

working_df = profiles_df.copy()
working_df["label"] = working_df.apply(_athlete_label, axis=1)

st.markdown("**Filtros**")
only_active = st.checkbox("So atletas ativas", value=True)
selecoes = sorted(value for value in working_df["selecao"].dropna().astype(str).str.strip().unique() if value)
escaloes = sorted(value for value in working_df["escalao"].dropna().astype(str).str.strip().unique() if value)
posicoes = sorted(value for value in working_df["posicao"].dropna().astype(str).str.strip().unique() if value)
generos = sorted(value for value in working_df["genero"].dropna().astype(str).str.strip().unique() if value)

filter_cols = st.columns([1.1, 1.1, 1.1, 1.0])
with filter_cols[0]:
    selecao_filter = st.multiselect("Selecao", selecoes)
with filter_cols[1]:
    escalao_filter = st.multiselect("Escalao", escaloes)
with filter_cols[2]:
    posicao_filter = st.multiselect("Posicao", posicoes)
with filter_cols[3]:
    genero_filter = st.multiselect("Genero", generos)

if only_active and "ativo" in working_df.columns:
    working_df = working_df[working_df["ativo"].fillna(False).astype(bool)].copy()
if selecao_filter:
    working_df = working_df[working_df["selecao"].astype(str).isin(selecao_filter)].copy()
if escalao_filter:
    working_df = working_df[working_df["escalao"].astype(str).isin(escalao_filter)].copy()
if posicao_filter:
    working_df = working_df[working_df["posicao"].astype(str).isin(posicao_filter)].copy()
if genero_filter:
    working_df = working_df[working_df["genero"].astype(str).isin(genero_filter)].copy()

working_df = working_df.sort_values(["nome", "atleta_id"], na_position="last").reset_index(drop=True)

st.caption(f"Atletas disponiveis para comparacao: {len(working_df)}")

comparison_mode = st.radio(
    "Modo de comparacao",
    options=["Perfil vs Perfil", "Perfil vs Referencial"],
    horizontal=True,
)

label_to_id = {row["label"]: row["atleta_id"] for _, row in working_df.iterrows()}
if comparison_mode == "Perfil vs Perfil":
    st.caption("Modo direto para comparar 2 a 4 atletas entre si e encontrar perfis semelhantes.")
    with st.container(border=True):
        default_labels = list(label_to_id.keys())[: min(2, len(label_to_id))]
        selected_labels = st.multiselect(
            "Selecionar atletas para comparar",
            options=list(label_to_id.keys()),
            default=default_labels,
            max_selections=4,
            placeholder="Escolha entre 2 e 4 atletas",
        )

        selected_ids = [label_to_id[label] for label in selected_labels if label in label_to_id]
        selected_df = working_df[working_df["atleta_id"].astype(str).isin(selected_ids)].copy()
        selected_df["_selection_order"] = selected_df["atleta_id"].astype(str).map({atleta_id: idx for idx, atleta_id in enumerate(selected_ids)})
        selected_df = selected_df.sort_values("_selection_order").drop(columns="_selection_order")

        if len(selected_df) < 2:
            st.info("Selecione pelo menos 2 atletas para abrir a comparacao de perfis.")
        else:
            st.caption(f"Perfis selecionados: {len(selected_df)}")
            header_cols = st.columns(len(selected_df))
            for idx, (_, row) in enumerate(selected_df.iterrows()):
                with header_cols[idx]:
                    _render_athlete_card(row)

            st.markdown("---")
            st.markdown("**Radar Comparativo de Perfil**")
            st.caption("Os 5 eixos representam o enquadramento relativo da atleta em ANT, FIS, TEC, TAT e PSI.")
            st.plotly_chart(
                _build_radar(selected_df),
                use_container_width=True,
                config={"displayModeBar": False},
                key="profile_comparison_radar",
            )

            st.markdown("**Tabela Comparativa**")
            table_df = selected_df[
                [
                    "atleta_id",
                    "nome",
                    "genero",
                    "selecao",
                    "escalao",
                    "posicao",
                    "idade",
                    "fis_estado_maturacional",
                    "ANT",
                    "FIS",
                    "TEC",
                    "TAT",
                    "PSI",
                ]
            ].rename(
                columns={
                    "atleta_id": "ID",
                    "nome": "Nome",
                    "genero": "Genero",
                    "selecao": "Selecao",
                    "escalao": "Escalao",
                    "posicao": "Posicao",
                    "idade": "Idade",
                    "fis_estado_maturacional": "Estado maturacional",
                }
            )
            for score_col in ["ANT", "FIS", "TEC", "TAT", "PSI"]:
                table_df[score_col] = table_df[score_col].map(lambda value: None if pd.isna(value) else int(round(float(value))))
            st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("**Perfis Semelhantes**")
            st.caption("A app procura atletas com perfil mais proximo, usando ANT | FIS | TEC | TAT | PSI e privilegiando genero, posicao e escalao.")

            similarity_cols = st.columns([2.2, 0.8])
            with similarity_cols[0]:
                base_profile_label = st.selectbox(
                    "Atleta base",
                    options=list(label_to_id.keys()),
                    index=0 if selected_labels else None,
                    placeholder="Escolha a atleta-base",
                )
            with similarity_cols[1]:
                top_n = st.selectbox("N resultados", options=[3, 5, 8], index=1)

            if base_profile_label:
                base_profile_id = label_to_id[base_profile_label]
                base_row = working_df[working_df["atleta_id"].astype(str) == str(base_profile_id)].head(1)
                if not base_row.empty:
                    similar_df = _find_similar_profiles(base_row.iloc[0], working_df, top_n=top_n)
                    if similar_df.empty:
                        st.info("Ainda nao existem dados suficientes para encontrar perfis semelhantes com seguranca.")
                    else:
                        st.dataframe(similar_df, use_container_width=True, hide_index=True)
else:
    st.caption("Modo de enquadramento para comparar uma atleta com a mediana de grupos de referencia.")
    with st.container(border=True):
        st.markdown("**Comparacao com Referenciais**")
        reference_cols = st.columns([2.0, 2.4])
        with reference_cols[0]:
            base_reference_label = st.selectbox(
                "Atleta base para referenciais",
                options=list(label_to_id.keys()),
                index=0 if label_to_id else None,
                placeholder="Escolha a atleta-base",
                key="reference_base_athlete",
            )
        with reference_cols[1]:
            reference_mode = st.selectbox(
                "Tipo de referencial",
                options=["Contexto da atleta", "Estado competitivo"],
                index=0,
                key="reference_mode_select",
            )
            if reference_mode == "Contexto da atleta":
                selected_reference_key = st.selectbox(
                    "Grupo de referencia",
                    options=[
                        "Mesma posicao",
                        "Mesmo escalao",
                        "Mesma selecao",
                    ],
                    index=1,
                    key="reference_group_select_context",
                )
            else:
                selected_reference_key = st.selectbox(
                    "Grupo de referencia",
                    options=[
                        "Internacionais",
                        "Estagio Selecao Nacional",
                        "Selecao Distrital",
                        "Processo Selecao",
                        "Referenciadas",
                        "Observadas",
                    ],
                    index=2,
                    key="reference_group_select_status",
                )

        if base_reference_label and selected_reference_key:
            base_reference_id = label_to_id[base_reference_label]
            base_reference_row = working_df[working_df["atleta_id"].astype(str) == str(base_reference_id)].head(1)
            if not base_reference_row.empty:
                preview_cols = st.columns([1.4, 2.2])
                with preview_cols[0]:
                    _render_athlete_card(base_reference_row.iloc[0])
                with preview_cols[1]:
                    st.markdown("**Grupo selecionado**")
                    st.caption("O referencial usa a mediana do grupo para ANT, FIS, TEC, TAT e PSI, sempre dentro do mesmo genero.")
                    st.caption(f"Tipo: {reference_mode}")
                    st.write(selected_reference_key)

                reference_df = _build_reference_profiles(base_reference_row.iloc[0], working_df, [selected_reference_key])
                if reference_df.empty:
                    st.info("Nao existem dados suficientes para montar o grupo de referencia selecionado.")
                else:
                    comparison_reference_df = pd.concat(
                        [base_reference_row.copy(), reference_df],
                        ignore_index=True,
                        sort=False,
                    )
                    st.plotly_chart(
                        _build_radar(comparison_reference_df),
                        use_container_width=True,
                        config={"displayModeBar": False},
                        key="reference_comparison_radar",
                    )
                    reference_table = comparison_reference_df[
                        ["nome", "ANT", "FIS", "TEC", "TAT", "PSI", "n_referencia"]
                    ].rename(
                        columns={
                            "nome": "Perfil",
                            "n_referencia": "N referencia",
                        }
                    )
                    for score_col in ["ANT", "FIS", "TEC", "TAT", "PSI"]:
                        reference_table[score_col] = reference_table[score_col].map(
                            lambda value: None if pd.isna(value) else int(round(float(value)))
                        )
                    st.dataframe(reference_table, use_container_width=True, hide_index=True)
