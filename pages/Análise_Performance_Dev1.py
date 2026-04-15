from itertools import combinations

import matplotlib.pyplot as plt
import mplsoccer as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial import ConvexHull, QhullError

from fpf_modules.selections import load_selection_options
from fpf_modules.supabase_manager import get_supabase_client, initialize_schema, read_table


SELECTION_OPTIONS = load_selection_options()
PHASE_OPTIONS = ["1P", "2P"]
VELOCIDADE_SENSACIONAL_KM_H = 25.0
FIELD_VIEW_OPTIONS = [
    "Movimento dos Jogadores",
    "Convex Hull",
    "Distancia entre Jogadores",
    "Movimento Relativo de Jogadores ao Longo do Tempo",
    "Aceleração / Desaceleração",
    "Velocidade",
]


st.set_page_config(page_title="FPF | Analise Posicional", layout="wide")


@st.cache_data(ttl=120)
def load_sessions_data():
    initialize_schema()
    frames = []

    reports = read_table(
        "session_reports",
        columns="session_sk,data,selecao,genero,contexto,jogo,session_fingerprint",
    )
    if reports is not None and not reports.empty:
        frames.append(reports)

    perf = read_table(
        "performance_metrics",
        columns="session_sk,data,selecao,genero,contexto,jogo",
    )
    if perf is not None and not perf.empty:
        frames.append(perf)

    if not frames:
        return pd.DataFrame(columns=["session_sk", "data", "selecao", "genero", "contexto", "jogo"])

    sessions = pd.concat(frames, ignore_index=True, sort=False)
    for col in ["selecao", "genero", "contexto", "jogo"]:
        if col not in sessions.columns:
            sessions[col] = ""
        sessions[col] = sessions[col].fillna("").astype(str).str.strip()

    sessions["session_sk"] = pd.to_numeric(sessions.get("session_sk"), errors="coerce")
    sessions = sessions[sessions["session_sk"].notna()].copy()
    sessions["session_sk"] = sessions["session_sk"].astype(int)
    sessions["data"] = pd.to_datetime(sessions.get("data"), errors="coerce")
    return (
        sessions.drop_duplicates("session_sk", keep="last")
        .sort_values(["data", "session_sk"], ascending=[False, False], na_position="last")
        .reset_index(drop=True)
    )


def _phase_filter(fase: str) -> tuple[str, int | str]:
    phase_id = {"Warm-Up": 0, "1P": 1, "2P": 2}.get(str(fase), None)
    if phase_id is not None:
        return "phase_id", phase_id
    return "fase", str(fase)


@st.cache_data(ttl=120)
def load_phase_time_bounds(session_sk: int, fase: str) -> tuple[float | None, float | None]:
    initialize_schema()
    phase_id = {"Warm-Up": 0, "1P": 1, "2P": 2}.get(str(fase), None)
    filters = {"session_sk": int(session_sk)}
    if phase_id is not None:
        filters["phase_id"] = phase_id
    else:
        filters["fase"] = str(fase)

    try:
        metrics = read_table(
            "performance_metrics",
            filters=filters,
            columns="session_sk,phase_id,fase,duracao_min",
        )
        if metrics is None or metrics.empty or "duracao_min" not in metrics.columns:
            return None, None
        durations = pd.to_numeric(metrics["duracao_min"], errors="coerce").dropna()
        durations = durations[durations > 0]
        if durations.empty:
            return None, None
        return 0.0, float(durations.max() * 60.0)
    except Exception as exc:
        st.error(f"Error reading phase duration from performance_metrics: {exc}")
        return None, None


@st.cache_data(ttl=120)
def load_tracking_for_session_phase(
    session_sk: int,
    fase: str,
    start_s: float,
    end_s: float,
    page_size: int = 1000,
):
    initialize_schema()
    client = get_supabase_client()
    filter_col, filter_value = _phase_filter(fase)
    columns = "session_sk,atleta_id,fase,phase_id,time_evento_s,x_utm,y_utm,x_norm,y_norm"
    rows = []
    try:
        athlete_response = (
            client.table("athlete_session")
            .select("athlete_sk")
            .eq("session_sk", int(session_sk))
            .execute()
        )
        athlete_sks = sorted(
            {
                int(value)
                for value in pd.to_numeric(
                    pd.Series([row.get("athlete_sk") for row in (athlete_response.data or [])]),
                    errors="coerce",
                ).dropna().tolist()
            }
        )
        if not athlete_sks:
            return pd.DataFrame(columns=["session_sk", "fase", "time_evento_s", "x_tr", "y_tr", "atleta_id"])

        for athlete_sk in athlete_sks:
            offset = 0
            while True:
                query = (
                    client.table("samples")
                    .select(columns)
                    .eq("session_sk", int(session_sk))
                    .eq("athlete_sk", int(athlete_sk))
                    .eq(filter_col, filter_value)
                    .gte("time_evento_s", float(start_s))
                    .lte("time_evento_s", float(end_s))
                    .order("time_evento_s")
                    .range(offset, offset + page_size - 1)
                )
                response = query.execute()
                batch = response.data or []
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
    except Exception as exc:
        st.error(f"Error reading from samples: {exc}")
        return pd.DataFrame(columns=["session_sk", "fase", "time_evento_s", "x_tr", "y_tr", "atleta_id"])

    if not rows:
        try:
            offset = 0
            while True:
                response = (
                    client.table("samples")
                    .select(columns)
                    .eq("session_sk", int(session_sk))
                    .eq(filter_col, filter_value)
                    .gte("time_evento_s", float(start_s))
                    .lte("time_evento_s", float(end_s))
                    .order("time_evento_s")
                    .range(offset, offset + page_size - 1)
                    .execute()
                )
                batch = response.data or []
                if not batch:
                    break
                rows.extend(batch)
                if len(batch) < page_size:
                    break
                offset += page_size
        except Exception as exc:
            st.error(f"Error reading from samples: {exc}")
            return pd.DataFrame(columns=["session_sk", "fase", "time_evento_s", "x_tr", "y_tr", "atleta_id"])

    samples = pd.DataFrame(rows)
    if samples is None or samples.empty:
        return pd.DataFrame(columns=["session_sk", "fase", "time_evento_s", "x_tr", "y_tr", "atleta_id"])

    samples["session_sk"] = pd.to_numeric(samples.get("session_sk"), errors="coerce").astype("Int64")
    samples["time_evento_s"] = pd.to_numeric(samples.get("time_evento_s"), errors="coerce")
    samples["atleta_id"] = samples.get("atleta_id", "").astype(str)
    samples["fase"] = samples.get("fase", "").astype(str)

    for col in ["x_utm", "y_utm", "x_norm", "y_norm"]:
        if col not in samples.columns:
            samples[col] = np.nan
        samples[col] = pd.to_numeric(samples[col], errors="coerce")

    norm_valid = (
        samples["x_norm"].notna()
        & samples["y_norm"].notna()
        & samples["x_norm"].between(0, 120)
        & samples["y_norm"].between(0, 80)
    )
    x_norm_unique = samples.loc[norm_valid, "x_norm"].round(2).nunique()
    y_norm_unique = samples.loc[norm_valid, "y_norm"].round(2).nunique()
    if norm_valid.any() and (x_norm_unique <= 3 or y_norm_unique <= 3):
        st.error(
            "Coordenadas normalizadas sem variacao suficiente. "
            "A sessao deve ser republicada para recalcular x_norm/y_norm a partir de x_utm/y_utm."
        )

    samples["x_tr"] = samples["x_norm"]
    samples["y_tr"] = samples["y_norm"]

    return samples[["session_sk", "fase", "time_evento_s", "x_tr", "y_tr", "atleta_id"]].copy()


@st.cache_data
def calcular_compactacao(tracking_df: pd.DataFrame) -> pd.DataFrame:
    if "time_evento_s" not in tracking_df.columns or tracking_df.empty:
        return pd.DataFrame()

    frame_data = tracking_df.groupby("time_evento_s").agg(
        x_min=("x_tr", "min"),
        x_max=("x_tr", "max"),
        y_min=("y_tr", "min"),
        y_max=("y_tr", "max"),
    )
    frame_data["comp_vertical"] = (frame_data["x_max"] - frame_data["x_min"]).round(2)
    frame_data["comp_horizontal"] = (frame_data["y_max"] - frame_data["y_min"]).round(2)
    return frame_data.drop(columns=["x_min", "x_max", "y_min", "y_max"]).reset_index()


def converter_para_relogio_fpf(segundos_totais: float) -> str:
    minutos = int(segundos_totais // 60)
    segundos = int(segundos_totais % 60)
    frame = int(round((segundos_totais % 1) * 10))
    if frame == 10:
        frame = 0
        segundos += 1
    return f"{minutos:02d}:{segundos:02d}.{frame}"


def format_session_label(row: pd.Series) -> str:
    data_value = pd.to_datetime(row.get("data"), errors="coerce")
    data_label = data_value.strftime("%d/%m/%Y") if pd.notna(data_value) else "Sem data"
    contexto = str(row.get("contexto") or "").strip() or "Sessao"
    jogo = str(row.get("jogo") or "").strip()
    if jogo:
        return f"{data_label} | {jogo}"
    return f"{data_label} | {contexto}"


def build_snapshot(df_fase: pd.DataFrame, momento: float) -> pd.DataFrame:
    if momento is None or df_fase.empty:
        return pd.DataFrame()
    return df_fase[
        (df_fase["time_evento_s"] == momento)
        & df_fase["x_tr"].notna()
        & df_fase["y_tr"].notna()
    ].copy()


def first_collective_timestamp(df_fase: pd.DataFrame, min_players: int = 10):
    if df_fase.empty or "time_evento_s" not in df_fase.columns:
        return None
    valid_df = df_fase[df_fase["x_tr"].notna() & df_fase["y_tr"].notna()].copy()
    if valid_df.empty:
        return None
    counts = valid_df.groupby("time_evento_s")["atleta_id"].nunique().sort_index()
    enough = counts[counts >= int(min_players)]
    return enough.index[0] if not enough.empty else counts.idxmax()


def _frame_player_count(df_fase: pd.DataFrame, momento: float) -> int:
    if df_fase.empty or momento is None:
        return 0
    return int(
        df_fase.loc[
            (df_fase["time_evento_s"] == momento)
            & df_fase["x_tr"].notna()
            & df_fase["y_tr"].notna(),
            "atleta_id",
        ].nunique()
    )


def _frame_status_text(n_players: int, expected_players: int = 10) -> tuple[str, str]:
    if n_players == expected_players:
        return "OK", f"{n_players}/{expected_players} jogadores de campo visiveis."
    if n_players > expected_players:
        return "POSSIVEL_SUBSTITUICAO", (
            f"{n_players}/{expected_players} jogadores visiveis. "
            "Possivel substituicao ou overlap GPS neste frame."
        )
    return "FRAME_INCOMPLETO", (
        f"{n_players}/{expected_players} jogadores visiveis. "
        "Frame incompleto: falha GPS, arranque desalinhado ou atleta sem coordenada valida."
    )


def render_positional_footer(
    df_fase: pd.DataFrame,
    *,
    momento: float | None = None,
    timestamps_intervalo: list[float] | None = None,
    expected_players: int = 10,
) -> None:
    st.markdown("---")
    footer_cols = st.columns([1.2, 2.8])

    with footer_cols[0]:
        if momento is not None:
            n_players = _frame_player_count(df_fase, momento)
            status, message = _frame_status_text(n_players, expected_players)
            st.markdown("**Rodape do frame**")
            st.caption(f"Momento: {converter_para_relogio_fpf(float(momento))}")
            if status == "OK":
                st.success(message)
            elif status == "POSSIVEL_SUBSTITUICAO":
                st.warning(message)
            else:
                st.error(message)
        else:
            st.markdown("**Rodape do intervalo**")
            st.caption("O grafico animado usa a timeline interna do Plotly.")

    with footer_cols[1]:
        valid_df = df_fase[df_fase["x_tr"].notna() & df_fase["y_tr"].notna()].copy()
        if timestamps_intervalo:
            valid_df = valid_df[valid_df["time_evento_s"].isin(timestamps_intervalo)].copy()

        if valid_df.empty:
            st.error("Sem frames com coordenadas validas neste intervalo.")
            return

        counts = valid_df.groupby("time_evento_s")["atleta_id"].nunique()
        total_frames = int(len(counts))
        ok_frames = int((counts == expected_players).sum())
        substitution_frames = int((counts > expected_players).sum())
        incomplete_frames = int((counts < expected_players).sum())

        st.markdown("**Resumo da fase/intervalo**")
        st.caption(
            f"Frames analisados: {total_frames} | "
            f"OK: {ok_frames} ({ok_frames / max(total_frames, 1) * 100:.1f}%) | "
            f"Possivel substituicao: {substitution_frames} ({substitution_frames / max(total_frames, 1) * 100:.1f}%) | "
            f"Incompletos: {incomplete_frames} ({incomplete_frames / max(total_frames, 1) * 100:.1f}%)"
        )


def draw_pitch(snapshot: pd.DataFrame):
    pitch = mpl.Pitch(
        pitch_type="statsbomb",
        pitch_color="#22312b",
        line_color="#c7d5cc",
    )
    fig, ax = pitch.draw(figsize=(10, 7))
    if snapshot.empty:
        return pitch, fig, ax

    pitch.scatter(
        snapshot["x_tr"],
        snapshot["y_tr"],
        s=400,
        c="#E30613",
        edgecolors="white",
        linewidth=1,
        alpha=0.9,
        ax=ax,
    )

    for _, row in snapshot.iterrows():
        pitch.annotate(
            str(row["atleta_id"]),
            (row["x_tr"], row["y_tr"]),
            ax=ax,
            color="white",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="center",
        )
    return pitch, fig, ax


def get_convex_hull(snapshot: pd.DataFrame):
    if snapshot.empty or len(snapshot) < 3:
        return None, None

    points = snapshot[["x_tr", "y_tr"]].dropna().drop_duplicates().to_numpy()
    if len(points) < 3:
        return None, None

    try:
        hull = ConvexHull(points)
    except QhullError:
        return None, None

    hull_points = points[hull.vertices]
    return [hull_points], float(hull.volume)


@st.cache_data
def calcular_aceleracao_desaceleracao(tracking_df: pd.DataFrame) -> pd.DataFrame:
    if tracking_df.empty:
        return tracking_df.copy()

    df = tracking_df.sort_values(["atleta_id", "time_evento_s"]).copy()
    grouped = df.groupby("atleta_id", sort=False)
    df["delta_t"] = grouped["time_evento_s"].diff()
    df["delta_x"] = grouped["x_tr"].diff()
    df["delta_y"] = grouped["y_tr"].diff()
    df["distancia_m"] = np.hypot(df["delta_x"], df["delta_y"])
    df["velocidade_m_s"] = np.where(df["delta_t"] > 0, df["distancia_m"] / df["delta_t"], np.nan)
    df["velocidade_m_s_suave"] = grouped["velocidade_m_s"].transform(
        lambda serie: serie.rolling(window=3, min_periods=1).mean()
    )
    df["velocidade_km_h"] = df["velocidade_m_s_suave"] * 3.6
    df["aceleracao_m_s2"] = grouped["velocidade_m_s_suave"].diff() / df["delta_t"]
    df["aceleracao_m_s2"] = df["aceleracao_m_s2"].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["aceleracao_pos_m_s2"] = df["aceleracao_m_s2"].clip(lower=0)
    df["desaceleracao_m_s2"] = df["aceleracao_m_s2"].clip(upper=0)
    return df.drop(columns=["delta_t", "delta_x", "delta_y", "distancia_m"], errors="ignore")


def draw_animated_tracking(
    df_intervalo: pd.DataFrame,
    timestamps_intervalo: list[float],
    pares_selecionados: list[str] | None = None,
    show_convex_hull: bool = False,
    movimento_relativo_jogadores: tuple[str, str] | None = None,
    janela_segundos: int = 1,
    show_aceleracao: bool = False,
    show_velocidade: bool = False,
) -> go.Figure:
    if df_intervalo.empty or not timestamps_intervalo:
        return go.Figure()

    frame_duration_ms = 100
    if len(timestamps_intervalo) > 1:
        frame_duration_ms = int(max(50, min(float(np.median(np.diff(timestamps_intervalo))) * 1000, 1000)))

    def frame_traces(momento: float) -> list[go.Scatter]:
        frame_df = df_intervalo.loc[
            (df_intervalo["time_evento_s"] == momento)
            & df_intervalo["x_tr"].notna()
            & df_intervalo["y_tr"].notna()
        ].copy()
        if show_aceleracao:
            hover_text = frame_df.apply(
                lambda row: (
                    f"Atleta {row['atleta_id']}<br>"
                    f"Aceleração: +{float(row.get('aceleracao_pos_m_s2', 0)):.2f} m/s²<br>"
                    f"Desaceleração: {float(row.get('desaceleracao_m_s2', 0)):.2f} m/s²"
                ),
                axis=1,
            )
            marker_size = 30
            trace_text = frame_df["atleta_id"].astype(str)
            marker_color = "rgba(227, 6, 19, 0.15)"
            marker_line_color = "rgba(255, 255, 255, 0.85)"
        elif show_velocidade:
            hover_text = frame_df.apply(
                lambda row: (
                    f"Atleta {row['atleta_id']}<br>"
                    f"Velocidade: {float(row.get('velocidade_m_s_suave', 0)):.2f} m/s<br>"
                    f"Velocidade: {float(row.get('velocidade_km_h', 0)):.1f} km/h"
                ),
                axis=1,
            )
            marker_size = 30
            trace_text = frame_df["atleta_id"].astype(str)
            marker_color = "rgba(227, 6, 19, 0.15)"
            marker_line_color = "rgba(255, 255, 255, 0.85)"
        else:
            hover_text = frame_df.apply(lambda row: f"Atleta {row['atleta_id']}<br>x={row['x_tr']:.1f}<br>y={row['y_tr']:.1f}", axis=1)
            marker_size = 24
            trace_text = frame_df["atleta_id"].astype(str)
            marker_color = "#E30613"
            marker_line_color = "white"

        traces = [
            go.Scatter(
                x=frame_df["x_tr"],
                y=frame_df["y_tr"],
                mode="markers+text",
                text=trace_text,
                textposition="middle center",
                textfont={"color": "white", "size": 10},
                marker={"size": marker_size, "color": marker_color, "line": {"color": marker_line_color, "width": 1}},
                hovertext=hover_text,
                hoverinfo="text",
            )
        ]

        if show_aceleracao:
            traces.extend(
                [
                    go.Scatter(
                        x=frame_df["x_tr"],
                        y=frame_df["y_tr"] - 3,
                        mode="text",
                        text=frame_df["aceleracao_pos_m_s2"].map(lambda valor: f"+{float(valor):.2f}"),
                        textfont={"color": "#39FF14", "size": 13},
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=frame_df["x_tr"],
                        y=frame_df["y_tr"] + 3,
                        mode="text",
                        text=frame_df["desaceleracao_m_s2"].map(lambda valor: f"{float(valor):.2f}"),
                        textfont={"color": "#FF1744", "size": 13},
                        hoverinfo="skip",
                    ),
                ]
            )

        if show_velocidade:
            velocidade_viva = frame_df["velocidade_km_h"].fillna(0) >= VELOCIDADE_SENSACIONAL_KM_H
            velocidade_ms_color = velocidade_viva.map(lambda acima_limiar: "#FF1744" if acima_limiar else "#00E5FF")
            velocidade_kmh_color = velocidade_viva.map(lambda acima_limiar: "#FF1744" if acima_limiar else "#FFA500")
            traces.extend(
                [
                    go.Scatter(
                        x=frame_df["x_tr"],
                        y=frame_df["y_tr"] - 3,
                        mode="text",
                        text=frame_df["velocidade_m_s_suave"].fillna(0).map(lambda valor: f"{float(valor):.2f} m/s"),
                        textfont={"color": velocidade_ms_color, "size": 13},
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=frame_df["x_tr"],
                        y=frame_df["y_tr"] + 3,
                        mode="text",
                        text=frame_df["velocidade_km_h"].fillna(0).map(lambda valor: f"{float(valor):.1f} km/h"),
                        textfont={"color": velocidade_kmh_color, "size": 13},
                        hoverinfo="skip",
                    ),
                ]
            )

        if show_convex_hull and len(frame_df) >= 3:
            points = frame_df[["x_tr", "y_tr"]].dropna().drop_duplicates().to_numpy()
            if len(points) >= 3:
                try:
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    hull_points = np.vstack([hull_points, hull_points[0]])
                    traces.append(
                        go.Scatter(
                            x=hull_points[:, 0],
                            y=hull_points[:, 1],
                            mode="lines",
                            fill="toself",
                            fillcolor="rgba(227, 6, 19, 0.25)",
                            line={"color": "#E30613", "width": 3},
                            hoverinfo="skip",
                        )
                    )
                except QhullError:
                    pass

        if movimento_relativo_jogadores:
            jogador_1_id, jogador_2_id = movimento_relativo_jogadores
            idx_tempo = timestamps_intervalo.index(momento)
            janela_frames = max(1, int(janela_segundos) * 10)
            janela_timestamps = timestamps_intervalo[max(0, idx_tempo - janela_frames + 1): idx_tempo + 1]
            trilho_df = df_intervalo.loc[
                df_intervalo["time_evento_s"].isin(janela_timestamps)
                & df_intervalo["atleta_id"].astype(str).isin([str(jogador_1_id), str(jogador_2_id)])
                & df_intervalo["x_tr"].notna()
                & df_intervalo["y_tr"].notna()
            ].copy()
            jogador_1_trilho = trilho_df.loc[trilho_df["atleta_id"].astype(str) == str(jogador_1_id)]
            jogador_2_trilho = trilho_df.loc[trilho_df["atleta_id"].astype(str) == str(jogador_2_id)]

            if not jogador_1_trilho.empty:
                traces.append(
                    go.Scatter(
                        x=jogador_1_trilho["x_tr"],
                        y=jogador_1_trilho["y_tr"],
                        mode="lines",
                        line={"color": "#2ECC71", "width": 4},
                        hoverinfo="skip",
                    )
                )
            if not jogador_2_trilho.empty:
                traces.append(
                    go.Scatter(
                        x=jogador_2_trilho["x_tr"],
                        y=jogador_2_trilho["y_tr"],
                        mode="lines",
                        line={"color": "#F2F2F2", "width": 4},
                        hoverinfo="skip",
                    )
                )

            jogador_1 = frame_df.loc[frame_df["atleta_id"].astype(str) == str(jogador_1_id)]
            jogador_2 = frame_df.loc[frame_df["atleta_id"].astype(str) == str(jogador_2_id)]
            if not jogador_1.empty and not jogador_2.empty:
                x1, y1 = jogador_1[["x_tr", "y_tr"]].iloc[0]
                x2, y2 = jogador_2[["x_tr", "y_tr"]].iloc[0]
                distancia_rel = float(np.hypot(x2 - x1, y2 - y1))
                traces.extend(
                    [
                        go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="lines",
                            line={"color": "#FFC857", "width": 4},
                            hoverinfo="skip",
                        ),
                        go.Scatter(
                            x=[x1, x2],
                            y=[y1, y2],
                            mode="markers",
                            marker={"size": 30, "color": ["#2ECC71", "#F2F2F2"], "line": {"color": "black", "width": 2}},
                            text=[jogador_1_id, jogador_2_id],
                            hovertemplate="Atleta %{text}<extra></extra>",
                        ),
                        go.Scatter(
                            x=[(x1 + x2) / 2],
                            y=[(y1 + y2) / 2],
                            mode="text",
                            text=[f"{distancia_rel:.2f} m"],
                            textfont={"color": "#FFC857", "size": 14},
                            hoverinfo="skip",
                        ),
                    ]
                )

        for par in pares_selecionados or []:
            jogador_1_id, jogador_2_id = par.split(" - ")
            jogador_1 = frame_df.loc[frame_df["atleta_id"].astype(str) == str(jogador_1_id)]
            jogador_2 = frame_df.loc[frame_df["atleta_id"].astype(str) == str(jogador_2_id)]
            if jogador_1.empty or jogador_2.empty:
                continue

            x1, y1 = jogador_1[["x_tr", "y_tr"]].iloc[0]
            x2, y2 = jogador_2[["x_tr", "y_tr"]].iloc[0]
            distancia_par = float(np.hypot(x2 - x1, y2 - y1))
            traces.extend(
                [
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line={"color": "#FFD166", "width": 4},
                        hoverinfo="skip",
                    ),
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="markers",
                        marker={"size": 28, "color": "#2ECC71", "line": {"color": "black", "width": 2}},
                        hovertemplate="Atleta %{text}<extra></extra>",
                        text=[jogador_1_id, jogador_2_id],
                    ),
                    go.Scatter(
                        x=[(x1 + x2) / 2],
                        y=[(y1 + y2) / 2],
                        mode="text",
                        text=[f"{distancia_par:.2f} m"],
                        textfont={"color": "#FFD166", "size": 14},
                        hoverinfo="skip",
                    ),
                ]
            )

        return traces

    first_time = timestamps_intervalo[0]
    fig = go.Figure(data=frame_traces(first_time))
    fig.frames = [
        go.Frame(
            data=frame_traces(momento),
            name=str(momento),
            layout=go.Layout(title_text=f"Momento do Jogo: {converter_para_relogio_fpf(momento)}"),
        )
        for momento in timestamps_intervalo
    ]

    slider_steps = [
        {
            "method": "animate",
            "label": converter_para_relogio_fpf(momento),
            "args": [
                [str(momento)],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for momento in timestamps_intervalo
    ]

    field_shapes = [
        {"type": "rect", "x0": 0, "y0": 0, "x1": 120, "y1": 80, "line": {"color": "#c7d5cc", "width": 2}},
        {"type": "line", "x0": 60, "y0": 0, "x1": 60, "y1": 80, "line": {"color": "#c7d5cc", "width": 1.5}},
        {"type": "circle", "x0": 50, "y0": 30, "x1": 70, "y1": 50, "line": {"color": "#c7d5cc", "width": 1.5}},
        {"type": "rect", "x0": 0, "y0": 18, "x1": 18, "y1": 62, "line": {"color": "#c7d5cc", "width": 1.5}},
        {"type": "rect", "x0": 102, "y0": 18, "x1": 120, "y1": 62, "line": {"color": "#c7d5cc", "width": 1.5}},
        {"type": "rect", "x0": 0, "y0": 30, "x1": 6, "y1": 50, "line": {"color": "#c7d5cc", "width": 1.5}},
        {"type": "rect", "x0": 114, "y0": 30, "x1": 120, "y1": 50, "line": {"color": "#c7d5cc", "width": 1.5}},
    ]

    fig.update_layout(
        title=f"Momento do Jogo: {converter_para_relogio_fpf(first_time)}",
        height=650,
        margin={"l": 10, "r": 10, "t": 85, "b": 10},
        paper_bgcolor="#1e1e1e",
        plot_bgcolor="#22312b",
        showlegend=False,
        shapes=field_shapes,
        xaxis={"range": [0, 120], "showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={
            "range": [80, 0],
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.02,
                "xanchor": "left",
                "y": 0.98,
                "yanchor": "top",
                "pad": {"r": 10, "t": 0},
                "showactive": True,
                "buttons": [
                    {
                        "label": "▶ Play / Ⅱ Pause",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        "args2": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "Momento: "},
                "steps": slider_steps,
                "pad": {"t": 35},
            }
        ],
    )
    return fig


try:
    sessions_df = load_sessions_data()
except Exception as exc:
    st.error(f"Erro ao processar dados: {exc}")
    st.stop()


st.title("Analise Posicional")
st.caption("Analise posicional com filtros centrais e navegacao temporal integrada por baixo do campo.")

with st.expander("Seleção de Dados Analisar:", expanded=True):
    filtros_col1, filtros_col2, filtros_col3 = st.columns(3)

    selection_index = 0 if SELECTION_OPTIONS else None
    selecao = filtros_col1.selectbox("Selecao", options=SELECTION_OPTIONS, index=selection_index)
    contexto = filtros_col2.selectbox("Contexto", options=["Treino", "Jogo"])

    session_source = sessions_df.copy()
    session_source["jogo"] = session_source["jogo"].fillna("").astype(str)
    session_source = session_source.loc[
        (session_source["selecao"] == selecao)
        & (session_source["contexto"] == contexto)
    ].copy()

    if contexto == "Jogo":
        session_source = session_source.loc[session_source["jogo"].str.strip() != ""].copy()
    else:
        session_source = session_source.loc[session_source["jogo"].str.strip() == ""].copy()

    session_source = session_source.sort_values(["data", "session_sk"], ascending=[False, False]).copy()
    session_source["session_label"] = session_source.apply(format_session_label, axis=1)

    session_options = session_source["session_label"].tolist()
    session_lookup = dict(zip(session_source["session_label"], session_source["session_sk"]))
    jogo_ou_treino = filtros_col3.selectbox(
        "Jogo ou Treino",
        options=session_options,
        index=0 if session_options else None,
        key="campo_sessao_selecionada",
    )

selected_session_sk = session_lookup.get(jogo_ou_treino)

st.divider()

if selected_session_sk is None:
    st.info("Seleciona um jogo ou treino para visualizar o campo.")
    st.stop()

nav_col1, nav_col2 = st.columns([2, 1.2])
campo = nav_col1.selectbox(
    "Métrica Posicional",
    options=FIELD_VIEW_OPTIONS,
    index=0,
    key="campo_visualizacao",
)
fase_selected = nav_col2.radio(
    "Fase",
    PHASE_OPTIONS,
    index=0,
    horizontal=True,
)

min_time_s, max_time_s = load_phase_time_bounds(int(selected_session_sk), fase_selected)
if min_time_s is None or max_time_s is None or max_time_s <= min_time_s:
    st.info("A sessao selecionada nao tem dados de tracking para esta fase.")
    st.stop()

timeline_context = f"{selected_session_sk}|{fase_selected}"
if st.session_state.get("timeline_context") != timeline_context:
    st.session_state["timeline_context"] = timeline_context
    st.session_state["selected_time_jogo"] = float(min_time_s)

default_end_s = min(float(max_time_s), float(min_time_s) + 60.0)
time_options = [float(value) for value in range(int(np.floor(min_time_s)), int(np.ceil(max_time_s)) + 1)]
default_interval = (
    float(int(np.floor(min_time_s))),
    float(int(np.floor(default_end_s))),
)
interval_start, interval_end = st.select_slider(
    "Intervalo de análise",
    options=time_options,
    value=default_interval,
    format_func=converter_para_relogio_fpf,
    key=f"tempo_intervalo_jogo_{timeline_context}",
    help="Relógio da fase selecionada. Ex.: 02:40 significa 2 minutos e 40 segundos da 1P/2P.",
)
interval_start = float(interval_start)
interval_end = float(interval_end)
if interval_end <= interval_start:
    st.warning("Escolhe um fim de intervalo superior ao inicio.")
    st.stop()

max_window_s = 300.0
if interval_end - interval_start > max_window_s:
    st.warning(
        "Para evitar timeout, escolhe uma janela ate 5 minutos. "
        "Depois podes mover o intervalo ao longo da fase."
    )
    st.stop()

st.caption(
    "Fase disponivel: "
    f"{converter_para_relogio_fpf(min_time_s)} - {converter_para_relogio_fpf(max_time_s)}"
)
st.caption(
    "Intervalo selecionado: "
    f"{converter_para_relogio_fpf(interval_start)} - {converter_para_relogio_fpf(interval_end)}"
)

with st.spinner("A carregar tracking apenas para o intervalo selecionado..."):
    df_fase = load_tracking_for_session_phase(
        int(selected_session_sk),
        fase_selected,
        interval_start,
        interval_end,
    )

timestamps = sorted(df_fase["time_evento_s"].dropna().unique().tolist()) if not df_fase.empty else []
if not timestamps:
    st.info("A sessao selecionada nao tem dados de tracking neste intervalo.")
    st.stop()

first_valid_time = first_collective_timestamp(df_fase, min_players=10)
default_start = first_valid_time if first_valid_time in timestamps else timestamps[0]

selected_time = st.session_state.get("selected_time_jogo", timestamps[0])
if selected_time not in timestamps:
    selected_time = default_start
st.session_state["selected_time_jogo"] = selected_time

expected_players = int(df_fase["atleta_id"].nunique()) if "atleta_id" in df_fase.columns else 0
frame_players = int(
    df_fase.loc[
        (df_fase["time_evento_s"] == selected_time)
        & df_fase["x_tr"].notna()
        & df_fase["y_tr"].notna(),
        "atleta_id",
    ].nunique()
)
st.caption(
    f"Atletas com dados na fase: {expected_players} | "
    f"Atletas com coordenadas validas neste momento: {frame_players}"
)

if campo == "Movimento dos Jogadores":
    df_intervalo = df_fase.loc[df_fase["time_evento_s"].isin(timestamps)].copy()
    st.plotly_chart(draw_animated_tracking(df_intervalo, timestamps), use_container_width=True)
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo.shape[0]}")
    st.stop()

snapshot = build_snapshot(df_fase, selected_time)
df_compactacao = calcular_compactacao(df_fase)
compactacao_frame = (
    df_compactacao.loc[df_compactacao["time_evento_s"] == selected_time]
    if not df_compactacao.empty
    else pd.DataFrame()
)
df_intervalo = df_fase.loc[df_fase["time_evento_s"].isin(timestamps)].copy()
jogadores_disponiveis = sorted(df_intervalo["atleta_id"].dropna().astype(str).unique().tolist()) if not df_intervalo.empty else []

jogadores_selecionados = []
pares_selecionados = []
distancias_pares = []
jogador_rel_1 = None
jogador_rel_2 = None
tempo_relativo = None
janela_segundos = 1
distancia_rel = None
delta_distancia_rel = None
distancia_media_rel = None
distancia_min_rel = None
distancia_max_rel = None
tendencia_rel = None
serie_distancias_rel = pd.DataFrame()
cores_pares = [
    ("Amarelo", "#FFD166"),
    ("Verde", "#06D6A0"),
    ("Azul Claro", "#4CC9F0"),
    ("Rosa", "#EF476F"),
    ("Laranja", "#F77F00"),
    ("Verde Lima", "#90BE6D"),
]

if campo == "Distancia entre Jogadores" and len(jogadores_disponiveis) >= 2:
    control_col1, control_col2 = st.columns(2)
    jogadores_default = jogadores_disponiveis[:2]
    jogadores_selecionados = control_col1.multiselect(
        "Jogadores",
        options=jogadores_disponiveis,
        default=jogadores_default,
        max_selections=4,
        key="dist_jogadores_selecionados",
    )
    pares_opcoes = [
        f"{jogador_1} - {jogador_2}"
        for jogador_1, jogador_2 in combinations(jogadores_selecionados, 2)
    ]
    pares_selecionados = control_col2.multiselect(
        "Pares",
        options=pares_opcoes,
        default=pares_opcoes[:1],
        key="dist_pares_selecionados",
    )
    st.plotly_chart(
        draw_animated_tracking(df_intervalo, timestamps, pares_selecionados=pares_selecionados),
        use_container_width=True,
    )
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo.shape[0]}")
    st.stop()
elif campo == "Convex Hull":
    st.plotly_chart(
        draw_animated_tracking(df_intervalo, timestamps, show_convex_hull=True),
        use_container_width=True,
    )
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo.shape[0]}")
    st.stop()
elif campo == "Movimento Relativo de Jogadores ao Longo do Tempo" and len(jogadores_disponiveis) >= 2:
    control_col1, control_col2, control_col3 = st.columns(3)
    jogador_rel_1 = control_col1.selectbox(
        "Jogador 1",
        options=jogadores_disponiveis,
        index=0,
        key="mov_rel_jogador_1",
    )
    jogador_rel_2 = control_col2.selectbox(
        "Jogador 2",
        options=jogadores_disponiveis,
        index=1 if len(jogadores_disponiveis) > 1 else 0,
        key="mov_rel_jogador_2",
    )
    janela_segundos = control_col3.selectbox(
        "Janela Temporal",
        options=[1, 2, 3, 5, 10],
        index=0,
        key="mov_rel_janela_segundos",
        format_func=lambda valor: f"{valor}s",
    )
    if jogador_rel_1 == jogador_rel_2:
        st.warning("Seleciona dois jogadores diferentes para visualizar o movimento relativo.")
        st.stop()

    st.plotly_chart(
        draw_animated_tracking(
            df_intervalo,
            timestamps,
            movimento_relativo_jogadores=(str(jogador_rel_1), str(jogador_rel_2)),
            janela_segundos=int(janela_segundos),
        ),
        use_container_width=True,
    )
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo.shape[0]}")
    st.stop()
elif campo == "Aceleração / Desaceleração":
    df_intervalo_acc = calcular_aceleracao_desaceleracao(df_intervalo)
    st.plotly_chart(
        draw_animated_tracking(df_intervalo_acc, timestamps, show_aceleracao=True),
        use_container_width=True,
    )
    st.caption("Valores em m/s². Aceleração acima do atleta e desaceleração abaixo.")
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo_acc.shape[0]}")
    st.stop()
elif campo == "Velocidade":
    df_intervalo_vel = calcular_aceleracao_desaceleracao(df_intervalo)
    st.plotly_chart(
        draw_animated_tracking(df_intervalo_vel, timestamps, show_velocidade=True),
        use_container_width=True,
    )
    st.caption(
        f"Velocidade em m/s por cima e km/h por baixo. Valores >= {VELOCIDADE_SENSACIONAL_KM_H:.0f} km/h aparecem a vermelho."
    )
    render_positional_footer(df_fase, timestamps_intervalo=timestamps)
    st.divider()
    st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas no intervalo: {df_intervalo_vel.shape[0]}")
    st.stop()
col_map, col_info = st.columns([3, 1])

with col_map:
    plot_placeholder = st.empty()
    slider_placeholder = st.empty()
    graph_placeholder = st.empty()

    timestamps_relativos = []
    tempo_visualizacao = selected_time

    if campo == "Movimento Relativo de Jogadores ao Longo do Tempo":
        timestamps_relativos = timestamps
        tempo_relativo = selected_time
        tempo_visualizacao = tempo_relativo

    snapshot = build_snapshot(df_fase, tempo_visualizacao)
    pitch, fig, ax = draw_pitch(snapshot)
    convex_hull, area = get_convex_hull(snapshot)

    if campo == "Convex Hull" and convex_hull is not None:
        pitch.polygon(convex_hull, ax=ax, edgecolor="#E30613", color="#E30613", alpha=0.3)

    elif campo == "Distancia entre Jogadores" and len(jogadores_selecionados) >= 2:
        coords_jogadores = {}
        for jogador_id in jogadores_selecionados:
            jogador_data = snapshot[snapshot["atleta_id"].astype(str) == str(jogador_id)]
            if not jogador_data.empty:
                coords_jogadores[str(jogador_id)] = tuple(jogador_data[["x_tr", "y_tr"]].iloc[0])

        if coords_jogadores:
            xs_sel = [coords_jogadores[jogador_id][0] for jogador_id in coords_jogadores]
            ys_sel = [coords_jogadores[jogador_id][1] for jogador_id in coords_jogadores]
            pitch.scatter(
                xs_sel,
                ys_sel,
                s=500,
                c="#2ECC71",
                edgecolors="black",
                linewidth=1.5,
                ax=ax,
                zorder=3,
            )

        for idx, par in enumerate(pares_selecionados):
            jogador_1_id, jogador_2_id = par.split(" - ")
            if jogador_1_id not in coords_jogadores or jogador_2_id not in coords_jogadores:
                continue
            x1, y1 = coords_jogadores[jogador_1_id]
            x2, y2 = coords_jogadores[jogador_2_id]
            distancia_par = float(np.hypot(x2 - x1, y2 - y1))
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            nome_cor, cor_par = cores_pares[idx % len(cores_pares)]
            distancias_pares.append({"Par": par, "Cor": nome_cor, "Distancia": round(distancia_par, 2)})
            ax.plot([x1, x2], [y1, y2], color=cor_par, linewidth=2.5, alpha=0.95)
            ax.text(
                x_mid,
                y_mid,
                f"{distancia_par:.2f} m",
                color="black",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor=cor_par, edgecolor="black", alpha=0.95),
            )

    elif (
        campo == "Movimento Relativo de Jogadores ao Longo do Tempo"
        and jogador_rel_1
        and jogador_rel_2
        and jogador_rel_1 != jogador_rel_2
        and tempo_relativo is not None
    ):
        idx_tempo = timestamps_relativos.index(tempo_relativo) if tempo_relativo in timestamps_relativos else len(timestamps_relativos) - 1
        janela_frames = max(1, int(janela_segundos) * 10)
        janela_timestamps = timestamps_relativos[max(0, idx_tempo - janela_frames + 1): idx_tempo + 1]
        trilho_df = df_fase.loc[
            df_fase["time_evento_s"].isin(janela_timestamps)
            & df_fase["atleta_id"].astype(str).isin([str(jogador_rel_1), str(jogador_rel_2)])
            & df_fase["x_tr"].notna()
            & df_fase["y_tr"].notna()
        ].copy()

        jogador_1_trilho = trilho_df[trilho_df["atleta_id"].astype(str) == str(jogador_rel_1)]
        jogador_2_trilho = trilho_df[trilho_df["atleta_id"].astype(str) == str(jogador_rel_2)]

        if not jogador_1_trilho.empty:
            ax.plot(jogador_1_trilho["x_tr"], jogador_1_trilho["y_tr"], color="#2ECC71", linewidth=2.2, alpha=0.95)
        if not jogador_2_trilho.empty:
            ax.plot(jogador_2_trilho["x_tr"], jogador_2_trilho["y_tr"], color="#F2F2F2", linewidth=2.2, alpha=0.95)

        jogador_1_serie = (
            jogador_1_trilho[["time_evento_s", "x_tr", "y_tr"]]
            .drop_duplicates("time_evento_s")
            .rename(columns={"x_tr": "x_1", "y_tr": "y_1"})
        )
        jogador_2_serie = (
            jogador_2_trilho[["time_evento_s", "x_tr", "y_tr"]]
            .drop_duplicates("time_evento_s")
            .rename(columns={"x_tr": "x_2", "y_tr": "y_2"})
        )
        serie_distancias_rel = pd.merge(
            jogador_1_serie,
            jogador_2_serie,
            on="time_evento_s",
            how="inner",
        ).sort_values("time_evento_s")

        if not serie_distancias_rel.empty:
            serie_distancias_rel["distancia_m"] = np.hypot(
                serie_distancias_rel["x_2"] - serie_distancias_rel["x_1"],
                serie_distancias_rel["y_2"] - serie_distancias_rel["y_1"],
            )
            serie_distancias_rel["tempo_label"] = serie_distancias_rel["time_evento_s"].map(converter_para_relogio_fpf)
            distancia_media_rel = float(serie_distancias_rel["distancia_m"].mean())
            distancia_min_rel = float(serie_distancias_rel["distancia_m"].min())
            distancia_max_rel = float(serie_distancias_rel["distancia_m"].max())
            distancia_inicial = float(serie_distancias_rel["distancia_m"].iloc[0])
            distancia_final = float(serie_distancias_rel["distancia_m"].iloc[-1])
            delta_distancia_rel = distancia_final - distancia_inicial

            if delta_distancia_rel <= -0.25:
                tendencia_rel = "Aproximacao"
            elif delta_distancia_rel >= 0.25:
                tendencia_rel = "Afastamento"
            else:
                tendencia_rel = "Estavel"

        snapshot_rel = trilho_df.loc[trilho_df["time_evento_s"] == tempo_relativo].copy()
        jogador_1_frame = snapshot_rel[snapshot_rel["atleta_id"].astype(str) == str(jogador_rel_1)]
        jogador_2_frame = snapshot_rel[snapshot_rel["atleta_id"].astype(str) == str(jogador_rel_2)]
        if not jogador_1_frame.empty and not jogador_2_frame.empty:
            x1, y1 = jogador_1_frame[["x_tr", "y_tr"]].iloc[0]
            x2, y2 = jogador_2_frame[["x_tr", "y_tr"]].iloc[0]
            distancia_rel = float(np.hypot(x2 - x1, y2 - y1))
            x_mid = (x1 + x2) / 2
            y_mid = (y1 + y2) / 2
            pitch.scatter([x1], [y1], s=520, c="#2ECC71", edgecolors="white", linewidth=1.5, ax=ax, zorder=4)
            pitch.scatter([x2], [y2], s=520, c="#F2F2F2", edgecolors="black", linewidth=1.5, ax=ax, zorder=4)
            ax.plot([x1, x2], [y1, y2], color="#FFC857", linewidth=2.5, alpha=0.95)
            ax.text(
                x_mid,
                y_mid,
                f"{distancia_rel:.2f} m",
                color="black",
                fontsize=10,
                fontweight="bold",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFC857", edgecolor="black", alpha=0.95),
            )

    plot_placeholder.pyplot(fig)
    render_positional_footer(df_fase, momento=tempo_visualizacao, timestamps_intervalo=timestamps)

    slider_placeholder.select_slider(
        "Momento do Jogo",
        options=timestamps,
        format_func=converter_para_relogio_fpf,
        key="selected_time_jogo",
    )

    if campo == "Movimento Relativo de Jogadores ao Longo do Tempo" and not serie_distancias_rel.empty:
        fig_dist, ax_dist = plt.subplots(figsize=(10, 3.2))
        ax_dist.plot(serie_distancias_rel["time_evento_s"], serie_distancias_rel["distancia_m"], color="#E30613", linewidth=2.2)
        ax_dist.scatter(
            serie_distancias_rel["time_evento_s"].iloc[-1],
            serie_distancias_rel["distancia_m"].iloc[-1],
            color="#FFC857",
            edgecolors="black",
            s=70,
            zorder=3,
        )
        ax_dist.axhline(serie_distancias_rel["distancia_m"].mean(), color="#F2F2F2", linewidth=1.3, linestyle="--")
        ax_dist.set_facecolor("#1e1e1e")
        fig_dist.patch.set_facecolor("#1e1e1e")
        ax_dist.tick_params(colors="white", labelsize=9)
        ax_dist.set_ylabel("Distancia (m)", color="white")
        ax_dist.set_xlabel("Tempo", color="white")
        tick_positions = serie_distancias_rel["time_evento_s"].tolist()
        tick_labels = serie_distancias_rel["tempo_label"].tolist()
        step_ticks = max(1, len(tick_positions) // 5)
        ax_dist.set_xticks(tick_positions[::step_ticks])
        ax_dist.set_xticklabels(tick_labels[::step_ticks], rotation=0)
        for spine in ax_dist.spines.values():
            spine.set_color("#666666")
        ax_dist.grid(axis="y", color="#444444", linestyle=":", linewidth=0.7, alpha=0.8)
        ax_dist.set_title("Distancia entre jogadores ao longo da janela", color="white", fontsize=11)
        fig_dist.tight_layout()
        graph_placeholder.pyplot(fig_dist)

with col_info:
    st.subheader("Resumo do Frame")
    st.metric("Tempo Selecionado", converter_para_relogio_fpf(selected_time))

    if campo == "Distancia entre Jogadores":
        if distancias_pares:
            if len(distancias_pares) == 1:
                st.metric("Distancia entre Jogadores", f"{distancias_pares[0]['Distancia']:.2f} m")
            else:
                df_distancias = pd.DataFrame(distancias_pares)
                df_distancias["Distancia"] = df_distancias["Distancia"].map(lambda valor: f"{valor:.2f} m")
                st.dataframe(df_distancias, use_container_width=True, hide_index=True)
        else:
            st.info("Seleciona pelo menos 2 jogadores e 1 par.")
    elif campo == "Movimento Relativo de Jogadores ao Longo do Tempo":
        st.metric("Tempo do Movimento", converter_para_relogio_fpf(tempo_relativo) if tempo_relativo is not None else "N/A")
        st.metric("Distancia Atual", f"{distancia_rel:.2f} m" if distancia_rel is not None else "N/A")
        st.metric("Distancia Media", f"{distancia_media_rel:.2f} m" if distancia_media_rel is not None else "N/A")
        st.metric("Distancia Minima", f"{distancia_min_rel:.2f} m" if distancia_min_rel is not None else "N/A")
        st.metric("Distancia Maxima", f"{distancia_max_rel:.2f} m" if distancia_max_rel is not None else "N/A")
        st.metric(
            f"Variacao em {janela_segundos}s",
            f"{delta_distancia_rel:+.2f} m" if delta_distancia_rel is not None else "N/A",
            delta=tendencia_rel,
        )
    else:
        if not compactacao_frame.empty:
            st.metric("Compactacao Vertical", f"{float(compactacao_frame['comp_vertical'].iloc[0]):.2f}")
            st.metric("Compactacao Horizontal", f"{float(compactacao_frame['comp_horizontal'].iloc[0]):.2f}")
        else:
            st.metric("Compactacao Vertical", "N/A")
            st.metric("Compactacao Horizontal", "N/A")
        st.metric("Area", f"{float(area):.2f}" if area is not None else "N/A")

st.divider()
st.caption(f"FPF UTM Engine v16 | Tracking rows carregadas: {df_fase.shape[0]}")
