import pandas as pd


def round_metrics_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Round numeric metrics to avoid excessive decimals in reports.
    """

    if df is None or df.empty:
        return df

    out = df.copy()

    int_cols = [
        "n_sprints",
        "n_acc_2_5",
        "n_dec_3_0",
        "n_points",
        "n_gaps_gt2s",
        "n_jumps_gt15m",
        "n_gaps_gt2s_qc",
        "session_sk",
        "athlete_sk",
        "phase_id",
    ]

    round_1 = [
        "duracao_min",
        "dist_m",
        "m_min",
        "peak_1m_m_min",
        "hsr_dist_m",
        "hsr_pct",
        "sprint_dist_m",
        "active_time_min",
        "active_pct",
        "pct_time_valid",
    ]

    round_2 = [
        "vmax_mps",
        "vmax_mps_qc",
    ]

    round_4 = [
        "rotation_rad",
    ]

    for col in int_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")

    for col in round_1:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(1)

    for col in round_2:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(2)

    for col in round_4:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(4)

    return out


def file_to_bytes(pathlike) -> bytes:
    with open(pathlike, "rb") as f:
        return f.read()


def converter_para_relogio_fpf(segundos_totais):
    """
    Exemplo: 4150.3s -> '69:10.3'
    (Minuto 69, Segundo 10, Frame 3)
    """
    minutos = int(segundos_totais // 60)
    segundos = int(segundos_totais % 60)
    frame = int(round((segundos_totais % 1) * 10))
    if frame == 10:
        frame = 0
        segundos += 1  # Ajuste de arredondamento

    return f"{minutos:02d}:{segundos:02d}.{frame}"
