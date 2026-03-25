"""
Contem as funções desenvolvidas com o objetivo da visualização de dados.
"""

import mplsoccer as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Streamlit elementos customizados ─────────────────────────────────────


def kpi_card(label, value, tooltip=None):
    title_attr = f'title="{tooltip}"' if tooltip else ""
    st.markdown(
        f"""
        <div class="fpt-kpi-card" {title_attr}>
            <div class="fpt-kpi-label">{label}</div>
            <div class="fpt-kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric(label, value):
    st.markdown(
        f"""
            <div style="padding:6px 0; border-bottom:1px solid
    #2a2a2a;">
                <div style="font-size:0.85rem; color:
    #9aa0a6; font-weight:600;">{label}</div>
                <div style="font-size:1.3rem; font-weight:700; color:
    #ffffff;">{value}</div>
            </div>
    """,
        unsafe_allow_html=True,
    )


# ── Plotly Pitch ─────────────────────────────────────


def draw_statsbomb_pitch_horizontal():
    shapes = [
        # Pitch outline
        dict(
            type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color="#c7d5cc", width=2)
        ),
        # Halfway line
        dict(
            type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color="#c7d5cc", width=2)
        ),
        # Centre circle
        dict(
            type="circle",
            x0=51,
            y0=31,
            x1=69,
            y1=49,
            line=dict(color="#c7d5cc", width=2),
        ),
        # Centre dot
        dict(
            type="circle",
            x0=59.5,
            y0=39.5,
            x1=60.5,
            y1=40.5,
            fillcolor="#c7d5cc",
            line=dict(color="white"),
        ),
        # Penalty areas
        dict(
            type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color="#c7d5cc", width=2)
        ),
        dict(
            type="rect",
            x0=102,
            y0=18,
            x1=120,
            y1=62,
            line=dict(color="#c7d5cc", width=2),
        ),
        # 6-yard boxes
        dict(
            type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color="#c7d5cc", width=2)
        ),
        dict(
            type="rect",
            x0=114,
            y0=30,
            x1=120,
            y1=50,
            line=dict(color="#c7d5cc", width=2),
        ),
        # Goals
        dict(
            type="rect", x0=-2, y0=36, x1=0, y1=44, line=dict(color="#c7d5cc", width=2)
        ),
        dict(
            type="rect",
            x0=120,
            y0=36,
            x1=122,
            y1=44,
            line=dict(color="#c7d5cc", width=2),
        ),
    ]
    return shapes


# Cores Heatmap
custom_hot = [
    [0.0, "rgba(34, 49, 43, 0)"],  # transparente — campo limpo
    [
        0.08,
        "rgba(34, 49, 43, 0.5)",
    ],  # verde campo — densidade muito baixa
    [0.2, "rgba(80, 80, 20, 0.8)"],  # transição rápida para quente
    [0.35, "rgba(180, 160, 0, 0.9)"],  # amarelo escuro
    [0.5, "rgba(240, 220, 0, 1.0)"],  # amarelo brilhante
    [0.65, "rgba(255, 140, 0, 1.0)"],  # laranja
    [0.8, "rgba(220, 50, 0, 1.0)"],  # vermelho alaranjado
    [1.0, "rgba(120, 0, 0, 1.0)"],  # vermelho escuro
]
