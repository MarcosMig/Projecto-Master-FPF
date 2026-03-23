"""
Contem as funções desenvolvidas com o objetivo da visualização de dados
"""

import mplsoccer as mpl
import numpy as np
import pandas as pd
import plotly.graph_objects as go


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
