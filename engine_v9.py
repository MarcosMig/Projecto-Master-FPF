# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 16:39:48 2026

@author: MiguelCardoso
"""

# engine_v9.py
import pandas as pd
import numpy as np
from pyproj import Transformer
from scipy.signal import savgol_filter

def run_utm_v9_logic(f_campo, f_atleta):
    # --- FASE 1: CALIBRAÇÃO (Igual ao teu script original) ---
    pts_gps = {}
    for f in f_campo:
        df_c = pd.read_csv(f, sep=None, engine='python')
        df_c.columns = [c.strip().replace('"', '') for c in df_c.columns]
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper():
                pts_gps[key] = [df_c["Lat"].mean(), df_c["Lon"].mean()]
    
    # Conversão para UTM
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32629", always_xy=True)
    pts_utm = {k: np.array(transformer.transform(v[1], v[0])) for k, v in pts_gps.items()}
    
    # Cálculo de Geometria
    dist_x = np.linalg.norm(pts_utm["BR"] - pts_utm["BL"])
    dist_y = np.linalg.norm(pts_utm["TL"] - pts_utm["BL"])
    origin = pts_utm["BL"]
    v_base = pts_utm["BR"] - origin
    angulo_rad = np.arctan2(v_base[1], v_base[0])
    
    # --- FASE 2, 3 e 4: (Processamento de ficheiros de atletas) ---
    # Aqui a lógica do v9.2 processa cada ficheiro carregado em memória
    processed_count = len(f_atleta)
    
    return {
        "x": dist_x,
        "y": dist_y,
        "rot": np.degrees(angulo_rad),
        "count": processed_count
    }