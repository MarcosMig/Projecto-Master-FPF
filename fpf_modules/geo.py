"""
Module responsible for coordinate transformations and geocoding.
"""

import numpy as np
import folium
from pyproj import Transformer
import requests

from .constants import (
    COL_LAT, COL_LON,
    GEOD
)

from .io_utils import (
    read_csv_upload,
    get_atleta_id
)



def calibrar_campo(f_campo_files, epsg: int):
    pts_gps = {}
    for f in f_campo_files:
        df_c = read_csv_upload(f)
        if COL_LAT not in df_c.columns or COL_LON not in df_c.columns:
            continue
        for key in ["BL", "BR", "TL", "TR"]:
            if key in f.name.upper():
                pts_gps[key] = [df_c[COL_LAT].mean(), df_c[COL_LON].mean()]

    if len(pts_gps) != 4:
        missing = [k for k in ["BL", "BR", "TL", "TR"] if k not in pts_gps]
        raise ValueError(f"Campo incompleto. Em falta: {', '.join(missing)}")

    clat = float(np.mean([p[0] for p in pts_gps.values()]))
    clon = float(np.mean([p[1] for p in pts_gps.values()]))

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts_utm = {}
    for key, (lat, lon) in pts_gps.items():
        x, y = transformer.transform(lon, lat)
        pts_utm[key] = np.array([x, y], dtype=float)

    dist_comprimento = float(np.linalg.norm(pts_utm["BR"] - pts_utm["BL"]))
    dist_largura = float(np.linalg.norm(pts_utm["TL"] - pts_utm["BL"]))

    origin = pts_utm["BL"]
    v_base = pts_utm["BR"] - origin
    angulo_rad = float(np.arctan2(v_base[1], v_base[0]))

    # Rotação para alinhar BL->BR no eixo X
    R = np.array(
        [
            [np.cos(-angulo_rad), -np.sin(-angulo_rad)],
            [np.sin(-angulo_rad), np.cos(-angulo_rad)],
        ],
        dtype=float,
    )

    return (
        pts_gps,
        (clat, clon),
        pts_utm,
        origin,
        R,
        angulo_rad,
        dist_comprimento,
        dist_largura,
    )

def order_corners_latlon(points):
    """
    Recebe lista de 4 pontos [(lat, lon), ...] e devolve dict com chaves BL, BR, TL, TR.
    Regra:
      - divide por lat (top 2 e bottom 2)
      - dentro de cada par, ordena por lon (esq/dir)
    """
    if points is None or len(points) != 4:
        raise ValueError("São necessários exatamente 4 pontos para ordenar cantos.")

    pts = [(float(lat), float(lon)) for lat, lon in points]
    pts_sorted_lat = sorted(pts, key=lambda p: p[0], reverse=True)  # maior lat = norte (topo)
    top = pts_sorted_lat[:2]
    bottom = pts_sorted_lat[2:]

    top_sorted = sorted(top, key=lambda p: p[1])      # menor lon = esquerda
    bottom_sorted = sorted(bottom, key=lambda p: p[1])

    TL = top_sorted[0]
    TR = top_sorted[1]
    BL = bottom_sorted[0]
    BR = bottom_sorted[1]

    return {"BL": [BL[0], BL[1]], "BR": [BR[0], BR[1]], "TL": [TL[0], TL[1]], "TR": [TR[0], TR[1]]}


def retangularizar_cantos_latlon(points_latlon, epsg: int):
    """
    A partir de 4 pontos (lat,lon) clicados, cria uma versão "retangular" consistente.
    Passos:
      1) Ordena pontos (TL/TR/BR/BL) por lat/lon (aprox)
      2) Converte para UTM
      3) Define eixo X pelo vetor médio esquerda->direita (top + bottom), e eixo Y perpendicular
      4) Projeta os 4 pontos nesses eixos, faz snap em min/max e reconstrói um retângulo perfeito
      5) Converte o retângulo de volta para lat/lon
    Retorna:
      pts_clicked (dict BL/BR/TL/TR latlon),
      pts_rect (dict BL/BR/TL/TR latlon)  # ajustado
    """
    pts_clicked = order_corners_latlon(points_latlon)

    # transformers
    to_utm = Transformer.from_crs("EPSG:4326", f"EPSG:{int(epsg)}", always_xy=True)
    to_wgs = Transformer.from_crs(f"EPSG:{int(epsg)}", "EPSG:4326", always_xy=True)

    # UTM coords dict
    pts_utm = {}
    for k, (lat, lon) in pts_clicked.items():
        x, y = to_utm.transform(float(lon), float(lat))
        pts_utm[k] = np.array([x, y], dtype=float)

    # centro
    C = np.mean(np.stack(list(pts_utm.values()), axis=0), axis=0)

    # eixo X: média (TL->TR) e (BL->BR)
    vx1 = pts_utm["TR"] - pts_utm["TL"]
    vx2 = pts_utm["BR"] - pts_utm["BL"]
    vx = vx1 + vx2
    norm_vx = float(np.linalg.norm(vx))
    if norm_vx < 1e-6:
        # fallback: BL->BR
        vx = pts_utm["BR"] - pts_utm["BL"]
        norm_vx = float(np.linalg.norm(vx))
        if norm_vx < 1e-6:
            raise ValueError("Não foi possível estimar eixo do campo a partir dos pontos.")

    ux = vx / norm_vx
    # eixo Y perpendicular (rot 90º)
    uy = np.array([-ux[1], ux[0]], dtype=float)

    # projecções
    def proj(p):
        d = p - C
        return float(np.dot(d, ux)), float(np.dot(d, uy))  # (x', y')

    proj_vals = {k: proj(v) for k, v in pts_utm.items()}
    xs = [v[0] for v in proj_vals.values()]
    ys = [v[1] for v in proj_vals.values()]
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))

    # reconstrução do retângulo em UTM
    # Nota: usando a convenção: topo = y_max (mais "norte" no referencial uy), base = y_min
    rect_proj = {
        "TL": (x_min, y_max),
        "TR": (x_max, y_max),
        "BL": (x_min, y_min),
        "BR": (x_max, y_min),
    }

    pts_rect_utm = {k: (C + x * ux + y * uy) for k, (x, y) in rect_proj.items()}

    # back to latlon
    pts_rect = {}
    for k, p in pts_rect_utm.items():
        lon, lat = to_wgs.transform(float(p[0]), float(p[1]))
        pts_rect[k] = [float(lat), float(lon)]

    return pts_clicked, pts_rect



def calibrar_campo_from_pts_gps(pts_gps: dict, epsg: int):
    """
    Igual ao calibrar_campo, mas recebe os 4 cantos já como lat/lon.
    pts_gps: {"BL":[lat,lon], "BR":[lat,lon], "TL":[lat,lon], "TR":[lat,lon]}
    """
    for k in ["BL", "BR", "TL", "TR"]:
        if k not in pts_gps:
            raise ValueError(f"Campo incompleto. Em falta: {k}")

    clat = float(np.mean([pts_gps[k][0] for k in pts_gps]))
    clon = float(np.mean([pts_gps[k][1] for k in pts_gps]))

    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    pts_utm = {}
    for key, (lat, lon) in pts_gps.items():
        x, y = transformer.transform(float(lon), float(lat))
        pts_utm[key] = np.array([x, y], dtype=float)

    dist_comprimento = float(np.linalg.norm(pts_utm["BR"] - pts_utm["BL"]))
    dist_largura = float(np.linalg.norm(pts_utm["TL"] - pts_utm["BL"]))

    origin = pts_utm["BL"]
    v_base = pts_utm["BR"] - origin
    angulo_rad = float(np.arctan2(v_base[1], v_base[0]))

    # Rotação para alinhar BL->BR no eixo X
    R = np.array(
        [
            [np.cos(-angulo_rad), -np.sin(-angulo_rad)],
            [np.sin(-angulo_rad), np.cos(-angulo_rad)],
        ],
        dtype=float,
    )

    return (
        pts_gps,
        (clat, clon),
        pts_utm,
        origin,
        R,
        angulo_rad,
        dist_comprimento,
        dist_largura,
    )



def sample_athlete_track_latlon(f_atleta_files, max_points=600):
    """
    Lê um atleta (primeiro ficheiro) e devolve uma amostra de pontos lat/lon para desenhar no mapa.
    Serve apenas para orientar o utilizador no 'pick' dos cantos.
    """
    if not f_atleta_files:
        return []
    try:
        df = read_csv_upload(f_atleta_files[0], nrows=5000)
        if COL_LAT not in df.columns or COL_LON not in df.columns:
            return []
        sub = df[[COL_LAT, COL_LON]].dropna()
        if sub.empty:
            return []
        if len(sub) > max_points:
            sub = sub.sample(n=max_points, random_state=7)
        pts = sub.values.tolist()
        return [(float(lat), float(lon)) for lat, lon in pts if np.isfinite(lat) and np.isfinite(lon)]
    except Exception:
        return []


def reverse_geocode_place_city_country(lat: float, lon: float):
    """Reverse geocode via OpenStreetMap Nominatim.
    Devolve (place_name, city, country). 'place_name' tenta capturar estádio/recinto quando disponível.
    """
    try:
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {"format": "jsonv2", "lat": lat, "lon": lon, "zoom": 18, "addressdetails": 1}
        headers = {"User-Agent": "FPF-Performance-Hub/1.0 (contact: performance@fpf.pt)"}
        r = requests.get(url, params=params, headers=headers, timeout=10)
        if r.status_code != 200:
            return None, None, None
        data = r.json()
        if not isinstance(data, dict):
            return None, None, None

        addr = data.get("address", {}) or {}
        city = (
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or addr.get("municipality")
            or addr.get("county")
        )
        country = addr.get("country")

        # Melhor esforço para capturar um nome de recinto/estádio
        place = (
            data.get("name")
            or addr.get("stadium")
            or addr.get("sports_centre")
            or addr.get("amenity")
            or data.get("display_name")
        )
        return place, city, country
    except Exception:
        return None, None, None



def geo_validacao_por_atleta(
    f_atleta_files, centroid_lat, centroid_lon, raio_m, amostra_n, min_pct_ok
):
    ok, fora, erros = [], [], []
    for f in f_atleta_files:
        aid = get_atleta_id(f.name)
        try:
            df = read_csv_upload(f, nrows=int(amostra_n))
            if COL_LAT not in df.columns or COL_LON not in df.columns:
                erros.append((aid, "Sem colunas Lat/Lon"))
                continue
            sub = df[[COL_LAT, COL_LON]].dropna()
            if sub.empty:
                erros.append((aid, "Sem amostras Lat/Lon válidas"))
                continue
            lat_med = float(sub[COL_LAT].median())
            lon_med = float(sub[COL_LON].median())
            _, _, dist_m = GEOD.inv(lon_med, lat_med, centroid_lon, centroid_lat)
            if dist_m <= raio_m:
                ok.append((aid, dist_m))
            else:
                fora.append((aid, dist_m))
        except Exception as e:
            erros.append((aid, str(e)))

    total = len(set([get_atleta_id(f.name) for f in f_atleta_files]))
    pct_ok = (len({a for a, _ in ok}) / max(1, total))
    passed = pct_ok >= min_pct_ok
    return passed, pct_ok, ok, fora, erros



def get_atletas_centroid_latlon(f_atleta_files, amostra_n=500):
    per_atleta = {}

    for f in f_atleta_files:
        aid = get_atleta_id(f.name)
        try:
            df = read_csv_upload(f, nrows=int(amostra_n))
            if COL_LAT not in df.columns or COL_LON not in df.columns:
                continue

            sub = df[[COL_LAT, COL_LON]].dropna()
            if sub.empty:
                continue

            lat_med = float(pd.to_numeric(sub[COL_LAT], errors="coerce").dropna().median())
            lon_med = float(pd.to_numeric(sub[COL_LON], errors="coerce").dropna().median())

            if np.isfinite(lat_med) and np.isfinite(lon_med):
                per_atleta.setdefault(aid, []).append((lat_med, lon_med))

        except Exception:
            continue

    if not per_atleta:
        return None, None

    atleta_meds = []
    for vals in per_atleta.values():
        lats = [v[0] for v in vals]
        lons = [v[1] for v in vals]
        atleta_meds.append((float(np.median(lats)), float(np.median(lons))))

    lat_c = float(np.median([x[0] for x in atleta_meds]))
    lon_c = float(np.median([x[1] for x in atleta_meds]))

    return lat_c, lon_c
