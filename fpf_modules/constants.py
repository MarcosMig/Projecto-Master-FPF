
from pyproj import Geod
import os

GEOD = Geod(ellps="WGS84")  # WGS84 geodesic distance (metros reais)

# -------------------------------
# Diretorios
# -------------------------------

HOME_DIR = os.getcwd()
OUTPUT_DIR = HOME_DIR + '/Data'
CAMPOS_DIR = OUTPUT_DIR + '/campos'
RAWDATA_DIR = OUTPUT_DIR + '/raw_data'
CLEANDATA_DIR = OUTPUT_DIR + '/clean_data'

# List all directories
directories = [OUTPUT_DIR, CAMPOS_DIR, RAWDATA_DIR, CLEANDATA_DIR]

# Create each one if it doesn't exist
if __name__ == "__main__":
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")

# -------------------------------
# Métricas Individuais (GPS-only)
# -------------------------------

HSR_MPS = 5.5 # High-Speed Running (m/s) ~19.8 km/h
SPRINT_MPS = 7.0 # Sprint (m/s) ~25.2 km/h
ACC_THR = 2.5 # m/s^2
DEC_THR = -3.0 # m/s^2
SPRINT_BOUT_MIN_S = 1.0 # duração mínima do bout de sprint (s)
ENGINE_VERSION = "v12-metrics"


COL_LAT = "Lat"
COL_LON = "Lon"
COL_TIME = "Time"
COL_FASE = "Fase"

# -------------------------------
# Opções objetos Streamlit
# -------------------------------

# --- Seleções (lista fechada) ---
SELECOES_OPCOES = [
    "AA M", "AA F",
    "U23 M", "U23 F",
    "U21 M", "U21 F",
    "U20 M", "U20 F",
    "U19 M", "U19 F",
    "U18 M", "U18 F",
    "U17 M", "U17 F",
    "U16 M", "U16 F",
    "U15 M", "U15 F",
    "U14 M", "U14 F",
    "U13 M", "U13 F",
]
