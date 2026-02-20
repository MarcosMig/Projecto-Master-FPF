# Projecto-Master-FPF
Plataforma Analise Posicional Multi Seleção

Overview
--------
Small helper project to detect and build local field models from GPS/CSV corner files.

Prerequisites
-------------
- Ubuntu or Debian-based Linux
- Python 3.8+
- System library for PROJ (required by `pyproj`): `libproj-dev`, `proj-data`, `proj-bin`

Quick setup
-----------
1. Install system deps (run as root or with sudo):

```sh
sudo apt update && sudo apt install -y python3 python3-venv python3-pip libproj-dev proj-data proj-bin
```

2. Create venv and install Python deps:

```sh
./run.sh setup
```

3. Run the field detection script (uses the venv):

```sh
./run.sh run
```

Files of interest
-----------------
- `Field Detection` — the main Python script (contains a space in the filename).
- `requirements.txt` — Python dependencies (`pandas`, `numpy`, `pyproj`).
- `run.sh` — helper to `setup` (create venv + install) and `run` the script.

Notes
-----
- Edit the `BASE_PATH` constant inside `Field Detection` to point to your local dataset before running.
- If `pyproj` fails to install, ensure the PROJ system packages above are present.
- The script expects a specific folder layout under `BASE_PATH` (see script comments).

Contact
-------
Author: Miguel Cardoso
