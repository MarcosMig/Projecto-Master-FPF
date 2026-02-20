#!/usr/bin/env bash
set -euo pipefail

# Simple helper to create a venv, install requirements, and optionally run the script.
# Usage: ./run.sh setup   # create venv and install
#        ./run.sh run     # ensure venv and run the Field Detection script

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
REQ="$ROOT_DIR/requirements.txt"
SCRIPT="$ROOT_DIR/Field Detection"

if [ "${1:-}" = "run" ]; then
  if [ ! -d "$VENV_DIR" ]; then
    echo "Virtualenv not found; creating and installing dependencies..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip
    "$VENV_DIR/bin/pip" install -r "$REQ"
  fi
  echo "Running script using venv: $VENV_DIR"
  "$VENV_DIR/bin/python3" "$SCRIPT"
  exit 0
fi

if [ "${1:-}" = "setup" ] || [ -z "${1:-}" ]; then
  echo "Creating virtual environment in $VENV_DIR and installing requirements..."
  python3 -m venv "$VENV_DIR"
  "$VENV_DIR/bin/pip" install --upgrade pip
  "$VENV_DIR/bin/pip" install -r "$REQ"
  echo "Setup complete. To run the script: ./run.sh run"
  exit 0
fi

echo "Unknown command. Use 'setup' or 'run'."
exit 2
