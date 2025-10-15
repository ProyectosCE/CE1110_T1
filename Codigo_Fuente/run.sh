#!/usr/bin/env bash
# run.sh - Crea/activa venv, instala requirements y ejecuta el entry .py
# Uso:
#   ./run.sh                            # instala (si hace falta) y corre experimento1.py
#   ./run.sh --setup                    # solo instala/actualiza deps
#   ./run.sh --no-install               # no instala, solo ejecuta
#   ./run.sh --entry main.py            # cambia archivo de entrada
#   ./run.sh -- --arg1 val --flag       # pasa args a tu script tras '--'

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="python"
command -v python3 >/dev/null 2>&1 && PYTHON_BIN="python3"

VENV_DIR=".venv"
ACT_SH="$VENV_DIR/bin/activate"
ACT_WIN="$VENV_DIR/Scripts/activate"

ENTRY="experimento1.py"
INSTALL=1

# Parsear --entry
for i in "$@"; do
  if [[ "$i" == "--entry" ]]; then
    shift
    ENTRY="${1:-$ENTRY}"
  fi
done

ensure_dirs() {
  mkdir -p audio figs src
}

create_venv_if_needed() {
  if [[ ! -d "$VENV_DIR" ]]; then
    echo "[run.sh] Creando entorno virtual en $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
}

activate_venv() {
  if [[ -f "$ACT_SH" ]]; then
    # shellcheck disable=SC1090
    source "$ACT_SH"
  elif [[ -f "$ACT_WIN" ]]; then
    # shellcheck disable=SC1090
    source "$ACT_WIN"
  else
    echo "[run.sh] No se encontro script de activacion del venv." >&2
    exit 1
  fi
}

install_requirements() {
  if [[ -f "requirements.txt" ]]; then
    python -m pip install --upgrade pip
    pip install -r requirements.txt
  else
    echo "[run.sh] WARNING: no existe requirements.txt; se omite instalación."
  fi
}

# ---- flujo ----
ensure_dirs
create_venv_if_needed
activate_venv

# flags principales
if [[ "${1:-}" == "--setup" ]]; then
  install_requirements
  echo "[run.sh] Setup completado."
  exit 0
fi

if [[ "${1:-}" == "--no-install" ]]; then
  INSTALL=0
  shift
fi

# separar args tras '--' para forwarding
forward=()
if [[ "$#" -gt 0 ]]; then
  for idx in $(seq 1 "$#"); do
    arg="${!idx}"
    if [[ "$arg" == "--" ]]; then
      # copiar lo que sigue literalmente
      shift $idx
      forward=("$@")
      break
    fi
  done
fi

if [[ $INSTALL -eq 1 ]]; then
  install_requirements
fi

# fallback si no existe entry
if [[ ! -f "$ENTRY" ]]; then
  if [[ -f "main.py" ]]; then
    echo "[run.sh] WARNING: no existe $ENTRY; usando main.py"
    ENTRY="main.py"
  else
    echo "[run.sh] ERROR: no existe $ENTRY ni main.py en $(pwd)" >&2
    exit 2
  fi
fi

echo "[run.sh] Ejecutando $ENTRY ${forward[*]}"
python "$ENTRY" "${forward[@]}"
