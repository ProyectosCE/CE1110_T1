# Tarea ASM – Parte 2 (FFT en audio)

Este repositorio contiene el código para **cargar una pista de audio libre**, calcular su **FFT** y **mostrar/guardar** los espectros de **magnitud** y **fase**. El objetivo es producir **figuras** (PNG) claras para incluir en el PDF del informe.

> **Formato de audio recomendado:** WAV.  
> También se pueden usar otros formatos si agregan bibliotecas como `librosa` y `soundfile`.

## Requisitos

- **Python 3.10 – 3.12**
- Conexión a internet solo para instalar dependencias la primera vez.
- Windows / macOS / Linux (probado con PowerShell, CMD, Git Bash y WSL).

## Estructura esperada

```
Codigo_Fuente/
├─ audio/           # pistas .wav (entrada)
├─ figs/            # figuras .png (salida)
├─ src/             # módulos Python (lógica)
├─ experimento1.py          # punto de entrada (invoca la FFT, genera figuras)
├─ requirements.txt # dependencias
├─ run.sh           # lanzador (Bash: Git Bash/WSL/macOS)
├─ run.bat          # lanzador (CMD)
├─ run.ps1          # lanzador (PowerShell)
└─ README.md
```

## Instalación y ejecución

### Opción A: PowerShell (Windows)
```powershell
cd Codigo_Fuente
powershell -ExecutionPolicy Bypass -File .\run.ps1 --setup   # crea venv e instala deps
powershell -ExecutionPolicy Bypass -File .\run.ps1           # ejecuta experimento1.py
```

### Opción B: CMD (Windows)

```bat
cd Codigo_Fuente
run.bat --setup   && ^
run.bat
```

### Opción C: Git Bash / WSL / macOS

```bash
cd Codigo_Fuente
chmod +x run.sh
./run.sh --setup
./run.sh
```

> Los lanzadores crean automáticamente: `audio/`, `figs/` y `src/`.
> Asegúrate de colocar tu pista libre como `audio/tu_pista.wav` (o ajusta el nombre en `experimento1.py`).

## Parámetros y salida

* **Consola**: se imprime resumen de `fs`, `N`, ventana, `Δf`, y picos detectados (si se implementa).
* **Figuras**:

  * `figs/waveform.png` – forma de onda (tiempo)
  * `figs/magnitude_db.png` – espectro de magnitud (Hz, dB)
  * `figs/phase_unwrapped.png` – espectro de fase desenrollada (rad)
* **Ventanas gráficas**: se abren para facilitar capturas (puedes desactivar via flag si lo implementas).

## Notas

* **WAV** evita problemas de códecs en Windows.
* Si se agregan soporte a más formatos, añade a `requirements.txt`: `soundfile`, `librosa`.
* Para regenerar figuras sin abrir ventanas, se puede implementar un flag `--save-only` en `experimento1.py`.

---
