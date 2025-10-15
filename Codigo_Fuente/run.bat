@echo off
REM run.bat - Crea/activa venv, instala requirements y ejecuta el entry .py
REM Uso:
REM   run.bat                         -> instala (si hace falta) y corre experimento1.py
REM   run.bat --setup                 -> solo instala/actualiza deps
REM   run.bat --no-install            -> no instala, solo ejecuta
REM   run.bat --entry main.py         -> cambia archivo de entrada
REM   run.bat -- args...              -> pasa args a tu script tras '--'

setlocal enabledelayedexpansion
set VENV_DIR=.venv
set ACTIVATE=%VENV_DIR%\Scripts\activate
set PYTHON=python
set ENTRY=experimento1.py

REM Parseo simple de --entry <file>
set i=1
:parse
if "%~%i"=="" goto parsedone
for %%A in (--entry) do (
  if "%%%i"=="%%A" (
    set /a j=i+1
    for %%B in (%%%j) do ( set ENTRY=%%B )
  )
)
set /a i=i+1
goto parse
:parsedone

IF NOT EXIST "%VENV_DIR%\Scripts\python.exe" (
    echo [run.bat] Creando entorno virtual en %VENV_DIR% ...
    %PYTHON% -m venv %VENV_DIR%
)

IF NOT EXIST "%ACTIVATE%.bat" (
    echo [run.bat] ERROR: no se encontro el activador del venv en %ACTIVATE%.bat
    exit /b 1
)

call "%ACTIVATE%.bat"

IF NOT EXIST audio mkdir audio
IF NOT EXIST figs mkdir figs
IF NOT EXIST src mkdir src

if "%1"=="--setup" (
    if exist requirements.txt (
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    ) else (
        echo [run.bat] WARNING: no existe requirements.txt; se omite instalacion.
    )
    echo [run.bat] Setup completado.
    exit /b 0
)

if "%1"=="--no-install" (
    shift
) else (
    if exist requirements.txt (
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    ) else (
        echo [run.bat] WARNING: no existe requirements.txt; se omite instalacion.
    )
)

REM Separar args tras '--' para pasarlos al script
setlocal DisableDelayedExpansion
set CMD_ARGS=
set PASS_THRU=0
for %%A in (%*) do (
  if "%%~A"=="--" (
    set PASS_THRU=1
  ) else (
    if !PASS_THRU! == 1 (
      set CMD_ARGS=!CMD_ARGS! %%~A
    )
  )
)
setlocal EnableDelayedExpansion

IF NOT EXIST "%ENTRY%" (
    if EXIST "main.py" (
        echo [run.bat] WARNING: no existe %ENTRY%; usando main.py
        set ENTRY=main.py
    ) else (
        echo [run.bat] ERROR: no existe %ENTRY% ni main.py
        exit /b 2
    )
)

echo [run.bat] Ejecutando %ENTRY% !CMD_ARGS!
python "%ENTRY%" !CMD_ARGS!
