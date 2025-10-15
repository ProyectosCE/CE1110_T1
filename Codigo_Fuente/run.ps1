# run.ps1 - Crea/activa venv, instala requirements y ejecuta el entry .py
# Uso:
#   .\run.ps1                                  # instala (si hace falta) y corre experimento1.py
#   .\run.ps1 --setup                          # solo instala/actualiza deps
#   .\run.ps1 --no-install                     # no instala, solo ejecuta experimento1.py
#   .\run.ps1 --entry main.py                  # cambia archivo de entrada
#   .\run.ps1 -- --arg1 valor --flag           # pasa args a tu script tras '--'

param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ArgsPassThru
)

$ErrorActionPreference = "Stop"
$VenvDir = ".venv"
$ActivatePs1 = Join-Path $VenvDir "Scripts\Activate.ps1"

# Defaults
$EntryFile = "experimento1.py"

function Ensure-Dirs {
  foreach ($d in @("audio","figs","src")) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d | Out-Null }
  }
}

function Ensure-Venv {
  if (-not (Test-Path (Join-Path $VenvDir "Scripts\python.exe"))) {
    Write-Host "[run.ps1] Creando entorno virtual en $VenvDir ..."
    python -m venv $VenvDir
  }
  if (-not (Test-Path $ActivatePs1)) {
    throw "[run.ps1] No se encontró $ActivatePs1"
  }
  . $ActivatePs1
}

function Install-Reqs {
  if (Test-Path "requirements.txt") {
    python -m pip install --upgrade pip
    pip install -r requirements.txt
  } else {
    Write-Warning "[run.ps1] No existe requirements.txt; se omite instalación."
  }
}

# ---- Parseo simple de flags ----
if ($null -eq $ArgsPassThru) { $ArgsPassThru = @() }

# Consumimos --entry <file> si existe
$entryIdx = [Array]::IndexOf($ArgsPassThru, "--entry")
if ($entryIdx -ge 0 -and ($entryIdx + 1) -lt $ArgsPassThru.Count) {
  $EntryFile = $ArgsPassThru[$entryIdx + 1]
  # quitar ambos de la lista
  $ArgsPassThru = $ArgsPassThru[0..($entryIdx-1)] + $ArgsPassThru[($entryIdx+2)..($ArgsPassThru.Count-1)]
}

# Detectar --setup y --no-install
$doSetup     = $ArgsPassThru -contains "--setup"
$noInstall   = $ArgsPassThru -contains "--no-install"

# Separar args después de '--' para pasarlos al script Python
$ddIdx = [Array]::IndexOf($ArgsPassThru, "--")
$forward = @()
if ($ddIdx -ge 0) {
  if ($ddIdx + 1 -lt $ArgsPassThru.Count) { $forward = $ArgsPassThru[($ddIdx+1)..($ArgsPassThru.Count-1)] }
  # quitar desde '--' hacia el final
  $ArgsPassThru = $ArgsPassThru[0..($ddIdx-1)]
}

# ---- Flujo principal ----
Ensure-Dirs
Ensure-Venv

if ($doSetup) {
  Install-Reqs
  Write-Host "[run.ps1] Setup completado."
  exit 0
}

if (-not $noInstall) {
  Install-Reqs
}

# Verificar archivo de entrada
if (-not (Test-Path $EntryFile)) {
  # fallback: si no está, probar con main.py
  if (Test-Path "main.py") {
    Write-Warning "[run.ps1] No existe $EntryFile; usando main.py"
    $EntryFile = "main.py"
  } else {
    throw "[run.ps1] No existe $EntryFile ni main.py en $(Get-Location)"
  }
}

Write-Host "[run.ps1] Ejecutando $EntryFile $($forward -join ' ')"
python $EntryFile @forward
