param(
    [switch]$Reinstall,
    [switch]$Check,
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvDir = if ($env:FPF_HUB_VENV) {
    $env:FPF_HUB_VENV
} else {
    Join-Path $env:LOCALAPPDATA "FPF-Hub\.venv"
}

$PythonExe = Join-Path $VenvDir "Scripts\python.exe"
$LockFile = Join-Path $ProjectRoot "requirements-lock.txt"
$RequirementsFile = if (Test-Path $LockFile) { $LockFile } else { Join-Path $ProjectRoot "requirements.txt" }

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Comando falhou ($LASTEXITCODE): $Command $($Arguments -join ' ')"
    }
}

function Find-SystemPython {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if ($python) {
        try {
            Invoke-Checked $python.Source @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)")
            return $python.Source
        } catch {}
    }

    throw "Python 3.10+ nao encontrado. Instala Python 3.10 ou superior, ou ajusta o PATH."
}

function New-HubVenv {
    if (Test-Path $PythonExe) {
        return
    }

    Write-Host "A criar ambiente isolado em $VenvDir"
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $VenvDir) | Out-Null
    $python = Find-SystemPython
    Invoke-Checked $python @("-m", "venv", $VenvDir)

    if (-not (Test-Path $PythonExe)) {
        throw "Nao foi possivel criar o ambiente Python em $VenvDir."
    }
}

function Install-HubRequirements {
    New-HubVenv

    if (-not (Test-Path $PythonExe)) {
        throw "Ambiente Python invalido: $PythonExe nao existe."
    }

    Write-Host "A instalar dependencias a partir de $RequirementsFile"
    Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-Checked $PythonExe @("-m", "pip", "install", "--upgrade", "--force-reinstall", "-r", $RequirementsFile)
}

function Test-HubEnvironment {
    if (-not (Test-Path $PythonExe)) {
        throw "Ambiente Python invalido: $PythonExe nao existe."
    }

    $code = "import importlib; [importlib.import_module(m) for m in ['streamlit','pandas','numpy','sklearn.cluster','joblib.externals.loky.backend.context']]; from sklearn.cluster import KMeans"

    Invoke-Checked $PythonExe @("-c", $code)
}

New-HubVenv

if ($Reinstall) {
    Install-HubRequirements
}

try {
    Test-HubEnvironment
} catch {
    Write-Host "Ambiente incompleto ou inconsistente. A reparar..."
    New-HubVenv
    Install-HubRequirements
    Test-HubEnvironment
}

if ($Check) {
    Write-Host "Ambiente FPF Hub OK: $PythonExe"
    exit 0
}

Write-Host "A iniciar FPF Hub com $PythonExe"
$streamlitArgs = @("-m", "streamlit", "run", "Home.py")

if ($Foreground) {
    Invoke-Checked $PythonExe $streamlitArgs
}

$process = Start-Process `
    -FilePath "powershell.exe" `
    -ArgumentList @(
        "-NoExit",
        "-ExecutionPolicy", "Bypass",
        "-Command",
        "& { Set-Location '$ProjectRoot'; & '$PythonExe' @('-m','streamlit','run','Home.py') }"
    ) `
    -WorkingDirectory $ProjectRoot `
    -PassThru

Start-Sleep -Seconds 4

if ($process.HasExited) {
    throw "O servidor Streamlit terminou durante o arranque."
}

Write-Host "FPF Hub iniciado em background."
Write-Host "PID: $($process.Id)"
Write-Host "Local URL: http://localhost:8501"
