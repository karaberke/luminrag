<#
.SYNOPSIS
Start LuminRAG locally with browser-owned workspaces and provider keys.
.EXAMPLE
.\scripts\run_local.ps1
.EXAMPLE
.\scripts\run_local.ps1 -BackendPort 8010 -FrontendPort 5174
.EXAMPLE
.\scripts\run_local.ps1 -SelfHosted
#>
[CmdletBinding()]
param(
    [switch]$SelfHosted,
    [ValidateRange(1024, 65535)][int]$BackendPort = 8000,
    [ValidateRange(1024, 65535)][int]$FrontendPort = 5173,
    [ValidateRange(10, 1800)][int]$StartupTimeoutSeconds = 300
)

Set-StrictMode -Version 3.0
$ErrorActionPreference = 'Stop'

function Assert-AvailablePort {
    param([int]$Port, [string]$Service)
    $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $Port)
    $probe.Server.ExclusiveAddressUse = $true
    try {
        $probe.Start()
    }
    catch {
        throw "$Service port $Port is unavailable. Stop the service using it yourself, or choose another -${Service}Port."
    }
    finally {
        $probe.Stop()
    }
}

if ($env:OS -ne 'Windows_NT') {
    throw 'This launcher is for Windows PowerShell. On other platforms, use the two-terminal commands in README.md.'
}
if ($BackendPort -eq $FrontendPort) {
    throw 'BackendPort and FrontendPort must be different.'
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$frontendRoot = Join-Path $projectRoot 'frontend'
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw 'The project virtual environment is missing. Follow the isolated, hash-locked installation steps in README.md, then run this launcher again.'
}
$nodeCommand = Get-Command node -ErrorAction SilentlyContinue
$pnpmCommand = Get-Command pnpm -ErrorAction SilentlyContinue
if ($null -eq $nodeCommand -or $null -eq $pnpmCommand) {
    throw 'Install Node.js 22 or newer and pnpm 10 or newer, then run: pnpm --dir frontend install --frozen-lockfile'
}
$nodeVersion = [version]((& $nodeCommand.Source --version).Trim().TrimStart('v'))
$pnpmVersion = [version]((& $pnpmCommand.Source --version).Trim())
if ($nodeVersion.Major -lt 22 -or $pnpmVersion.Major -lt 10) {
    throw 'LuminRAG needs Node.js 22 or newer and pnpm 10 or newer. Update them, then install the frontend dependencies.'
}
if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules\.bin\vite.cmd') -PathType Leaf)) {
    throw 'Frontend dependencies are missing. From the project folder run: pnpm --dir frontend install --frozen-lockfile'
}

$dependencyCheck = @'
import importlib.util
import sys
if sys.version_info < (3, 12):
    print("Python 3.12 or newer is required in .venv.")
    raise SystemExit(1)
modules = ("fastapi", "uvicorn", "fastembed", "numpy", "networkx", "anthropic", "openai", "instructor", "markitdown", "pymupdf", "limits", "defusedxml", "multipart", "yaml", "PIL")
missing = [name for name in modules if importlib.util.find_spec(name) is None]
if missing:
    print("Missing backend packages: " + ", ".join(missing))
    raise SystemExit(1)
'@
$dependencyResult = & $pythonPath -c $dependencyCheck
if ($LASTEXITCODE -ne 0) {
    throw "$dependencyResult Install the locked backend packages with: .venv\Scripts\python.exe -m pip install --require-hashes --only-binary=:all: -r requirements-bootstrap.lock -r requirements-hosted.lock"
}

Assert-AvailablePort -Port $BackendPort -Service 'Backend'
Assert-AvailablePort -Port $FrontendPort -Service 'Frontend'

$backendUrl = "http://127.0.0.1:$BackendPort"
$frontendUrl = "http://127.0.0.1:$FrontendPort"
$logDirectory = Join-Path $projectRoot ('.tmp\local-run\' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$stdoutPath = Join-Path $logDirectory 'backend.stdout.log'
$stderrPath = Join-Path $logDirectory 'backend.stderr.log'
$environmentNames = @('HOST', 'PORT', 'LUMIN_HOSTED', 'LUMIN_RELOAD', 'LUMINRAG_SKIP_DOTENV', 'ALLOWED_ORIGINS', 'VITE_API_URL')
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
$backendProcess = $null

try {
    $env:HOST = '127.0.0.1'
    $env:PORT = [string]$BackendPort
    $env:LUMIN_HOSTED = if ($SelfHosted) { '0' } else { '1' }
    $env:LUMIN_RELOAD = '0'
    if (-not $SelfHosted) {
        $env:LUMINRAG_SKIP_DOTENV = '1'
    }
    $env:ALLOWED_ORIGINS = "$frontendUrl,http://localhost:$FrontendPort"
    $env:VITE_API_URL = $backendUrl

    Write-Host "Starting the backend on $backendUrl ..."
    Write-Host 'First startup may take a few minutes to download the local retrieval models.'
    $backendProcess = Start-Process -FilePath $pythonPath -ArgumentList @('-m', 'backend') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    $deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $ready = $false
    while ([DateTime]::UtcNow -lt $deadline) {
        $backendProcess.Refresh()
        if ($backendProcess.HasExited) {
            throw "The backend stopped during startup. Inspect $stderrPath. If packages are missing, follow the hash-locked installation steps in README.md."
        }
        try {
            $response = Invoke-RestMethod -Uri "$backendUrl/api/ready" -Method Get -TimeoutSec 2
            if ($response.status -eq 'ready') {
                $ready = $true
                break
            }
        }
        catch {
            # Not ready yet. Never print response bodies or provider details.
        }
        Start-Sleep -Milliseconds 500
    }
    if (-not $ready) {
        throw "Backend startup timed out. Inspect $stderrPath, or retry with a larger -StartupTimeoutSeconds value."
    }

    Write-Host "Open $frontendUrl after the frontend starts. Press Ctrl+C to stop both services."
    if ($SelfHosted) {
        Write-Host 'Self-hosted mode uses the persistent local knowledge base and normal backend configuration.'
    }
    else {
        Write-Host 'Public mode keeps knowledge bases in your browser. Connect your provider in AI settings.'
    }
    & $pnpmCommand.Source --dir $frontendRoot dev --host 127.0.0.1 --port $FrontendPort --strictPort
    if ($LASTEXITCODE -ne 0) {
        throw 'The frontend stopped with an error. Review its output above; the backend has been stopped.'
    }
}
finally {
    try {
        if ($null -ne $backendProcess) {
            $backendProcess.Refresh()
            if (-not $backendProcess.HasExited) {
                # The venv launcher can own a Python child. Target only this process
                # tree; never find or stop a process by its port or executable name.
                & "$env:SystemRoot\System32\taskkill.exe" /PID $backendProcess.Id /T /F 2>$null | Out-Null
                $backendProcess.WaitForExit(5000) | Out-Null
            }
            $backendProcess.Dispose()
        }
    }
    finally {
        foreach ($name in $environmentNames) {
            [Environment]::SetEnvironmentVariable($name, $previousEnvironment[$name], 'Process')
        }
    }
}
