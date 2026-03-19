param(
    [Parameter(Mandatory = $true)]
    [string]$ServerExe,

    [Parameter(Mandatory = $true)]
    [string]$TorchLibDir,

    [Parameter(Mandatory = $true)]
    [string]$PidFile
)

$ErrorActionPreference = 'Stop'

if (-not (Test-Path -Path $ServerExe)) {
    Write-Error "Server executable not found: $ServerExe"
}

$env:PATH = "$TorchLibDir;$env:PATH"

$proc = Start-Process -FilePath $ServerExe -PassThru -WindowStyle Hidden
Start-Sleep -Milliseconds 300

if ($proc.HasExited) {
    Write-Error "Server exited immediately with code $($proc.ExitCode)"
}

Set-Content -Path $PidFile -Value $proc.Id -Encoding ascii
Write-Output "Started ABQnn IPC server pid=$($proc.Id)"
