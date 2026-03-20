param(
    [Parameter(Mandatory = $true)]
    [string]$PidFile
)

$ErrorActionPreference = 'SilentlyContinue'

if (Test-Path -Path $PidFile) {
    $pidText = Get-Content -Path $PidFile | Select-Object -First 1
    if ($pidText) {
        $serverPid = [int]$pidText
        Stop-Process -Id $serverPid -Force
        Write-Output "Stopped ABQnn IPC server pid=$serverPid"
    }
    Remove-Item -Path $PidFile -Force
}
