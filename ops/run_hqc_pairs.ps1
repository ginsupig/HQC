<#
.SYNOPSIS
    HQC Kalman pairs launcher (Windows, Alpaca paper).

.DESCRIPTION
    Invoked by the 'hqc-pairs' scheduled task on weekday mornings. main_pairs.py
    runs until it is stopped and flattens at 15:55 ET via EODLiquidationManager,
    so this launcher:
      * refuses to start a second copy if one is already running,
      * refuses to start unless .env holds Alpaca keys,
      * runs `python -u main_pairs.py --config config/pairs.yaml` from .venv,
        writing stdout/stderr to dated files under state\logs,
      * stops the process at 16:05 ET (after the EOD flatten) if still running.
#>

[CmdletBinding()]
param(
    [string]$RepoDir = '',
    [string]$Config = 'config\pairs.yaml',
    [int]$StopHourEt = 16,
    [int]$StopMinuteEt = 5
)

$ErrorActionPreference = 'Stop'
if (-not $RepoDir) { $RepoDir = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path) }

$LogDir = Join-Path $RepoDir 'state\logs'
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$day = Get-Date -Format 'yyyy-MM-dd'
$LogFile = Join-Path $LogDir "launcher_$day.log"

function Write-Log {
    param([string]$Message)
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    "$stamp  $Message" | Tee-Object -FilePath $LogFile -Append
}

$mutex = New-Object System.Threading.Mutex($false, 'Global\hqc_pairs_runtime')
if (-not $mutex.WaitOne(0)) {
    Write-Log 'HQC pairs already running - skipping this launch.'
    exit 0
}

try {
    Set-Location $RepoDir

    $envFile = Join-Path $RepoDir '.env'
    $keys = @{}
    if (Test-Path $envFile) {
        foreach ($line in Get-Content $envFile) {
            if ($line -match '^\s*(ALPACA_API_KEY|ALPACA_API_SECRET)\s*=\s*(\S+)') { $keys[$Matches[1]] = $Matches[2] }
        }
    }
    if (-not $keys['ALPACA_API_KEY'] -or -not $keys['ALPACA_API_SECRET']) {
        Write-Log 'ALPACA_API_KEY / ALPACA_API_SECRET missing from .env - not starting.'
        exit 1
    }

    $python = Join-Path $RepoDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $python)) {
        Write-Log "No virtualenv at $python - not starting."
        exit 1
    }

    $et = [TimeZoneInfo]::FindSystemTimeZoneById('Eastern Standard Time')
    $nowEt = [TimeZoneInfo]::ConvertTime((Get-Date), $et)
    $stopEt = $nowEt.Date.AddHours($StopHourEt).AddMinutes($StopMinuteEt)
    if ($nowEt -ge $stopEt) {
        Write-Log 'Past today''s stop time - not starting.'
        exit 0
    }
    $stopLocal = [TimeZoneInfo]::ConvertTime($stopEt, $et, [TimeZoneInfo]::Local)

    $env:PYTHONUTF8 = '1'
    $out = Join-Path $LogDir "pairs_$day.out.log"
    $err = Join-Path $LogDir "pairs_$day.err.log"
    Write-Log "Starting HQC pairs ($Config); stop at $($stopEt.ToString('HH:mm')) ET."
    $proc = Start-Process -FilePath $python -ArgumentList @('-u', 'main_pairs.py', '--config', $Config) `
        -WorkingDirectory $RepoDir -RedirectStandardOutput $out -RedirectStandardError $err `
        -NoNewWindow -PassThru

    while (-not $proc.HasExited -and (Get-Date) -lt $stopLocal) {
        Start-Sleep -Seconds 30
    }
    if (-not $proc.HasExited) {
        Write-Log 'Stop time reached - stopping HQC pairs.'
        Stop-Process -Id $proc.Id -Force
        $proc.WaitForExit()
        exit 0
    }
    Write-Log "HQC pairs exited on its own with code $($proc.ExitCode)."
    exit $proc.ExitCode
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
