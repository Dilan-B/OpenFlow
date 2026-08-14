<#
.SYNOPSIS
    Smoke-test a packaged OpenFlow build.

.DESCRIPTION
    Everything here is a failure that a source checkout cannot reproduce: paths
    resolved from __file__ move inside _internal/, sys.executable stops being an
    interpreter, and modules that were never collected only fail on the machine
    that downloaded the installer.

    The regression this exists for: "Start with Windows" wrote a Startup entry
    reading `OpenFlow.exe -m openflow --minimized`, argparse rejected it, and the
    app exited 2 at sign-in with no window and nothing in the log.

.EXAMPLE
    pwsh scripts/smoke_frozen_build.ps1 -ExePath dist/OpenFlow/OpenFlow.exe
#>
param(
    [string]$ExePath = "dist/OpenFlow/OpenFlow.exe",
    [int]$TimeoutSeconds = 120
)

$ErrorActionPreference = "Stop"
$failures = @()

function Invoke-Exe {
    <#
      Returns Exited/Code/Stdout/Stderr. Never throws on a still-running
      process: "it is still up" is the correct outcome for the GUI paths, and
      only the caller knows whether that is a pass.
    #>
    param([string[]]$Arguments, [int]$WaitSeconds = $TimeoutSeconds)
    $out = New-TemporaryFile
    $err = New-TemporaryFile
    try {
        $p = Start-Process -FilePath $ExePath -ArgumentList $Arguments -PassThru `
            -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow
        $exited = $p.WaitForExit($WaitSeconds * 1000)
        if ($exited) {
            # The timed overload can leave ExitCode stale and async output
            # unflushed; the argument-less call settles both.
            $p.WaitForExit()
        } else {
            try { $p.Kill($true) } catch { }
            try { $p.WaitForExit(5000) | Out-Null } catch { }
        }
        [pscustomobject]@{
            Exited = $exited
            Code   = $(if ($exited) { $p.ExitCode } else { $null })
            Stdout = (Get-Content $out -Raw)
            Stderr = (Get-Content $err -Raw)
        }
    } finally {
        Remove-Item $out, $err -Force -ErrorAction SilentlyContinue
    }
}

function Assert-ExitCode {
    param([string]$Label, [string[]]$Arguments, [int]$Expected)
    $r = Invoke-Exe -Arguments $Arguments
    if (-not $r.Exited) {
        Write-Host "  FAIL  $Label -- still running after ${TimeoutSeconds}s" -ForegroundColor Red
        $script:failures += $Label
        return $r
    }
    if ($r.Code -eq $Expected) {
        Write-Host "  PASS  $Label (exit $($r.Code))" -ForegroundColor Green
    } else {
        Write-Host "  FAIL  $Label -- expected exit $Expected, got $($r.Code)" -ForegroundColor Red
        if ($r.Stdout) { Write-Host "        stdout: $($r.Stdout.Trim())" }
        if ($r.Stderr) { Write-Host "        stderr: $($r.Stderr.Trim())" }
        $script:failures += $Label
    }
    return $r
}

if (-not (Test-Path $ExePath)) { throw "no packaged build at $ExePath" }
Write-Host "smoke-testing $ExePath`n"

# 1. The build's own view of whether it can launch itself. Covers launch target,
#    working directory, icon, std handles, and every module PyInstaller had to
#    collect.
$selfTest = Assert-ExitCode -Label "--self-test" -Arguments @("--self-test") -Expected 0
if ($selfTest.Stdout) { $selfTest.Stdout.TrimEnd() -split "`r?`n" | ForEach-Object { Write-Host "        $_" } }

# 2. The exact arguments the Startup shortcut carries, launched the way Windows
#    launches them at sign-in. Three outcomes, and only one is a failure:
#      still running  -- the app came up and is sitting in the tray (a developer
#                        machine with a real microphone). This is the good case.
#      exit 3         -- no audio device, normal on a CI runner. Arguments were
#                        still accepted, which is what this check is about.
#      exit 2         -- the app rejected its own startup shortcut. The bug.
$r = Invoke-Exe -Arguments @("--minimized") -WaitSeconds 25
if (-not $r.Exited) {
    Write-Host "  PASS  startup arguments accepted (app came up and kept running)" -ForegroundColor Green
} elseif ($r.Code -eq 2) {
    Write-Host "  FAIL  startup arguments rejected by the app's own parser (exit 2)" -ForegroundColor Red
    if ($r.Stderr) { Write-Host "        stderr: $($r.Stderr.Trim())" }
    $failures += "startup arguments"
} else {
    Write-Host "  PASS  startup arguments accepted (exit $($r.Code))" -ForegroundColor Green
}

# 3. The historical break, pinned so it cannot silently come back: interpreter
#    flags are NOT valid arguments to the frozen exe, so a launcher that emits
#    them must fail loudly here rather than at somebody's sign-in.
Assert-ExitCode -Label "interpreter flags still rejected" `
    -Arguments @("-m", "openflow", "--minimized") -Expected 2 | Out-Null

# 4. Ordinary CLI paths that must keep working in a windowed build, where
#    stdout/stderr are None and printing is the thing that used to crash.
Assert-ExitCode -Label "--clean" -Arguments @("--clean", "hello, um, world") -Expected 0 | Out-Null

if ($failures.Count) {
    Write-Host "`n$($failures.Count) smoke check(s) failed: $($failures -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "`nall smoke checks passed" -ForegroundColor Green
exit 0
