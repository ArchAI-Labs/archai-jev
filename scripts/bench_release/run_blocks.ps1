# Blocks of one process per contender, in alternating order, one model per process .
# The order of the contenders is reversed in odd blocks. Release builds only. The machine is kept awake with a
# checked per-process request (the hex literal 0x80000001 is a negative Int32 in PowerShell: it is written as an
# unsigned number) and the system event log is read at the end: a suspend or a big clock change inside a block makes
# its numbers worthless.
#
#   scripts\bench_release\run_blocks.ps1 -Contenders scripts\bench_release\contenders_default.json -Blocks 3 -Out $env:TEMP\bench-default
#
# The contenders file is a JSON list: [{ "label": "cpu-t8", "pkg": "C:\\path\\to\\unpacked-wheel",
#   "env": { "ARCHAI_JEV_NUM_THREADS": "8" }, "fresh_env_dir": "CUDA_CACHE_PATH" }, ...]
# `pkg` is an unpacked wheel (PYTHONPATH); `fresh_env_dir` names an environment variable that gets a NEW EMPTY
# folder for every process (the kernel/shader cache, to measure the first use without it).
#
# Preconditions that decide whether the numbers mean anything, CHECKED here and written in environment.json:
# the charger connected and the "High performance" (or "Ultimate performance") power plan. The plan "Balanced" is
# the documented cause of the two CPU latency regimes . -AllowBalanced runs anyway and marks the run
# valid_for_record = false (only to try the tools).
param(
    [Parameter(Mandatory = $true)][string]$Contenders,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Blocks = 3,
    [int]$StartBlock = 0,
    [string]$Cache = "$env:TEMP\archai-jev-bench",
    [string]$Sections = "",
    [string]$Regimes = "",
    [switch]$AllowBalanced
)
# "Continue", not "Stop": the model prints its notice on stderr and `2>&1` would turn it into a terminating error.
# Every real failure below is an explicit `throw`.
$ErrorActionPreference = "Continue"
Add-Type -Namespace W -Name P -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$root = (Resolve-Path "$here\..\..").Path
$py = "$root\.venv\Scripts\python.exe"
$list = Get-Content $Contenders -Raw | ConvertFrom-Json
if ($Blocks -lt 3) { throw "at least 3 blocks (common.MIN_BLOCKS)" }

$plan = (powercfg /getactivescheme) -join " "
# Windows 11 keeps the scheme "Balanced" and puts the "power mode" slider on top of it as an OVERLAY; "Best performance"
# (ded574b5-...) on AC is the modern High performance. Both are recorded.
$overlay = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction SilentlyContinue).ActiveOverlayAcPowerScheme
$highPerf = ($plan -match "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c|e9a42b02-d5df-448d-aa00-03f14749eb61") -or ($overlay -eq "ded574b5-45a0-4f42-8737-46345c09c238")
$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
$onAc = ($null -eq $battery) -or ($battery.BatteryStatus -ne 1)
$valid = $highPerf -and $onAc
if (-not $valid -and -not $AllowBalanced) {
    throw "NOT a recordable run: power plan = '$plan', on AC = $onAc. Plug in the charger and switch to High performance " +
        "(powercfg -duplicatescheme 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c, then select it), or pass -AllowBalanced to just try the tools."
}
if ([W.P]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw "SetThreadExecutionState failed: the machine is NOT kept awake" }
$tag = if ($StartBlock -eq 0) { "" } else { "_from$StartBlock" }
if (Test-Path "$Out\environment$tag.json") { throw "$Out already holds this run: refusing to overwrite raw data" }
New-Item -ItemType Directory -Force $Out | Out-Null
$started = Get-Date

$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$load = 1..5 | ForEach-Object { Start-Sleep -Seconds 1; (Get-CimInstance Win32_Processor | Select-Object -First 1).LoadPercentage }
$top = Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 Name, CPU | ForEach-Object { "$($_.Name)=$([int]$_.CPU)" }
$smi = try { (nvidia-smi --query-gpu=name,driver_version,pstate,clocks.sm,clocks.max.sm,power.draw,temperature.gpu,memory.used --format=csv,noheader) -join " | " } catch { "no nvidia-smi" }
@{
    date = $started.ToString("o"); valid_for_record = $valid; cpu = $cpu.Name; logical = $cpu.NumberOfLogicalProcessors
    cpu_clock_mhz = $cpu.CurrentClockSpeed; cpu_max_clock_mhz = $cpu.MaxClockSpeed
    power_plan = $plan; power_mode_overlay_ac = $overlay; high_performance = $highPerf; on_ac_power = $onAc; battery_percent = $battery.EstimatedChargeRemaining
    idle_load_percent = $load; busiest_processes = $top; nvidia_smi = $smi
    contenders = $list; blocks = $Blocks; cache = $Cache; sections = $Sections; regimes = $Regimes
} | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$Out\environment$tag.json"
if (-not $valid) { Write-Warning "-AllowBalanced: this run is marked valid_for_record = false" }

$env:ARCHAI_JEV_CACHE = $Cache
$env:ARCHAI_JEV_OFFLINE = "1"
$extra = @()
if ($Sections) { $extra += @("--sections", $Sections) }
if ($Regimes) { $extra += @("--regimes", $Regimes) }

# Before each process, wait (up to 5 minutes) until the machine is quiet: the mean of 5 one-second load samples at or
# below 6 % (about one core of 20). The bench also measures the foreign load DURING each cell and the summary leaves out
# the cells above common.NOISY_FOREIGN_CORES; this only avoids wasting a process on a machine that is clearly busy.
function Wait-Quiet {
    for ($i = 0; $i -lt 20; $i++) {
        $s = 1..5 | ForEach-Object { Start-Sleep -Seconds 1; (Get-CimInstance Win32_Processor | Select-Object -First 1).LoadPercentage }
        $mean = ($s | Measure-Object -Average).Average
        if ($mean -le 6) { return }
        $top = Get-Process | Sort-Object CPU -Descending | Select-Object -First 3 -ExpandProperty ProcessName
        Write-Host "   machine busy ($([math]::Round($mean)) % load; $($top -join ', ')): waiting"
        Start-Sleep -Seconds 10
    }
    Write-Warning "the machine stayed busy for 5 minutes: measuring anyway (the foreign-load meter will say)"
}

for ($b = $StartBlock; $b -lt $StartBlock + $Blocks; $b++) {
    $order = if ($b % 2 -eq 0) { $list } else { $list[($list.Count - 1)..0] }
    foreach ($m in $order) {
        # a clean environment for every process: what the previous contender set must not leak into the next one
        foreach ($name in @("S14_DEVICE", "S14_OP_OFFLOAD", "S14_PHASES", "S14_LIST", "ARCHAI_JEV_NUM_THREADS", "CUDA_CACHE_PATH", "CUDA_VISIBLE_DEVICES", "S14_DLL_DIR", "GGML_VK_VISIBLE_DEVICES")) {
            Remove-Item "env:$name" -ErrorAction SilentlyContinue
        }
        $env:ARCHAI_JEV_CACHE = $Cache
        Wait-Quiet
        $env:PYTHONPATH = $m.pkg
        if ($m.env) { foreach ($p in $m.env.PSObject.Properties) { Set-Item "env:$($p.Name)" $p.Value } }
        if ($m.fresh_env_dir) {
            $dir = Join-Path $env:TEMP ("s14-fresh-" + [guid]::NewGuid().ToString("N"))
            New-Item -ItemType Directory $dir | Out-Null
            Set-Item "env:$($m.fresh_env_dir)" $dir
        }
        $env:S14_T0 = [string][double]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0)
        $file = "$Out\block${b}_$($m.label).json"
        Write-Host "== block $b : $($m.label)"
        & $py "$here\bench_latency.py" --label $m.label --out $file @extra 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$Out\block${b}_$($m.label).log" | Select-String "done in|Error|Traceback|loaded"
        if (-not (Test-Path $file)) { throw "no result for block $b $($m.label): see $Out\block${b}_$($m.label).log" }
        if ($m.fresh_env_dir) { Remove-Item (Get-Item "env:$($m.fresh_env_dir)").Value -Recurse -Force -ErrorAction SilentlyContinue }
    }
}
$ended = Get-Date
$events = Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $started; EndTime = $ended.AddSeconds(5) } -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -in 42, 107 -or ($_.ProviderName -match 'Kernel-General' -and $_.Id -eq 1 -and ([int]([regex]::Match($_.Message, '(?:Delta ora|Time delta): (\d+) ms').Groups[1].Value) -ge 5000)) }
@{ started = $started.ToString("o"); ended = $ended.ToString("o"); suspend_or_clock_events = @($events).Count } |
    ConvertTo-Json | Set-Content -Encoding utf8 "$Out\events$tag.json"
"DONE" | Set-Content "$Out\DONE$tag"
