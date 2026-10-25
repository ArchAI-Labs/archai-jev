# Kev (PyTorch) against archai_jev on the same checkpoint, in alternating blocks (order reversed in odd blocks), one
# process per contender per block. Same rules as run_blocks.ps1 (charger + best performance, machine kept awake, quiet
# machine before each process, events log at the end). Release wheel only.
#
#   scripts\bench_release\kev\run_kev_blocks.ps1 -Blocks 3 -Out $env:TEMP\bench-kev
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Blocks = 3,
    [int]$StartBlock = 0,
    [string]$Cache = "$env:TEMP\archai-jev-bench-kev",
    # an unpacked wheel to put first on PYTHONPATH; empty = the installed archai_jev
    [string]$Pkg = "",
    [switch]$AllowBalanced
)
$ErrorActionPreference = "Continue"
Add-Type -Namespace W -Name P -MemberDefinition '[System.Runtime.InteropServices.DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
$here = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$root = (Resolve-Path "$here\..\..\..").Path
$pyAj = "$root\.venv\Scripts\python.exe"
$pyKev = $env:KEV_PYTHON   # the python.exe of the environment that has torch, transformers, peft (and KEV_SRC set)
if (-not $pyKev -or -not $env:KEV_SRC) { throw "set KEV_PYTHON and KEV_SRC (see README, Performance)" }

# label, runner, extra args, environment
$list = @(
    @{ label = "kev-default";   runner = "kev"; args = @();                                       env = @{} },
    @{ label = "kev-t8";        runner = "kev"; args = @("--intra1", "8", "--intra4", "5", "--intra16", "1"); env = @{ OMP_NUM_THREADS = "8" } },
    @{ label = "ajev-bf16";     runner = "aj";  args = @("--dtype", "bf16");                      env = @{} },
    @{ label = "ajev-bf16-aa";  runner = "aj";  args = @("--dtype", "bf16");                      env = @{} },
    @{ label = "ajev-bf16-t8";  runner = "aj";  args = @("--dtype", "bf16");                      env = @{ ARCHAI_JEV_NUM_THREADS = "8" } },
    @{ label = "ajev-f32";      runner = "aj";  args = @("--dtype", "f32");                       env = @{} }
)

$plan = (powercfg /getactivescheme) -join " "
$overlay = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes" -ErrorAction SilentlyContinue).ActiveOverlayAcPowerScheme
$highPerf = ($plan -match "8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c|e9a42b02-d5df-448d-aa00-03f14749eb61") -or ($overlay -eq "ded574b5-45a0-4f42-8737-46345c09c238")
$battery = Get-CimInstance Win32_Battery | Select-Object -First 1
$onAc = ($null -eq $battery) -or ($battery.BatteryStatus -ne 1)
$valid = $highPerf -and $onAc
if (-not $valid -and -not $AllowBalanced) { throw "NOT a recordable run: power plan = '$plan', on AC = $onAc" }
if ([W.P]::SetThreadExecutionState([uint32]2147483649) -eq 0) { throw "SetThreadExecutionState failed" }
$tag = if ($StartBlock -eq 0) { "" } else { "_from$StartBlock" }
if (Test-Path "$Out\environment$tag.json") { throw "$Out already holds this run: refusing to overwrite raw data" }
New-Item -ItemType Directory -Force $Out | Out-Null
$started = Get-Date
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$load = 1..5 | ForEach-Object { Start-Sleep -Seconds 1; (Get-CimInstance Win32_Processor | Select-Object -First 1).LoadPercentage }
$top = Get-Process | Sort-Object CPU -Descending | Select-Object -First 8 Name, CPU | ForEach-Object { "$($_.Name)=$([int]$_.CPU)" }
@{ date = $started.ToString("o"); valid_for_record = $valid; cpu = $cpu.Name; logical = $cpu.NumberOfLogicalProcessors
   power_plan = $plan; power_mode_overlay_ac = $overlay; on_ac_power = $onAc; battery_percent = $battery.EstimatedChargeRemaining
   idle_load_percent = $load; busiest_processes = $top; blocks = $Blocks; contenders = ($list | ForEach-Object { $_.label }) } |
    ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 "$Out\environment$tag.json"

function Wait-Quiet {
    for ($i = 0; $i -lt 20; $i++) {
        $s = 1..5 | ForEach-Object { Start-Sleep -Seconds 1; (Get-CimInstance Win32_Processor | Select-Object -First 1).LoadPercentage }
        if (($s | Measure-Object -Average).Average -le 6) { return }
        Start-Sleep -Seconds 10
    }
    Write-Warning "the machine stayed busy for 5 minutes: measuring anyway"
}

for ($b = $StartBlock; $b -lt $StartBlock + $Blocks; $b++) {
    $order = if ($b % 2 -eq 0) { $list } else { $list[($list.Count - 1)..0] }
    foreach ($m in $order) {
        foreach ($name in @("OMP_NUM_THREADS", "ARCHAI_JEV_NUM_THREADS", "PYTHONPATH", "ARCHAI_JEV_CACHE", "ARCHAI_JEV_OFFLINE", "HF_HUB_OFFLINE")) { Remove-Item "env:$name" -ErrorAction SilentlyContinue }
        Wait-Quiet
        $os0 = Get-CimInstance Win32_OperatingSystem
        "block $b $($m.label): free commit $([math]::Round($os0.FreeVirtualMemory/1MB,1)) GiB, free RAM $([math]::Round($os0.FreePhysicalMemory/1MB,1)) GiB" | Add-Content "$Out\headroom.txt"
        foreach ($k in $m.env.Keys) { Set-Item "env:$k" $m.env[$k] }
        $env:KB_T0 = [string][double]([DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0)
        $file = "$Out\block${b}_$($m.label).json"
        Write-Host "== block $b : $($m.label)"
        if ($m.runner -eq "kev") {
            & $pyKev "$here\kev_torch.py" --label $m.label --out $file @($m.args) 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$Out\block${b}_$($m.label).log" | Select-String "done|Error|Traceback|startup"
        } else {
            if ($Pkg) { $env:PYTHONPATH = $Pkg }; $env:ARCHAI_JEV_CACHE = $Cache; $env:ARCHAI_JEV_OFFLINE = "1"
            & $pyAj "$here\archai_kev.py" --label $m.label --out $file @($m.args) 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath "$Out\block${b}_$($m.label).log" | Select-String "done|Error|Traceback|load"
        }
        if (-not (Test-Path $file)) {
            # a failed process is data (the commit headroom is written next to it), not a reason to lose the rest of the run
            $os = Get-CimInstance Win32_OperatingSystem
            "block $b $($m.label): no result; free commit $([math]::Round($os.FreeVirtualMemory/1MB,1)) GiB, free RAM $([math]::Round($os.FreePhysicalMemory/1MB,1)) GiB" | Add-Content "$Out\failures.txt"
            Write-Warning "no result for block $b $($m.label): see $Out\block${b}_$($m.label).log"
        }
    }
}
$ended = Get-Date
$events = Get-WinEvent -FilterHashtable @{ LogName = 'System'; StartTime = $started; EndTime = $ended.AddSeconds(5) } -ErrorAction SilentlyContinue |
    Where-Object { $_.Id -in 42, 107 }
@{ started = $started.ToString("o"); ended = $ended.ToString("o"); suspend_events = @($events).Count } | ConvertTo-Json | Set-Content -Encoding utf8 "$Out\events$tag.json"
"DONE" | Set-Content "$Out\DONE$tag"
