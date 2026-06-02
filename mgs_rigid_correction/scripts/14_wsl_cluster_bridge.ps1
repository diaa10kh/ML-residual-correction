param(
    [int]$PollSeconds = 2
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$BridgeDir = Join-Path $ProjectRoot ".codex_cluster_bridge"
$RequestPath = Join-Path $BridgeDir "request.json"
$StatusPath = Join-Path $BridgeDir "status.json"
$OutputPath = Join-Path $BridgeDir "last_output.log"

$ClusterUser = if ($env:MGS_CLUSTER_USER) { $env:MGS_CLUSTER_USER } else { "cda6556" }
$ClusterHost = if ($env:MGS_CLUSTER_HOST) { $env:MGS_CLUSTER_HOST } else { "hpc2.rz.tuhh.de" }
$HypoRemote = if ($env:MGS_HYPO_REMOTE) { $env:MGS_HYPO_REMOTE } else { "/work/gbt/$ClusterUser/MGS_Rigid_Correction/FULL_HYPO" }
$McRemote = if ($env:MGS_MC_REMOTE) { $env:MGS_MC_REMOTE } else { "/work/gbt/$ClusterUser/MGS_Rigid_Correction/FULL_MC" }

New-Item -ItemType Directory -Force -Path $BridgeDir | Out-Null

function ConvertTo-BashSingleQuoted {
    param([string]$Value)
    return "'" + $Value.Replace("'", "'\''") + "'"
}

function Write-BridgeStatus {
    param(
        [string]$State,
        [string]$Action,
        [string]$Message,
        [int]$ExitCode = 0
    )

    [ordered]@{
        timestamp = (Get-Date).ToString("s")
        state = $State
        action = $Action
        message = $Message
        exit_code = $ExitCode
        output_log = $OutputPath
    } | ConvertTo-Json | Set-Content -Path $StatusPath -Encoding UTF8
}

function Invoke-Logged {
    param(
        [string]$Action,
        [string[]]$Command
    )

    Set-Location $ProjectRoot
    Write-BridgeStatus -State "running" -Action $Action -Message "Started."

    $lines = New-Object System.Collections.Generic.List[string]
    $exitCode = 0

    try {
        $commandArgs = @()
        if ($Command.Count -gt 1) {
            $commandArgs = $Command[1..($Command.Count - 1)]
        }
        $output = & $Command[0] @commandArgs 2>&1
        $exitCode = $LASTEXITCODE
        foreach ($line in $output) {
            $lines.Add([string]$line)
        }
    }
    catch {
        $exitCode = 1
        $lines.Add($_.Exception.Message)
    }

    $lines | Set-Content -Path $OutputPath -Encoding UTF8

    if ($exitCode -eq 0) {
        Write-BridgeStatus -State "completed" -Action $Action -Message "Completed." -ExitCode $exitCode
    }
    else {
        Write-BridgeStatus -State "failed" -Action $Action -Message "Failed. Check last_output.log." -ExitCode $exitCode
    }
}

function Invoke-WslProjectBash {
    param(
        [string]$Action,
        [string]$BashCommand
    )

    $wslProjectRoot = (& wsl wslpath -a $ProjectRoot 2>&1)
    if ($LASTEXITCODE -ne 0) {
        throw "Could not map project path into WSL: $wslProjectRoot"
    }

    $quotedProjectRoot = ConvertTo-BashSingleQuoted -Value ([string]$wslProjectRoot).Trim()
    Invoke-Logged -Action $Action -Command @("wsl", "bash", "-lc", "cd $quotedProjectRoot && $BashCommand")
}

function Invoke-Action {
    param($Request)

    $action = [string]$Request.action
    switch ($action) {
        "status" {
            Invoke-Logged -Action $action -Command @("wsl", "ssh", "$ClusterUser@$ClusterHost", "squeue -u $ClusterUser")
        }
        "cancel_all" {
            Invoke-Logged -Action $action -Command @("wsl", "ssh", "$ClusterUser@$ClusterHost", "scancel -u $ClusterUser; squeue -u $ClusterUser")
        }
        "cancel_job" {
            if (-not $Request.job_id) {
                throw "cancel_job requires job_id."
            }
            Invoke-Logged -Action $action -Command @("wsl", "ssh", "$ClusterUser@$ClusterHost", "scancel $($Request.job_id); squeue -u $ClusterUser")
        }
        "submit_hypo" {
            $remote = ConvertTo-BashSingleQuoted -Value $HypoRemote
            Invoke-WslProjectBash -Action $action -BashCommand "MGS_MATRIX_CSV=data/extracted/run_metadata_full.csv MGS_ROOT_REMOTE=$remote bash ./scripts/12_cluster_transfer_submit_fetch.sh submit_all"
        }
        "submit_mc" {
            $remote = ConvertTo-BashSingleQuoted -Value $McRemote
            Invoke-WslProjectBash -Action $action -BashCommand "MGS_MATRIX_CSV=data/extracted/run_metadata_full_mohr_coulomb.csv MGS_ROOT_REMOTE=$remote MGS_SLURM_JOB_NAME=mgs_full_mcm MGS_PROJECT_NAME=MGS_Full_Mohr_Coulomb bash ./scripts/12_cluster_transfer_submit_fetch.sh submit_all"
        }
        "fetch_hypo" {
            $remote = ConvertTo-BashSingleQuoted -Value $HypoRemote
            Invoke-WslProjectBash -Action $action -BashCommand "MGS_MATRIX_CSV=data/extracted/run_metadata_full.csv MGS_ROOT_REMOTE=$remote bash ./scripts/12_cluster_transfer_submit_fetch.sh fetch"
        }
        "fetch_mc" {
            $remote = ConvertTo-BashSingleQuoted -Value $McRemote
            Invoke-WslProjectBash -Action $action -BashCommand "MGS_MATRIX_CSV=data/extracted/run_metadata_full_mohr_coulomb.csv MGS_ROOT_REMOTE=$remote bash ./scripts/12_cluster_transfer_submit_fetch.sh fetch"
        }
        "exit" {
            Write-BridgeStatus -State "stopped" -Action $action -Message "Bridge stopped by request."
            exit 0
        }
        default {
            throw "Unsupported action '$action'. Allowed actions: status, cancel_all, cancel_job, submit_hypo, submit_mc, fetch_hypo, fetch_mc, exit."
        }
    }
}

$lastId = ""
Write-BridgeStatus -State "waiting" -Action "" -Message "Waiting for request.json."
Write-Host "WSL cluster bridge is waiting for requests."
Write-Host "Project: $ProjectRoot"
Write-Host "Request file: $RequestPath"
Write-Host "Press Ctrl+C to stop."

while ($true) {
    if (Test-Path $RequestPath) {
        try {
            $request = Get-Content -Path $RequestPath -Raw | ConvertFrom-Json
            $requestId = [string]$request.id
            if ($requestId -and $requestId -ne $lastId) {
                $lastId = $requestId
                Invoke-Action -Request $request
                if (([string]$request.action) -ne "exit") {
                    Write-BridgeStatus -State "waiting" -Action ([string]$request.action) -Message "Waiting for next request."
                }
            }
        }
        catch {
            $_.Exception.Message | Set-Content -Path $OutputPath -Encoding UTF8
            Write-BridgeStatus -State "failed" -Action "parse_or_dispatch" -Message $_.Exception.Message -ExitCode 1
        }
    }

    Start-Sleep -Seconds $PollSeconds
}
