param(
    [string]$BatchDirectory = ".\data\processed\tatoeba_review_batch_0001",
    [string]$Reviewer = "Suborno",
    [int]$Limit = 200
)

$ErrorActionPreference = "Stop"

$QueuePath = Join-Path $BatchDirectory "review_queue.jsonl"
$ProgressPath = Join-Path $BatchDirectory "review_progress.jsonl"

if (!(Test-Path $QueuePath)) {
    throw "Review queue not found: $QueuePath"
}

$Levels = @(
    "basic_english",
    "elementary_intermediate_english",
    "upper_intermediate_english",
    "advanced_english"
)

$Queue = @(
    Get-Content $QueuePath -Encoding UTF8 |
    ForEach-Object { $_ | ConvertFrom-Json }
)

$Reviewed = @{}
if (Test-Path $ProgressPath) {
    foreach ($line in Get-Content $ProgressPath -Encoding UTF8) {
        if ($line.Trim()) {
            $item = $line | ConvertFrom-Json
            $Reviewed[[string]$item.sentence_id] = $item
        }
    }
}

$Pending = @(
    $Queue | Where-Object {
        !$Reviewed.ContainsKey([string]$_.sentence_id)
    }
)

Write-Host "`n========== VYREX AI HUMAN REVIEW ==========" -ForegroundColor Cyan
Write-Host "Total sentences : $($Queue.Count)"
Write-Host "Already reviewed: $($Reviewed.Count)"
Write-Host "Remaining       : $($Pending.Count)"
Write-Host "Session limit   : $Limit"
Write-Host "`nKeyboard controls:"
Write-Host "1 = Basic English"
Write-Host "2 = Elementary / Intermediate"
Write-Host "3 = Upper-intermediate"
Write-Host "4 = Advanced"
Write-Host "R = Needs review"
Write-Host "X = Reject"
Write-Host "S = Skip temporarily"
Write-Host "Q = Save and quit"

$count = 0

foreach ($item in $Pending) {
    if ($count -ge $Limit) { break }

    Clear-Host
    Write-Host "========== VYREX AI ENGLISH REVIEW ==========" -ForegroundColor Cyan
    Write-Host "Reviewed this session: $count / $Limit"
    Write-Host "Sentence ID: $($item.sentence_id)" -ForegroundColor Yellow
    Write-Host "License: $($item.license)"
    Write-Host "Author: $($item.author)"
    Write-Host "`nSENTENCE:" -ForegroundColor Green
    Write-Host $item.text -ForegroundColor White
    Write-Host "`nChoose classification:"
    Write-Host "1. Basic English"
    Write-Host "2. Elementary / Intermediate"
    Write-Host "3. Upper-intermediate"
    Write-Host "4. Advanced"
    Write-Host "R. Needs review"
    Write-Host "X. Reject"
    Write-Host "S. Skip"
    Write-Host "Q. Quit"

    $choice = (Read-Host "`nYour decision").Trim().ToUpperInvariant()

    if ($choice -eq "Q") { break }
    if ($choice -eq "S") { continue }

    $status = ""
    $level = ""

    switch ($choice) {
        "1" { $status = "approved"; $level = $Levels[0] }
        "2" { $status = "approved"; $level = $Levels[1] }
        "3" { $status = "approved"; $level = $Levels[2] }
        "4" { $status = "approved"; $level = $Levels[3] }
        "R" { $status = "needs_review" }
        "X" { $status = "rejected" }
        default {
            Write-Host "Invalid choice. Sentence skipped." -ForegroundColor Red
            Start-Sleep -Seconds 1
            continue
        }
    }

    $rationale = ""
    if ($status -eq "approved") {
        do {
            $rationale = (Read-Host "Why does this sentence fit this level?").Trim()
        } while (!$rationale)
    }
    elseif ($status -eq "needs_review") {
        $rationale = (Read-Host "Why does this need further review?").Trim()
    }
    else {
        $rationale = (Read-Host "Reason for rejection").Trim()
    }

    $decision = [ordered]@{
        sentence_id = [string]$item.sentence_id
        text = [string]$item.text
        license = [string]$item.license
        author = [string]$item.author
        attribution_source = if ($item.license -eq "CC-BY-2.0-FR") { "Tatoeba" } else { "" }
        source_archive = [string]$item.source_archive
        source_sha256 = [string]$item.source_sha256
        review_status = $status
        assigned_level = $level
        rationale = $rationale
        reviewer = $Reviewer
        reviewed_at = (Get-Date).ToString("o")
    }

    $json = $decision | ConvertTo-Json -Compress -Depth 5
    Add-Content -Path $ProgressPath -Value $json -Encoding UTF8

    $Reviewed[[string]$item.sentence_id] = $decision
    $count++

    Write-Host "`nDecision saved." -ForegroundColor Green
    Start-Sleep -Milliseconds 400
}

Write-Host "`n========== SESSION SUMMARY ==========" -ForegroundColor Cyan
Write-Host "Reviewed this session: $count"
Write-Host "Total recorded decisions: $($Reviewed.Count)"
Write-Host "Remaining: $($Queue.Count - $Reviewed.Count)"
Write-Host "Progress file: $ProgressPath"
