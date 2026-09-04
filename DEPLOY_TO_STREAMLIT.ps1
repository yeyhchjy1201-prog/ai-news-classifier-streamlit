<#
.SYNOPSIS
Prepares this standalone package for Streamlit Community Cloud and pushes it to GitHub.

.DESCRIPTION
Checks that this folder has no local secret file, initializes or reuses only this
folder's Git repository, stages an allow-listed set of deployment files, optionally
pushes to a GitHub remote, then opens Streamlit Community Cloud for the user to
complete account-authorized deployment. This script never reads API-key values.

.PARAMETER GitHubRemote
Optional HTTPS or SSH Git remote. If omitted, the script opens GitHub's new
repository page and asks for the clone URL after the user creates an empty repository.

.PARAMETER PrepareOnly
Validates and creates the local Git commit but does not ask for a remote, push, or
open browser pages.

.PARAMETER NoOpenBrowser
Prevents the script from opening GitHub or Streamlit Community Cloud in a browser.

.PARAMETER GitUserName
Optional display name for the commit created in this deployment folder only.

.PARAMETER GitUserEmail
Optional email address for the commit created in this deployment folder only.

.NOTES
This file is ASCII-only for Windows PowerShell 5.1 compatibility.
#>

[CmdletBinding(SupportsShouldProcess = $true, ConfirmImpact = 'Medium')]
param(
    [string]$GitHubRemote,
    [switch]$PrepareOnly,
    [switch]$NoOpenBrowser,
    [string]$GitUserName,
    [string]$GitUserEmail
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Invoke-GitCommand {
    param(
        [Parameter(Mandatory = $true)][string]$GitExecutable,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    $safeDirectory = $PackageRoot.Replace('\', '/')
    & $GitExecutable -c "safe.directory=$safeDirectory" @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Git command failed: git $($Arguments -join ' ')"
    }
}

function Test-DeploymentPackage {
    param([Parameter(Mandatory = $true)][string]$PackageRoot)

    $requiredFiles = @(
        'app.py',
        'requirements.txt',
        '.gitignore',
        '.streamlit\config.toml',
        '.streamlit\secrets.toml.example',
        'models\news_classifier.joblib',
        'src\__init__.py',
        'src\keyword_extractor.py',
        'src\openai_document.py',
        'README.md'
    )
    foreach ($relativePath in $requiredFiles) {
        $fullPath = Join-Path $PackageRoot $relativePath
        if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
            throw "Required deployment file is missing: $relativePath"
        }
    }

    $forbiddenFiles = @('.streamlit\secrets.toml', '.env')
    foreach ($relativePath in $forbiddenFiles) {
        $fullPath = Join-Path $PackageRoot $relativePath
        if (Test-Path -LiteralPath $fullPath -PathType Leaf) {
            throw "Secret file detected: $relativePath. Remove it from this deployment folder before publishing."
        }
    }

    $gitignorePath = Join-Path $PackageRoot '.gitignore'
    $gitignoreText = Get-Content -LiteralPath $gitignorePath -Raw
    if ($gitignoreText -notmatch '(?m)^\.streamlit/secrets\.toml\s*$') {
        throw '.gitignore must exclude .streamlit/secrets.toml before publishing.'
    }
}

function Get-GitExecutable {
    $git = Get-Command git.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($null -eq $git) {
        throw 'Git for Windows was not found. Install Git for Windows, sign in to GitHub, then run DEPLOY_TO_STREAMLIT.bat again.'
    }
    return $git.Source
}

function Ensure-LocalGitIdentity {
    param(
        [Parameter(Mandatory = $true)][string]$GitExecutable,
        [Parameter(Mandatory = $true)][string]$PackageRoot,
        [string]$RequestedName,
        [string]$RequestedEmail
    )

    $safeDirectory = $PackageRoot.Replace('\', '/')
    $name = (& $GitExecutable -c "safe.directory=$safeDirectory" -C $PackageRoot config --get user.name 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($name)) {
        if ([string]::IsNullOrWhiteSpace($RequestedName)) {
            $RequestedName = Read-Host 'Enter the Git display name for this deployment repository'
        }
        if ([string]::IsNullOrWhiteSpace($RequestedName)) {
            throw 'A Git display name is required to create the deployment commit.'
        }
        Invoke-GitCommand -GitExecutable $GitExecutable -PackageRoot $PackageRoot -Arguments @('-C', $PackageRoot, 'config', '--local', 'user.name', $RequestedName)
    }

    $email = (& $GitExecutable -c "safe.directory=$safeDirectory" -C $PackageRoot config --get user.email 2>$null | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($email)) {
        if ([string]::IsNullOrWhiteSpace($RequestedEmail)) {
            $RequestedEmail = Read-Host 'Enter the Git email for this deployment repository'
        }
        if ([string]::IsNullOrWhiteSpace($RequestedEmail)) {
            throw 'A Git email is required to create the deployment commit.'
        }
        Invoke-GitCommand -GitExecutable $GitExecutable -PackageRoot $PackageRoot -Arguments @('-C', $PackageRoot, 'config', '--local', 'user.email', $RequestedEmail)
    }

    Write-Info 'A Git identity is configured for this deployment folder only.'
}

function Test-GitRemoteUrl {
    param([Parameter(Mandatory = $true)][string]$RemoteUrl)

    if ($RemoteUrl -notmatch '^(https://|git@)[^\s]+') {
        throw 'GitHub remote must be an HTTPS or SSH clone URL.'
    }
}

function Invoke-DeploymentPreparation {
    if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
        throw 'The deployment package folder could not be determined.'
    }
    $packageRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
    Test-DeploymentPackage -PackageRoot $packageRoot
    $gitExecutable = Get-GitExecutable

    Write-Info "Deployment package verified: $packageRoot"
    Write-Host 'No real secret file was found. The public first release will keep Cloud Secrets empty.'

    if ($WhatIfPreference) {
        Write-Host '[WHATIF] Would initialize or reuse this folder as a Git repository.'
        Write-Host '[WHATIF] Would stage only the allow-listed deployment files.'
        Write-Host '[WHATIF] Would push only after a GitHub remote is provided.'
        return
    }

    $safeDirectory = $packageRoot.Replace('\', '/')
    $gitMetadataPath = Join-Path $packageRoot '.git'
    $isRepository = Test-Path -LiteralPath $gitMetadataPath
    if (-not $isRepository) {
        Write-Info 'Creating a Git repository in this deployment folder.'
        Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments @('-C', $packageRoot, 'init')
    }
    Ensure-LocalGitIdentity -GitExecutable $gitExecutable -PackageRoot $packageRoot -RequestedName $GitUserName -RequestedEmail $GitUserEmail

    $allowedFiles = @(
        'app.py',
        'requirements.txt',
        '.gitignore',
        '.streamlit/config.toml',
        '.streamlit/secrets.toml.example',
        'models/news_classifier.joblib',
        'src/__init__.py',
        'src/keyword_extractor.py',
        'src/openai_document.py',
        'README.md',
        'DEPLOY_PACKAGE_MANIFEST.txt',
        'DEPLOY_TO_STREAMLIT.bat',
        'DEPLOY_TO_STREAMLIT.ps1'
    )
    Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments (@('-C', $packageRoot, 'add', '--') + $allowedFiles)

    $pendingChanges = (& $gitExecutable -c "safe.directory=$safeDirectory" -C $packageRoot status --porcelain | Select-Object -First 1)
    if (-not [string]::IsNullOrWhiteSpace($pendingChanges)) {
        Write-Info 'Creating the local deployment commit.'
        Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments @('-C', $packageRoot, 'commit', '-m', 'Prepare public Streamlit deployment')
    }
    else {
        Write-Info 'The allow-listed deployment files are already committed.'
    }

    $branch = (& $gitExecutable -c "safe.directory=$safeDirectory" -C $packageRoot branch --show-current | Select-Object -First 1)
    if ([string]::IsNullOrWhiteSpace($branch)) {
        throw 'The Git branch could not be determined after the local commit.'
    }

    if ($PrepareOnly) {
        Write-Host "Preparation completed. Current Git branch: $branch"
        Write-Host 'Run DEPLOY_TO_STREAMLIT.bat again and paste the GitHub clone URL to publish.'
        return
    }

    if ([string]::IsNullOrWhiteSpace($GitHubRemote)) {
        if (-not $NoOpenBrowser) {
            Start-Process 'https://github.com/new'
        }
        Write-Host ''
        Write-Host 'Create an EMPTY GitHub repository in the browser, then copy its HTTPS Clone URL.'
        $GitHubRemote = Read-Host 'Paste the GitHub Clone URL, or press Enter to finish preparation only'
        if ([string]::IsNullOrWhiteSpace($GitHubRemote)) {
            Write-Host 'Preparation completed. No remote was added and nothing was published.'
            return
        }
    }

    Test-GitRemoteUrl -RemoteUrl $GitHubRemote
    $remoteNames = @(& $gitExecutable -c "safe.directory=$safeDirectory" -C $packageRoot remote)
    $hasOrigin = $remoteNames -contains 'origin'
    if (-not $hasOrigin) {
        Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments @('-C', $packageRoot, 'remote', 'add', 'origin', $GitHubRemote)
    }
    else {
        $existingRemote = (& $gitExecutable -c "safe.directory=$safeDirectory" -C $packageRoot remote get-url origin | Select-Object -First 1)
        if ($existingRemote -ne $GitHubRemote) {
            Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments @('-C', $packageRoot, 'remote', 'set-url', 'origin', $GitHubRemote)
        }
    }

    Write-Info "Pushing branch '$branch' to GitHub. Git may ask you to authenticate."
    Invoke-GitCommand -GitExecutable $gitExecutable -PackageRoot $packageRoot -Arguments @('-C', $packageRoot, 'push', '-u', 'origin', $branch)

    Write-Host ''
    Write-Host 'GitHub push completed.' -ForegroundColor Green
    Write-Host 'In Streamlit Community Cloud choose:'
    Write-Host "  Repository: $GitHubRemote"
    Write-Host "  Branch:     $branch"
    Write-Host '  Entry file: app.py'
    Write-Host '  Python:     3.12'
    Write-Host '  Secrets:    leave empty for the safe public first release'

    if (-not $NoOpenBrowser) {
        Start-Process 'https://share.streamlit.io'
    }
}

try {
    Invoke-DeploymentPreparation
}
catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
