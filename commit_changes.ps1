<#
commit_changes.ps1

Helper PowerShell script to initialize a git repo (if missing), stage all
changes, and commit them with a default message. Run this from the project
root (where this file lives).

Usage:
  Open PowerShell in the project folder and run:
    .\commit_changes.ps1

You must have `git` installed and available on PATH for this to work.
#>

try {
    $git = Get-Command git -ErrorAction Stop
} catch {
    Write-Error "git is not installed or not on PATH. Install Git and re-run this script."
    exit 1
}

Push-Location (Split-Path -Parent $MyInvocation.MyCommand.Definition)

if (-not (Test-Path -Path .git)) {
    Write-Host "Initializing new git repository..."
    git init
} else {
    Write-Host "Git repository already initialized."
}

# ensure there is a user.name and user.email set (local scope)
$name = git config user.name
$email = git config user.email
if ([string]::IsNullOrWhiteSpace($name) -or [string]::IsNullOrWhiteSpace($email)) {
    Write-Host "git user.name or user.email not set locally. Please provide values to set locally for this repo."
    $inpName = Read-Host "Enter name for git commits (or leave blank to skip)"
    if ($inpName -and $inpName.Trim() -ne '') { git config user.name "$inpName" }
    $inpEmail = Read-Host "Enter email for git commits (or leave blank to skip)"
    if ($inpEmail -and $inpEmail.Trim() -ne '') { git config user.email "$inpEmail" }
}

git add -A

try {
    git commit -m "Add simulator and integrate into app.py; add simulate_input.py"
    Write-Host "Changes committed."
} catch {
    Write-Warning "Commit failed. If there are no changes to commit, this may be expected."
    Write-Host "Git output: $_"
}

Pop-Location
