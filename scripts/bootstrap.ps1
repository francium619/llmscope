# bootstrap.ps1 - restore everything needed to build llmscope on a clean Windows box.
#
# Deliberately requires no administrator rights: the toolchain is fetched as
# portable archives rather than installed. Dependencies are pinned to exact
# commits, because both llama.cpp and FTXUI move fast enough that "latest" is not
# a reproducible input.
#
#   powershell -ExecutionPolicy Bypass -File scripts\bootstrap.ps1
#
[CmdletBinding()]
param(
    [string]$ToolsDir = "$env:USERPROFILE\dev-tools",
    [switch]$SkipToolchain,
    [switch]$SkipModel
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$repo = Split-Path -Parent $PSScriptRoot

# Pinned dependency versions. Update deliberately, never incidentally.
$LLAMA_REPO   = 'https://github.com/ggml-org/llama.cpp.git'
$LLAMA_COMMIT = '7b13a8404d7e219c13d1a243e2a21a857a6e99d9'
$FTXUI_REPO   = 'https://github.com/ArthurSonzogni/FTXUI.git'
$FTXUI_COMMIT = '86efdecbaa0a5cf9aadcdd0f63de959b09d252ec'

function Get-PinnedRepo($url, $commit, $dest) {
    if (Test-Path (Join-Path $dest '.git')) {
        $have = (git -C $dest rev-parse HEAD).Trim()
        if ($have -eq $commit) { Write-Host "ok   $(Split-Path -Leaf $dest) @ $($commit.Substring(0,7))"; return }
    }
    if (Test-Path $dest) { Remove-Item -Recurse -Force $dest }
    New-Item -ItemType Directory -Force -Path $dest | Out-Null

    # Fetch only the pinned commit rather than the whole history.
    git -C $dest init -q
    git -C $dest remote add origin $url
    git -C $dest fetch -q --depth 1 origin $commit
    git -C $dest checkout -q FETCH_HEAD
    Write-Host "got  $(Split-Path -Leaf $dest) @ $($commit.Substring(0,7))"
}

Write-Host '== dependencies =='
New-Item -ItemType Directory -Force -Path (Join-Path $repo 'external') | Out-Null
Get-PinnedRepo $LLAMA_REPO $LLAMA_COMMIT (Join-Path $repo 'external\llama.cpp')
Get-PinnedRepo $FTXUI_REPO $FTXUI_COMMIT (Join-Path $repo 'external\FTXUI')

if (-not $SkipToolchain) {
    Write-Host '== portable toolchain (no admin required) =='
    $dl = Join-Path $ToolsDir '_dl'
    New-Item -ItemType Directory -Force -Path $dl | Out-Null

    function Get-LatestAsset($repoName, $pattern) {
        $rel = Invoke-RestMethod "https://api.github.com/repos/$repoName/releases/latest" `
                                 -Headers @{ 'User-Agent' = 'llmscope-bootstrap' }
        $a = $rel.assets | Where-Object { $_.name -like $pattern } | Select-Object -First 1
        if (-not $a) { throw "no asset matching $pattern in $repoName" }
        return $a
    }

    if (-not (Test-Path "$ToolsDir\ninja\ninja.exe")) {
        $a = Get-LatestAsset 'ninja-build/ninja' 'ninja-win.zip'
        Invoke-WebRequest $a.browser_download_url -OutFile "$dl\ninja.zip"
        Expand-Archive "$dl\ninja.zip" -DestinationPath "$ToolsDir\ninja" -Force
    }
    if (-not (Get-ChildItem "$ToolsDir\cmake" -Filter cmake.exe -Recurse -ErrorAction SilentlyContinue)) {
        $a = Get-LatestAsset 'Kitware/CMake' 'cmake-*-windows-x86_64.zip'
        Invoke-WebRequest $a.browser_download_url -OutFile "$dl\cmake.zip"
        Expand-Archive "$dl\cmake.zip" -DestinationPath "$ToolsDir\cmake" -Force
    }
    if (-not (Get-ChildItem "$ToolsDir\w64devkit" -Filter 'g++.exe' -Recurse -ErrorAction SilentlyContinue)) {
        $a = $null
        try { $a = Get-LatestAsset 'skeeto/w64devkit' 'w64devkit-x64-*.exe' } catch {}
        if (-not $a) { $a = Get-LatestAsset 'skeeto/w64devkit' 'w64devkit-*.zip' }
        $out = Join-Path $dl $a.name
        Invoke-WebRequest $a.browser_download_url -OutFile $out
        if ($a.name -like '*.zip') {
            Expand-Archive $out -DestinationPath "$ToolsDir\w64devkit" -Force
        } else {
            & $out -y -o"$ToolsDir\w64devkit" | Out-Null
        }
    }

    foreach ($n in @('ninja.exe', 'cmake.exe', 'g++.exe')) {
        $f = Get-ChildItem $ToolsDir -Filter $n -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($f) { Write-Host "ok   $n -> $($f.FullName)" } else { Write-Warning "MISSING $n" }
    }
}

if (-not $SkipModel) {
    Write-Host '== sample model =='
    $model = Join-Path $repo 'models\qwen2.5-0.5b-instruct-q8_0.gguf'
    if (Test-Path $model) {
        Write-Host 'ok   model already present'
    } else {
        New-Item -ItemType Directory -Force -Path (Join-Path $repo 'models') | Out-Null
        $url = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf?download=true'
        Invoke-WebRequest $url -OutFile $model -MaximumRedirection 5
        Write-Host ('got  model ({0:N0} MB)' -f ((Get-Item $model).Length / 1MB))
    }
}

Write-Host ''
Write-Host 'Next:'
Write-Host "  `$env:PATH = '$ToolsDir\w64devkit\w64devkit\bin;$ToolsDir\ninja;' + (Get-ChildItem '$ToolsDir\cmake' -Filter cmake.exe -Recurse | Select-Object -First 1).DirectoryName + ';' + `$env:PATH"
Write-Host '  cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release'
Write-Host '  cmake --build build'
Write-Host '  .\build\bin\llmscope.exe --demo'
