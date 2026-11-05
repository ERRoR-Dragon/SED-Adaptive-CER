[CmdletBinding()]
param(
    [string]$RuntimeRoot,
    [string]$PythonExecutable,
    [string]$Output = 'runs/ade150',
    [int]$Fit = 640,
    [int]$Dev = 160,
    [int]$EvalImages = 2000,
    [int]$TimingImages = 256,
    [int]$Seed = 42,
    [switch]$Float32
)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
$runtimeFile = Join-Path $project '.sed-runtime-path'
if (-not $RuntimeRoot) {
    if ($env:SED_RUNTIME_ROOT) { $RuntimeRoot = $env:SED_RUNTIME_ROOT }
    elseif (Test-Path -LiteralPath $runtimeFile) { $RuntimeRoot = (Get-Content -Raw -LiteralPath $runtimeFile).Trim() }
    else { $RuntimeRoot = Join-Path $project '.runtime' }
}
if (-not (Test-Path -LiteralPath $RuntimeRoot -PathType Container)) {
    throw 'Run scripts/setup.ps1 to prepare the local runtime.'
}
$runtime = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$sed = Join-Path $runtime ('SED-' + (Get-Content -Raw -LiteralPath "$runtime/SED_COMMIT.txt").Trim())
$d2 = Join-Path $runtime ('detectron2-' + (Get-Content -Raw -LiteralPath "$runtime/DETECTRON2_COMMIT.txt").Trim())
if (-not $PythonExecutable) {
    $venvPython = Join-Path $runtime 'venv/Scripts/python.exe'
    if (Test-Path -LiteralPath $venvPython) { $PythonExecutable = $venvPython }
    else { $PythonExecutable = (py -3.13 -c 'import sys; print(sys.executable)').Trim() }
}
git -C $sed apply --reverse --check (Join-Path $project 'patches/adaptive_cer.patch')
if ($LASTEXITCODE -ne 0) { throw 'Apply the current adaptive CER patch to the staged SED source before running.' }

$originalPath = $env:PATH
$originalPythonPath = $env:PYTHONPATH
$pythonPaths = @("$project/src", "$runtime/site", "$runtime/site/win32",
                 "$runtime/site/win32/lib", "$runtime/site/pywin32_system32",
                 $d2, "$sed/open_clip/src")
$env:PYTHONPATH = $pythonPaths -join [IO.Path]::PathSeparator
$env:PATH = "$runtime/site/pywin32_system32;$originalPath"
Push-Location $project
try {
    $runArgs = @(
        '-u', '-m', 'sed_adaptive_cer.experiment', '--sed-root', $sed,
        '--weights', "$runtime/sed-b.pth", '--openclip-weights', "$runtime/openclip-base.pth",
        '--ade-zip', "$runtime/ADEChallengeData2016.zip", '--coco-images', "$runtime/val2017.zip",
        '--coco-masks', "$runtime/stuffthingmaps_trainval2017.zip",
        '--output', $Output, '--fit', "$Fit", '--dev', "$Dev",
        '--eval-images', "$EvalImages", '--timing-images', "$TimingImages", '--seed', "$Seed"
    )
    if ($Float32) { $runArgs += '--fp32' }
    & $PythonExecutable @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Benchmark exited with code $LASTEXITCODE. Rerun the same command to resume completed rows." }
} finally {
    Pop-Location
    $env:PATH = $originalPath
    $env:PYTHONPATH = $originalPythonPath
}
