[CmdletBinding()]
param([string]$RuntimeRoot)

$ErrorActionPreference = 'Stop'
$project = Split-Path -Parent $PSScriptRoot
if (-not $RuntimeRoot) { $RuntimeRoot = Join-Path $project '.runtime' }
New-Item -ItemType Directory -Force -Path $RuntimeRoot | Out-Null
$runtime = (Resolve-Path -LiteralPath $RuntimeRoot).Path
$sedCommit = 'db20d93c0ec8b0ce7fae267a10f826dfc7da15f5'
$d2Commit = '247691a1c292654867deb35cabb27f18d75a1363'

function Get-Asset([string]$Name, [string]$Url, [string]$Sha256) {
    $file = Join-Path $runtime $Name
    if (Test-Path -LiteralPath $file -PathType Leaf) {
        if ((Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash -ne $Sha256) {
            throw "Unexpected SHA-256 for $file. Point -RuntimeRoot at a clean directory."
        }
        return
    }
    Write-Host "Downloading $Name"
    $partial = "$file.part"
    curl.exe --fail --location --silent --show-error --retry 2 --connect-timeout 15 --max-time 1800 --output $partial $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed for $Name" }
    if ((Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash -ne $Sha256) {
        throw "Download hash differs for $Name"
    }
    Move-Item -LiteralPath $partial -Destination $file
}

Get-Asset 'SED.zip' "https://codeload.github.com/xb534/SED/zip/$sedCommit" '765793898eab628a18c8214a4357c4323eacc639a85ed7398f593bbde934bf89'
Get-Asset 'detectron2.zip' "https://codeload.github.com/facebookresearch/detectron2/zip/$d2Commit" '03c18cc63041eeefc3662e1c003de11eb570d05a2b1126af9591fc256a348638'
Get-Asset 'sed-b.pth' 'https://drive.usercontent.google.com/download?id=1qx6zGZgSPkF6TObregRz4uzQqSRHrgUw&export=download&confirm=t' '17d43f88ca8ef8ea64037fb4fd414fc103a6e647ae13594bda157b1411a268de'
Get-Asset 'openclip-base.pth' 'https://huggingface.co/laion/CLIP-convnext_base_w_320-laion_aesthetic-s13B-b82K/resolve/main/open_clip_pytorch_model.bin' 'e254e10a70357b2e514afa45841835e5210186f53ca2a95eb13965d54d445f09'
Get-Asset 'ADEChallengeData2016.zip' 'https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip' '7ff1be44964418441f542a7cc1e1a650e7dc0fc275f5d23252bc9bbdbc977b29'
Get-Asset 'val2017.zip' 'http://images.cocodataset.org/zips/val2017.zip' '4f7e2ccb2866ec5041993c9cf2a952bbed69647b115d0f74da7ce8f4bef82f05'
Get-Asset 'stuffthingmaps_trainval2017.zip' 'https://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip' 'da7f213d6a581bd6912a957573d7b4e5061de0d41f733bb27249b0fa981239d7'

$sed = Join-Path $runtime "SED-$sedCommit"
$d2 = Join-Path $runtime "detectron2-$d2Commit"
if (-not (Test-Path -LiteralPath $sed)) {
    Expand-Archive -LiteralPath "$runtime/SED.zip" -DestinationPath $runtime -Force
}
if (-not (Test-Path -LiteralPath $d2)) {
    Expand-Archive -LiteralPath "$runtime/detectron2.zip" -DestinationPath $runtime -Force
}
$patch = Join-Path $project 'patches/adaptive_cer.patch'
git -C $sed apply --reverse --check $patch 2>$null
if ($LASTEXITCODE -ne 0) {
    git -C $sed apply $patch
    if ($LASTEXITCODE -ne 0) { throw 'SED patch could not be applied to the source archive.' }
}
Set-Content -LiteralPath "$runtime/SED_COMMIT.txt" -Value $sedCommit
Set-Content -LiteralPath "$runtime/DETECTRON2_COMMIT.txt" -Value $d2Commit

$venv = Join-Path $runtime 'venv'
$python = Join-Path $venv 'Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    py -3.13 -m venv $venv
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.13 is required for this setup script.' }
}
& $python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
if ($LASTEXITCODE -ne 0) { throw 'PyTorch setup failed.' }
& $python -m pip install --no-cache-dir -r (Join-Path $project 'requirements-windows.txt')
if ($LASTEXITCODE -ne 0) { throw 'Dependency setup failed.' }

# Older timm imports MaxxViT even for ConvNeXt. Python 3.13 requires factory defaults.
$timm = Join-Path $venv 'Lib/site-packages/timm/models/maxxvit.py'
$code = Get-Content -Raw -LiteralPath $timm
if (-not $code.Contains('field(default_factory=MaxxVitConvCfg)')) {
    $code = $code.Replace('from dataclasses import dataclass, replace', 'from dataclasses import dataclass, replace, field')
    $code = $code -replace '(?m)^    conv_cfg: MaxxVitConvCfg = MaxxVitConvCfg\(\)(?=\r?$)', '    conv_cfg: MaxxVitConvCfg = field(default_factory=MaxxVitConvCfg)'
    $code = $code -replace '(?m)^    transformer_cfg: MaxxVitTransformerCfg = MaxxVitTransformerCfg\(\)(?=\r?$)', '    transformer_cfg: MaxxVitTransformerCfg = field(default_factory=MaxxVitTransformerCfg)'
    [IO.File]::WriteAllText($timm, $code, [Text.UTF8Encoding]::new($false))
}
& $python -c "import torch, torchvision, timm; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
if ($LASTEXITCODE -ne 0) { throw 'CUDA or the Python runtime is unavailable.' }

Set-Content -LiteralPath (Join-Path $project '.sed-runtime-path') -Value $runtime
Write-Host 'Ready. Run .\scripts\run.ps1 from the repository.'
