# Train the kale-img-1 style LoRA on MOSS-Image via KohakuMossTrainer.
# Windows port of train_lora.sh. Target: single RTX 3090 (24GB, sm86).
# Run from the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\train_lora.ps1
# Extra trainer knobs pass straight through, e.g.:
#   .\scripts\train_lora.ps1 --set LR=2e-4 --set STEPS=2000
#
# Prereqs: Python 3.12 + uv (or just Python 3.12), NVIDIA driver CUDA >= 12.6,
# Visual Studio Build Tools with the C++ workload (kernel compilation).

[CmdletBinding()]
param(
    [string]$TrainerDir = "trainer/KohakuMossTrainer",
    [string]$Weights = "models/MOSS-Image-Base",
    [string]$DataDir = "dataset/images",
    [string]$RunName = "kale-img-1",
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$Extra
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

# 1. Trainer ---------------------------------------------------------------
if (-not (Test-Path $TrainerDir)) {
    git clone https://github.com/KohakuBlueleaf/KohakuMossTrainer $TrainerDir
}
$TrainerAbs = (Resolve-Path $TrainerDir).Path
if (-not (Test-Path "$TrainerAbs/.venv")) {
    & powershell -ExecutionPolicy Bypass -File "$TrainerAbs/scripts/install/install.ps1"
    if ($LASTEXITCODE -ne 0) { throw "trainer install failed" }
}

# Run a command inside the trainer dir with its venv tools on PATH.
$env:PATH = "$TrainerAbs/.venv/Scripts;$env:PATH"
function KOGINE {
    Push-Location $TrainerAbs
    try {
        & $args[0] @($args[1..($args.Count - 1)])
        if ($LASTEXITCODE -ne 0) { throw "$($args[0]) failed (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }
}

KOGINE kohakumoss-doctor

# 2. Weights ---------------------------------------------------------------
# bf16 HF layout (~26 GB). For the smaller int8 ComfyUI layout (~15 GB) see
# trainer docs/guides/weights.md and point -Weights at the ComfyUI models dir.
if (-not (Test-Path $Weights)) {
    KOGINE hf download OpenMOSS-Team/MOSS-Image-Base --local-dir "$Root/$Weights"
}

# 3. Config ----------------------------------------------------------------
KOGINE kohakumoss-config new train `
    --weights "$Root/$Weights" `
    --data "$Root/$DataDir" `
    --name "$RunName"

$Cfg = "configs/user/${RunName}.py"
$CfgData = "configs/user/${RunName}_data.py"

# 4. Cache text embeddings + latents (one-time) ----------------------------
KOGINE kogine run scripts/data/inspect_data.py --config $CfgData
KOGINE kogine run scripts/data/cache_text.py    --config $CfgData
KOGINE kogine run scripts/data/cache_latents.py --config $CfgData

# 5. Train -----------------------------------------------------------------
KOGINE kogine run scripts/train/moss.py --config $Cfg @Extra

Write-Output "Checkpoints + exports: $TrainerDir/runs/moss/$RunName/"
