# build_release_windows.ps1
# Run this in Developer PowerShell for VS 2022

param(
    [string]$Version = "v1.0.0"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Building Cardinal Release Package" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Build Cardinal
Write-Host "`n[1/4] Building Cardinal..." -ForegroundColor Yellow
cmake --build build --config Release -j8

if ($LASTEXITCODE -ne 0) {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

# Create release folder
Write-Host "`n[2/4] Creating release package..." -ForegroundColor Yellow
$releaseName = "Cardinal-$Version-windows-x64"
$releaseDir = "release-$releaseName"

Remove-Item -Path $releaseDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Path $releaseDir -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\bin" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\data\memory" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\logs" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\models" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\src\prompts" -Force | Out-Null
New-Item -ItemType Directory -Path "$releaseDir\src\verifier" -Force | Out-Null

# Create models placeholder
New-Item -ItemType File -Path "$releaseDir\models\PLACE_MODEL_HERE.txt" -Force | Out-Null
Set-Content -Path "$releaseDir\models\PLACE_MODEL_HERE.txt" -Value @"
Place your GGUF model file here.

Example:
  models\Qwen_Qwen3.5-4B-Q4_K_M.gguf

Then update config.json to point to: models/your-model.gguf
"@

# Copy files
Copy-Item "build\bin\Release\cardinal.exe" "$releaseDir\bin\"
Copy-Item "build\bin\Release\*.dll" "$releaseDir\bin\"
Copy-Item "vendor\llama.cpp\build\bin\Release\*.dll" "$releaseDir\bin\" -ErrorAction SilentlyContinue
Copy-Item "config.json" "$releaseDir\"
Copy-Item "README.md" "$releaseDir\"
Copy-Item "DOCUMENTATION.md" "$releaseDir\"
Copy-Item "LICENSE" "$releaseDir\"
Copy-Item "src\verifier\cardinal_kb.pl" "$releaseDir\src\verifier\"
Copy-Item "src\prompts\feeling_schema.gbnf" "$releaseDir\src\prompts\"

# Create run.bat for Windows users
Write-Host "`n[3/4] Creating run.bat..." -ForegroundColor Yellow
$runBatContent = @'
@echo off
echo ========================================
echo Cardinal AGI - Starting...
echo ========================================
echo Press Ctrl+C to stop
echo ========================================
echo.

bin\cardinal.exe --serve

pause
'@
Set-Content -Path "$releaseDir\run.bat" -Value $runBatContent -Encoding ASCII

# Create README for the release package
Write-Host "`n[4/4] Creating release README..." -ForegroundColor Yellow
$releaseReadme = @'
# Cardinal AGI - Quick Start

## Windows

1. Place your model file in the `models\` folder
2. Edit `config.json` to set the model path
3. Double-click `run.bat` to start

## Default API Key

The default API key is in `config.json`. Change it before exposing to network.

## Requirements

- Windows 10/11
- NVIDIA GPU recommended (runs on CPU without one)
- Microsoft Visual C++ Redistributable (if not installed, download from Microsoft)

## Need Help?

See README.md for full documentation.
'@
Set-Content -Path "$releaseDir\README_RELEASE.txt" -Value $releaseReadme -Encoding ASCII

# Create ZIP
Write-Host "Creating ZIP archive..." -ForegroundColor Yellow
Compress-Archive -Path "$releaseDir\*" -DestinationPath "$releaseName.zip" -Force

# Cleanup
Remove-Item -Recurse -Force $releaseDir

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "Release created: $releaseName.zip" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nThe ZIP contains:"
Write-Host "  - bin\cardinal.exe"
Write-Host "  - run.bat (double-click to start)"
Write-Host "  - models\ (place your .gguf here)"
Write-Host "  - config.json (edit settings)"