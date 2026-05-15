# Windows 本地一键构建脚本（优先使用 MinGW 工具链）
param(
  [string] $CMakeExe = "",
  [string] $Generator = "MinGW Makefiles"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path

$candidates = @(
  $CMakeExe,
  "C:\mingw64\bin\cmake.exe",
  "C:\Program Files\CMake\bin\cmake.exe"
) | Where-Object { $_ -and $_.Trim().Length -gt 0 }

$cmake = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $cmake) {
  Write-Error "未找到 cmake.exe。已尝试: $($candidates -join ', ')。请安装 CMake 或通过 -CMakeExe 指定路径。"
}

Write-Host "Using: $cmake" -ForegroundColor Cyan
& $cmake -S $root -B (Join-Path $root "build") -G $Generator -DAMIO_ENABLE_URING=OFF
& $cmake --build (Join-Path $root "build") --config Release

$exeVs = Join-Path $root "build\Release\run_tests.exe"
$exeSingle = Join-Path $root "build\run_tests.exe"
if (Test-Path $exeVs) {
  Write-Host "Tests: $exeVs" -ForegroundColor Green
  & $exeVs
} elseif (Test-Path $exeSingle) {
  Write-Host "Tests: $exeSingle" -ForegroundColor Green
  & $exeSingle
} else {
  Write-Warning "未找到 run_tests.exe（请确认生成器与输出目录：VS 多为 build\Release\，Ninja 多为 build\）。"
}
