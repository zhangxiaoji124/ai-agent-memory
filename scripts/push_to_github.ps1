# 推送 main 到 https://github.com/zhangxiaoji124/ai-agent-memory
# 用法: powershell -ExecutionPolicy Bypass -File scripts/push_to_github.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
git config lfs.locksverify false
Write-Host "正在推送到 ai-agent-memory (main) ..."
git push -u ai-agent-memory main
Write-Host "完成: https://github.com/zhangxiaoji124/ai-agent-memory"
