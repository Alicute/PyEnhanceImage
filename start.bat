@echo off
echo 正在启动交互式图像增强实验平台...
cd /d "%~dp0"
uv run python src/main.py
pause