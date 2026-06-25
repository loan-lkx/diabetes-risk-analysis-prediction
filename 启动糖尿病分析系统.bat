@echo off
chcp 65001 >nul
title 糖尿病分析与预测系统

echo ============================================
echo   糖尿病分析与预测系统 - 启动中...
echo ============================================
echo.

REM 检查 Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    echo 下载地址：https://www.python.org/downloads/
    pause
    exit /b
)

REM 检查并安装依赖
echo [1/3] 检查依赖...
pip install -r "%~dp0requirements.txt" -q
echo.

echo [2/3] 启动服务...
start "" http://localhost:8501
streamlit run "%~dp0app.py"
