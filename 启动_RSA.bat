@echo off
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [错误] 未检测到 Python。
    echo 请先安装 Python 3.10 或 3.11，并勾选 "Add Python to PATH"。
    echo.
    pause
    exit /b 1
)

python -c "import mne,mne_rsa,numpy,scipy,pandas,matplotlib" >nul 2>nul
if errorlevel 1 (
    echo 首次运行，正在自动安装依赖（可能需要几分钟，请耐心等待）...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [错误] 依赖安装失败。请检查网络连接后重新运行。
        pause
        exit /b 1
    )
    echo.
    echo 依赖安装完成，正在启动程序...
)

python rsa_eeg_gui.py
if errorlevel 1 pause
