@echo off
chcp 65001 > nul
echo ========================================
echo テクスチャ対応 Image Inpainting Web App
echo ========================================
echo.

REM venvを使用
if exist ".venv\Scripts\activate.bat" (
    echo [INFO] venv環境を使用します（LaMa対応）
    call .venv\Scripts\activate.bat
) else (
    echo [INFO] システムPythonを使用します
    echo [1/2] 依存パッケージをインストール中...
    pip install -r requirements.txt -q
)

echo.
echo [2/2] サーバーを起動中...
echo.
echo ブラウザで http://localhost:5001 を開いてください
echo 終了するには Ctrl+C を押してください
echo.
echo ========================================
python app.py

pause
