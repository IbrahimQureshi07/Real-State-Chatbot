@echo off
cd /d "%~dp0"
echo Installing dependencies if needed...
pip install -r requirements.txt -q
echo.
echo Starting backend + frontend at http://localhost:8000
echo Open this URL in your browser.
echo.
uvicorn main:app --reload --host 0.0.0.0 --port 8000
