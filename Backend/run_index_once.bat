@echo off
cd /d "%~dp0"
echo Running FAQ index (Pinecone upload)...
python index_faq.py
echo.
pause
