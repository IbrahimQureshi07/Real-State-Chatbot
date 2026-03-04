@echo off
cd /d "%~dp0Backend"
echo Running FAQ index (Pinecone upload)...
python index_faq.py
echo.
pause
