# Use Python 3.11 slim (smaller image; sentence-transformers needs torch)
FROM python:3.11-slim

WORKDIR /app

# Copy project
COPY Backend /app/Backend
COPY Frontend /app/Frontend

# Install backend deps (from Backend/requirements.txt)
RUN pip install --no-cache-dir -r Backend/requirements.txt

# Run from Backend dir; main.py serves Frontend from ../Frontend
WORKDIR /app/Backend
EXPOSE 8000
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
