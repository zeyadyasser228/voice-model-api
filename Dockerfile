FROM python:3.13-slim

# Install FFmpeg (pulls in all required shared libraries automatically)
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Start the application using Railway's PORT variable
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}