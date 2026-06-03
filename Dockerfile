FROM python:3.13-slim

# Install FFmpeg and required libraries
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libavutil57 \
    libavformat59 \
    libavcodec59 \
    libswresample4 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables
ENV PORT=8080

# Start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]