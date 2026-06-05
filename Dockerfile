FROM python:3.11-slim

# Install system dependencies required by dlib/face_recognition
RUN apt-get update && apt-get install -y --no-install-recommends \
    libx11-6 \
    libx11-dev \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--timeout", "120", "--workers", "2", "output.app:app"]
