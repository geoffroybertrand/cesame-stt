FROM python:3.11-slim

ENV RUNNING_IN_DOCKER=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --no-deps "whisperx @ git+https://github.com/m-bain/whisperX.git"

# Application code
COPY . .

# Create recordings directory
RUN mkdir -p recordings

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
