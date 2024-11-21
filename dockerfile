FROM python:3.11-slim

WORKDIR /app

# Install system dependencies including curl for healthcheck
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*
    
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt


ENV PYTHONPATH=/app

COPY . .
EXPOSE 8051

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8052"]